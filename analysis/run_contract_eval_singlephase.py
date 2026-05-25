"""Single-phase PageRank baseline under the EVALUATION_CONTRACT.

Mirrors analysis/run_contract_eval.py exactly (same 84 bins, same seed=42,
same three comparisons, same three eps configurations), but the chain is a
plain PageRank with damping d on the row-stochastic graph and
teleportation prior E_b:

    v_{k+1} = (1 - d) * E_b + d * P^T v_k

The damping value comes from the single-geometric calibration on all-K
trip lengths (E[K_total] = 18.68 -> p = 1 / (1 + E[K]) = 0.0508), so
    d = 1 - p = 0.9492.

This is the single-phase analogue of the calibrated (beta, rho) used
by the two-phase contract evaluation.
"""
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz

P_GEOM = 0.0508       # from single-geometric fit on K_total (all-K)
DAMPING = 1.0 - P_GEOM  # 0.9492
TOL = 1e-7
MAX_ITER = 300
SEED = 42
BINS_PER_WEEKDAY = 12

RAW_NPZ = str(config.DATA_DIR / "popularity_results_osm.npz")
SMOOTH_NPZ = str(config.DATA_DIR / "popularity_results_smoothed_osm.npz")
GRAPH_JSON = str(config.GRAPH_FILE)


def build_pagerank_matrix(graph):
    """Column-stochastic P (so P @ v moves probability mass along outgoing
    edges, weighted uniformly). Identical to how the two-phase code builds
    P_up_cs / P_down_cs but without speed-lane weights."""
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    adj = graph.get("adjacency", {})
    row, col, data = [], [], []
    for i, lid in enumerate(links):
        out_lids = [ol for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not out_lids:
            continue
        w = 1.0 / len(out_lids)
        for ol in out_lids:
            # row-stochastic: row i, col j, weight 1/d_out(i)
            row.append(lid_to_idx[ol])  # col-stochastic == transpose
            col.append(i)
            data.append(w)
    P_cs = csr_matrix((data, (row, col)), shape=(N, N))
    return P_cs, links, lid_to_idx


def sample_target_bins(times, seed=SEED, bins_per_weekday=BINS_PER_WEEKDAY):
    """Same sampling as the two-phase contract eval."""
    time_set = set(times)
    by_weekday = {i: [] for i in range(7)}
    for t in times:
        dt = datetime.fromisoformat(t)
        sibling = (dt + timedelta(days=7)).isoformat(sep=" ")
        if sibling in time_set:
            by_weekday[dt.weekday()].append((t, sibling))
    rng = random.Random(seed)
    pairs = []
    for wd in range(7):
        eligible = by_weekday[wd]
        if not eligible:
            continue
        n = min(bins_per_weekday, len(eligible))
        pairs.extend(rng.sample(eligible, n))
    return pairs


def load_prior(npz_path, N, links, lid_to_idx):
    print(f"  loading {npz_path}")
    b = load_popularity_npz(npz_path)
    matrix = b["matrix"]
    times = b["times"]
    pop_link_ids = b["link_ids"]
    proj = np.fromiter(
        (lid_to_idx.get(lid, -1) for lid in pop_link_ids),
        dtype=np.int64, count=len(pop_link_ids),
    )
    valid = proj >= 0
    rows = {}
    for i, t in enumerate(times):
        row = matrix.getrow(i).toarray().ravel()
        E = np.zeros(N)
        np.add.at(E, proj[valid], row[valid])
        s = E.sum()
        if s > 0:
            E /= s
        rows[str(t)] = E
    return rows, [str(t) for t in times]


def single_phase_iteration(P_cs, E_b, d, tol, max_iter):
    """Standard PageRank iteration with teleportation prior E_b."""
    v = E_b.copy()
    for k in range(max_iter):
        v_new = (1.0 - d) * E_b + d * (P_cs @ v)
        s = float(v_new.sum())
        if s > 0:
            v_new /= s
        diff = float(np.abs(v_new - v).sum())
        v = v_new
        if diff < tol:
            return v, k + 1
    return v, max_iter


def mre(truth, pred, eps, top_k=100):
    if eps == 0.0:
        mask = truth > 0
        if mask.sum() == 0:
            return 0.0, 0.0
        rel = np.abs(truth[mask] - pred[mask]) / truth[mask]
        overall = float(np.mean(rel))
        top = np.argsort(truth)[::-1][:top_k]
        truth_safe = np.where(truth > 0, truth, 1.0)
        top_mre = float(np.mean(np.abs(truth[top] - pred[top]) / truth_safe[top]))
    else:
        rel = np.abs(truth - pred) / (truth + eps)
        overall = float(np.mean(rel))
        top = np.argsort(truth)[::-1][:top_k]
        top_mre = float(np.mean(rel[top]))
    return overall, top_mre


def main():
    print("Loading graph...")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    P_cs, links, lid_to_idx = build_pagerank_matrix(graph)
    N = len(links)
    print(f"  N = {N}, P_cs nnz = {P_cs.nnz:,}")

    print("Loading raw prior (popularity_results_osm.npz)...")
    raw_rows, raw_times = load_prior(RAW_NPZ, N, links, lid_to_idx)
    print(f"  {len(raw_rows):,} bins")

    print("Loading diffused prior (popularity_results_smoothed_osm.npz)...")
    diff_rows, diff_times = load_prior(SMOOTH_NPZ, N, links, lid_to_idx)
    print(f"  {len(diff_rows):,} bins")

    pairs = sample_target_bins(diff_times)
    print(f"\nSampled {len(pairs)} bin pairs (12 per weekday, seed={SEED})")
    print(f"Single-phase PageRank: p_geom = {P_GEOM}, damping d = {DAMPING}")

    configs = [
        ("raw, eps=1e-6",   raw_rows,  1e-6),
        ("diff, eps=1e-6",  diff_rows, 1e-6),
        ("diff, eps=0",     diff_rows, 0.0),
    ]
    results = {}
    for cfg_name, rows, eps in configs:
        print(f"\n== Config: {cfg_name} ==")
        agg = {"dd_top": [], "dd_overall": [],
               "fc_top": [], "fc_overall": [],
               "rec_top": [], "rec_overall": []}
        t0 = time.time()
        for i, (t, tprime) in enumerate(pairs):
            if t not in rows or tprime not in rows:
                continue
            E_t = rows[t]
            E_tprime = rows[tprime]
            v_pr, _ = single_phase_iteration(
                P_cs, E_t, DAMPING, TOL, MAX_ITER
            )
            ov, top = mre(E_tprime, E_t, eps)
            agg["dd_top"].append(top); agg["dd_overall"].append(ov)
            ov, top = mre(E_tprime, v_pr, eps)
            agg["fc_top"].append(top); agg["fc_overall"].append(ov)
            ov, top = mre(E_t, v_pr, eps)
            agg["rec_top"].append(top); agg["rec_overall"].append(ov)
            if (i + 1) % 20 == 0:
                print(f"  ... {i + 1}/{len(pairs)} bins  "
                      f"({time.time() - t0:.0f}s)")
        results[cfg_name] = {
            k: {"mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "std": float(np.std(v))}
            for k, v in agg.items()
        }
        print(f"  Done in {time.time() - t0:.0f}s")

    print("\n" + "=" * 100)
    print("  Single-phase PageRank baseline (mean MRE across 84 bins)")
    print("=" * 100)
    print(f"  damping d = {DAMPING}   (p_geom = {P_GEOM})\n")
    print(f"  {'Comparison':40s}  {'No diff (eps=1e-6)':>22s}  "
          f"{'Diff (eps=1e-6)':>22s}  {'Diff (eps=0)':>22s}")
    print(f"  {'':40s}  {'T-100':>10s} {'Ov':>10s}  "
          f"{'T-100':>10s} {'Ov':>10s}  {'T-100':>10s} {'Ov':>10s}")
    print("  " + "-" * 100)
    for label, top_k, ov_k in [
        ("1. dataset -> dataset (next week)", "dd_top", "dd_overall"),
        ("2. dataset -> model (forecast)   ", "fc_top", "fc_overall"),
        ("3. dataset -> model (reconstruct)", "rec_top", "rec_overall"),
    ]:
        line = f"  {label:40s}"
        for cfg_name, _, _ in configs:
            r = results[cfg_name]
            line += f"  {r[top_k]['mean']:>10.3f} {r[ov_k]['mean']:>10.3f}"
        print(line)

    out_path = config.DATA_DIR / "../results/contract_eval/contract_eval_singlephase.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"damping": DAMPING, "p_geom": P_GEOM,
                   "eps_configs": [c[0] for c in configs],
                   "results": results}, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
