"""Second 840-bin convergence + accuracy check for the single-phase
PageRank baseline. Different seed (seed=7) than the first run and the
contract (seed=42), so the bin set is disjoint-ish.

Records, for every sampled (t, t+7day) pair:
    - iteration count and final L1 diff (convergence)
    - Top-100 and Overall MRE for the three contract comparisons
      (dataset-to-dataset, forecast, reconstruction)
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

P_GEOM = 0.0508
DAMPING = 1.0 - P_GEOM  # 0.9492
TOL = 1e-7
MAX_ITER = 300
N_SAMPLE = 840
BINS_PER_WEEKDAY = 120  # 840 / 7
SEED = 7  # different from the contract eval seed of 42

RAW_NPZ = str(config.DATA_DIR / "popularity_results_osm.npz")
SMOOTH_NPZ = str(config.DATA_DIR / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON = str(config.GRAPH_FILE)


def build_pagerank_matrix(graph):
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
            row.append(lid_to_idx[ol])
            col.append(i)
            data.append(w)
    return csr_matrix((data, (row, col)), shape=(N, N)), links, lid_to_idx


def sample_target_bins(times, seed, bins_per_weekday):
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
        n = min(bins_per_weekday, len(eligible))
        pairs.extend(rng.sample(eligible, n))
    return pairs


def load_prior(npz_path, N, links, lid_to_idx):
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
    v = E_b.copy()
    last_diff = None
    for k in range(max_iter):
        v_new = (1.0 - d) * E_b + d * (P_cs @ v)
        s = float(v_new.sum())
        if s > 0:
            v_new /= s
        diff = float(np.abs(v_new - v).sum())
        v = v_new
        if diff < tol:
            return v, k + 1, diff, "converged"
        last_diff = diff
    return v, max_iter, last_diff, "hit_cap"


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
    print(f"seed = {SEED},  N_SAMPLE = {N_SAMPLE},  "
          f"max_iter = {MAX_ITER},  tol = {TOL}")
    print(f"single-phase d = {DAMPING}  (p_geom = {P_GEOM})\n")

    print("Loading graph and matrices...")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    P_cs, links, lid_to_idx = build_pagerank_matrix(graph)
    N = len(links)
    print(f"  N = {N},  P_cs nnz = {P_cs.nnz:,}")

    raw_rows, raw_times = load_prior(RAW_NPZ, N, links, lid_to_idx)
    diff_rows, _ = load_prior(SMOOTH_NPZ, N, links, lid_to_idx)
    print(f"  raw bins = {len(raw_rows):,},  diff bins = {len(diff_rows):,}")

    pairs = sample_target_bins(raw_times, seed=SEED,
                                bins_per_weekday=BINS_PER_WEEKDAY)
    print(f"  Sampled {len(pairs)} (t, t+7days) pairs\n")

    configs = [
        ("raw, eps=1e-6",   raw_rows,  1e-6),
        ("diff, eps=1e-6",  diff_rows, 1e-6),
        ("diff, eps=0",     diff_rows, 0.0),
    ]

    iters_all = []
    hit_cap_all = 0
    final_diffs_all = []
    results = {}

    for cfg_name, rows, eps in configs:
        print(f"== Config: {cfg_name} ==")
        agg = {"dd_top": [], "dd_overall": [],
               "fc_top": [], "fc_overall": [],
               "rec_top": [], "rec_overall": []}
        t0 = time.time()
        local_iters, local_hits = [], 0
        for i, (t, tprime) in enumerate(pairs):
            if t not in rows or tprime not in rows:
                continue
            E_t = rows[t]
            E_tprime = rows[tprime]
            v_pr, k, diff, status = single_phase_iteration(
                P_cs, E_t, DAMPING, TOL, MAX_ITER
            )
            local_iters.append(k)
            if status == "hit_cap":
                local_hits += 1
            ov, top = mre(E_tprime, E_t, eps)
            agg["dd_top"].append(top); agg["dd_overall"].append(ov)
            ov, top = mre(E_tprime, v_pr, eps)
            agg["fc_top"].append(top); agg["fc_overall"].append(ov)
            ov, top = mre(E_t, v_pr, eps)
            agg["rec_top"].append(top); agg["rec_overall"].append(ov)
            if (i + 1) % 200 == 0:
                print(f"  ... {i + 1}/{len(pairs)}  "
                      f"({time.time() - t0:.0f}s)  hits={local_hits}")
        print(f"  Done in {time.time() - t0:.0f}s, "
              f"hit-cap count = {local_hits}\n")
        if cfg_name == "raw, eps=1e-6":
            iters_all = local_iters
            hit_cap_all = local_hits
        results[cfg_name] = {
            k: {"mean": float(np.mean(v)),
                "median": float(np.median(v))}
            for k, v in agg.items()
        }

    iters_arr = np.asarray(iters_all)
    converged = iters_arr < MAX_ITER

    print("=" * 100)
    print(f"  CONVERGENCE on {len(iters_arr)} bins  "
          f"(seed = {SEED}, single-phase, d = {DAMPING})")
    print("=" * 100)
    print(f"  bins hit 300-iter cap:    {hit_cap_all} / {len(iters_arr)}  "
          f"({100*hit_cap_all/len(iters_arr):.2f}%)")
    if converged.any():
        ci = iters_arr[converged]
        print(f"  bins converged in <300:   {converged.sum()} "
              f"({100*converged.mean():.2f}%)")
        print(f"    iterations to converge: "
              f"min={ci.min()}  median={int(np.median(ci))}  "
              f"95th={int(np.percentile(ci, 95))}  max={ci.max()}")

    print()
    print("=" * 100)
    print(f"  CONTRACT ACCURACY on {len(iters_arr)} bins, single-phase")
    print("=" * 100)
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


if __name__ == "__main__":
    main()
