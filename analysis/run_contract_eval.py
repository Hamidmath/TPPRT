"""Run the EVALUATION_CONTRACT against the current popularity matrices.

Configurations evaluated:
    1. No diffusion (raw E_b), eps = 1e-6
    2. Diffused (E_b tilde), eps = 1e-6
    3. Diffused (E_b tilde), eps = 0 (skip cells with truth = 0)

Per the contract:
    - 84 random target bins (12 per weekday, fixed seed 42).
    - Each bin t must have a sibling t' = t + 7 days in the matrix.
    - Three comparisons per (bin, config):
        dd : truth=E_b(t'),  pred=E_b(t)       (week-to-week baseline)
        fc : truth=E_b(t'),  pred=chain(E_b(t)) (forecast)
        rec: truth=E_b(t),   pred=chain(E_b(t)) (reconstruction)
    - Aggregate: Top-100 MRE and Overall MRE, mean over the 84 bins.
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
from core.pagerank import build_phase_kernels, power_iteration, PARAMS

# Parameters set by this script (override defaults)
BETA = 0.1019
RHO = 0.1014
TOL = 1e-7
MAX_ITER = 300
SEED = 42
BINS_PER_WEEKDAY = 12

RAW_NPZ = str(config.DATA_DIR / "popularity_results_osm.npz")
SMOOTH_NPZ = str(config.DATA_DIR / "popularity_results_smoothed_osm.npz")
GRAPH_JSON = str(config.GRAPH_FILE)


def sample_target_bins(times, seed=SEED, bins_per_weekday=BINS_PER_WEEKDAY):
    """Pick `bins_per_weekday` per weekday such that bin t and t+7days
    both exist in `times`. Returns a list of (t_str, tprime_str) pairs."""
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
            print(f"  [warn] no eligible bins for weekday {wd}")
            continue
        n = min(bins_per_weekday, len(eligible))
        pairs.extend(rng.sample(eligible, n))
    return pairs


def load_prior(npz_path, N, links, lid_to_idx):
    """Return (matrix_rows: dict {time_str -> np.ndarray of size N},
    times_list)."""
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


def mre(truth, pred, eps, top_k=100):
    if eps == 0.0:
        # skip cells with truth==0
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
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    print(f"  N = {N}")

    print("Building P_up, P_down kernels...")
    P_up_cs, P_down_cs = build_phase_kernels(graph)

    print("Loading raw prior (popularity_results_osm.npz)...")
    raw_rows, raw_times = load_prior(RAW_NPZ, N, links, lid_to_idx)
    print(f"  {len(raw_rows):,} bins, span {raw_times[0]} .. {raw_times[-1]}")

    print("Loading diffused prior (popularity_results_smoothed_osm.npz)...")
    diff_rows, diff_times = load_prior(SMOOTH_NPZ, N, links, lid_to_idx)
    print(f"  {len(diff_rows):,} bins, span {diff_times[0]} .. {diff_times[-1]}")

    # Sample bins: contract requires t and t+7days both in matrix.
    # Use the smoothed-prior time index (dense, every bin) for sampling.
    pairs = sample_target_bins(diff_times)
    print(f"\nSampled {len(pairs)} bin pairs (12 per weekday, seed={SEED})")

    configs = [
        ("raw, eps=1e-6",      raw_rows,  1e-6),
        ("diff, eps=1e-6",     diff_rows, 1e-6),
        ("diff, eps=0",        diff_rows, 0.0),
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

            # chain
            v_up, v_down, _ = power_iteration(
                P_up_cs, P_down_cs, E_t, BETA, RHO, TOL, MAX_ITER
            )
            v_chain = v_up + v_down
            v_chain /= v_chain.sum() if v_chain.sum() > 0 else 1.0

            # dd: truth=E_tprime, pred=E_t
            ov, top = mre(E_tprime, E_t, eps)
            agg["dd_top"].append(top)
            agg["dd_overall"].append(ov)
            # fc: truth=E_tprime, pred=v_chain
            ov, top = mre(E_tprime, v_chain, eps)
            agg["fc_top"].append(top)
            agg["fc_overall"].append(ov)
            # rec: truth=E_t, pred=v_chain
            ov, top = mre(E_t, v_chain, eps)
            agg["rec_top"].append(top)
            agg["rec_overall"].append(ov)
            if (i + 1) % 20 == 0:
                print(f"  ... {i + 1}/{len(pairs)} bins  "
                      f"({time.time() - t0:.0f}s)")
        results[cfg_name] = {k: {"mean": float(np.mean(v)),
                                  "median": float(np.median(v)),
                                  "std": float(np.std(v))}
                              for k, v in agg.items()}
        print(f"  Done in {time.time() - t0:.0f}s")

    # Print table
    print("\n" + "=" * 100)
    print("  Contract evaluation summary (mean MRE across 84 bins)")
    print("=" * 100)
    print(f"  beta = {BETA}, rho = {RHO}")
    print()
    print(f"  {'Comparison':40s}  {'No diff (eps=1e-6)':>22s}  "
          f"{'Diff (eps=1e-6)':>22s}  {'Diff (eps=0)':>22s}")
    print(f"  {'':40s}  {'T-100':>10s} {'Ov':>10s}  "
          f"{'T-100':>10s} {'Ov':>10s}  "
          f"{'T-100':>10s} {'Ov':>10s}")
    print("  " + "-" * 100)
    for label, top_k, ov_k in [
        ("1. dataset -> dataset (next week)", "dd_top", "dd_overall"),
        ("2. dataset -> model (forecast)   ", "fc_top", "fc_overall"),
        ("3. dataset -> model (reconstruct)", "rec_top", "rec_overall"),
    ]:
        line = f"  {label:40s}"
        for cfg_name, _, _ in configs:
            r = results[cfg_name]
            line += (f"  {r[top_k]['mean']:>10.3f} {r[ov_k]['mean']:>10.3f}")
        print(line)

    out_path = config.DATA_DIR / "../results/contract_eval/contract_eval_noise4.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"beta": BETA, "rho": RHO, "eps_configs": [c[0] for c in configs],
                   "results": results}, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
