"""How many of N random bins hit the 300-iteration cap for the
single-phase PageRank chain at the calibrated damping d = 0.9492?

Tolerance is the same 1e-7 used by the contract eval. The chain
returns (v, k+1) from the loop; hitting the cap means
k+1 == MAX_ITER, i.e. the L1 change never dropped below 1e-7
within MAX_ITER iterations.
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

DAMPING = 0.9492
TOL = 1e-7
MAX_ITER = 300
N_SAMPLE = 840
SEED = 42

RAW_NPZ = str(config.DATA_DIR / "popularity_results_osm.npz")
SMOOTH_NPZ = str(config.DATA_DIR / "popularity_results_smoothed_osm.npz")
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
    P_cs = csr_matrix((data, (row, col)), shape=(N, N))
    return P_cs, links, lid_to_idx


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


def main():
    print(f"max_iter = {MAX_ITER},  tol = {TOL},  d = {DAMPING}")
    print(f"N_SAMPLE = {N_SAMPLE} bins,  seed = {SEED}\n")

    print("Loading graph...")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    P_cs, links, lid_to_idx = build_pagerank_matrix(graph)
    N = len(links)
    print(f"  N = {N}")

    # Use the raw prior (per contract). Diffused gives same iteration count
    # qualitatively since the damping rate is what governs convergence.
    print(f"Loading raw prior: {RAW_NPZ}")
    rows, times = load_prior(RAW_NPZ, N, links, lid_to_idx)
    print(f"  {len(rows):,} bins")

    rng = random.Random(SEED)
    sample = rng.sample(times, min(N_SAMPLE, len(times)))
    print(f"Sampled {len(sample)} random bins\n")

    iters = []
    final_diffs = []
    hits = 0
    converged_iters = []
    t0 = time.time()
    for i, t in enumerate(sample):
        E_t = rows[t]
        v, k, diff, status = single_phase_iteration(
            P_cs, E_t, DAMPING, TOL, MAX_ITER
        )
        iters.append(k)
        final_diffs.append(diff)
        if status == "hit_cap":
            hits += 1
        else:
            converged_iters.append(k)
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(sample)}  "
                  f"({time.time() - t0:.0f}s)  hits={hits}")

    iters = np.asarray(iters)
    final_diffs = np.asarray(final_diffs)
    converged_iters = np.asarray(converged_iters) if converged_iters else np.array([])

    print()
    print("=" * 70)
    print(f"  RESULTS: {N_SAMPLE} bins, single-phase chain at d = {DAMPING}")
    print("=" * 70)
    print(f"  bins hit cap (300 iter):  {hits} / {len(sample)}  "
          f"({100*hits/len(sample):.2f}%)")
    if hits:
        diffs_at_cap = final_diffs[iters == MAX_ITER]
        print(f"    final L1 diff at cap:   "
              f"min={diffs_at_cap.min():.2e}  "
              f"median={np.median(diffs_at_cap):.2e}  "
              f"max={diffs_at_cap.max():.2e}")
    if len(converged_iters):
        print(f"  bins that converged:     {len(converged_iters)} "
              f"({100*len(converged_iters)/len(sample):.2f}%)")
        print(f"    iterations to converge: "
              f"min={converged_iters.min()}  "
              f"median={int(np.median(converged_iters))}  "
              f"95th={int(np.percentile(converged_iters, 95))}  "
              f"max={converged_iters.max()}")
    print(f"\n  wall time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
