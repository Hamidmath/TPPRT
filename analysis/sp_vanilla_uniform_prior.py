"""Vanilla single-phase PageRank with a UNIFORM teleportation prior.

  v_{k+1} = alpha * u + (1 - alpha) * P_cs v_k,   u_i = 1/N

P_cs is the column-stochastic, uniform-out-edge transition on the
99,716-link road graph (no road-type weights). The stationary v* does
not depend on the time bin t, so it is computed once.

We then evaluate the same 840 (t, t+7d) calendar pairs used by Table 10
(seed=7, 120 per weekday). For each pair, the target is the diffused
popularity at t+7d and the prediction is v* (constant across pairs).
We report Overall MRE averaged over the 840 pairs.

Output: results/sp_vanilla_uniform.json
"""
import json
import os
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

INPUTS  = Path(os.environ.get("TPPR_INPUTS",  str(config.DATA_DIR)))
RESULTS = Path(os.environ.get("TPPR_RESULTS", str(config.PROJECT_ROOT / "results")))
RESULTS.mkdir(parents=True, exist_ok=True)

SMOOTH_NPZ = str(INPUTS / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON = str(INPUTS / "city_graph_full.json")
OUT = RESULTS / "sp_vanilla_uniform.json"

EPS = 1e-6
SEED = 7
BPW = 120
TOL, MAX_ITER = 1e-9, 500
ALPHAS = [0.15, 0.05]


def build_P_cs(graph):
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    adj = graph.get("adjacency", {})
    rows, cols, data = [], [], []
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ:
            continue
        w = 1.0 / len(succ)
        for j in succ:
            rows.append(j); cols.append(i); data.append(w)
    P_cs = csr_matrix((data, (rows, cols)),
                       shape=(N, N), dtype=np.float64)
    return P_cs, links, lid_to_idx


def sp_iter(P_cs, prior, alpha, tol=TOL, max_iter=MAX_ITER):
    v = prior.copy()
    one_minus_a = 1.0 - alpha
    for _ in range(max_iter):
        v_new = alpha * prior + one_minus_a * (P_cs @ v)
        s = v_new.sum()
        if s > 0:
            v_new /= s
        if float(np.abs(v_new - v).sum()) < tol:
            return v_new
        v = v_new
    return v


def load_diffused(npz_path, N, lid_to_idx):
    b = load_popularity_npz(npz_path)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_ids = [str(lid) for lid in b["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_ids], dtype=np.int64)
    valid = proj >= 0
    rows = {}
    for ti, t in enumerate(times):
        row = M.getrow(ti).toarray().ravel().astype(np.float64)
        E = np.zeros(N, dtype=np.float64)
        np.add.at(E, proj[valid], row[valid])
        s = E.sum()
        if s > 0:
            E /= s
        rows[t] = E
    return rows


def sample_pairs(times, seed=SEED, bpw=BPW):
    rng = random.Random(seed)
    by_wd = {i: [] for i in range(7)}
    tset = set(times)
    for t in times:
        dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        sib = (dt + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        if sib in tset:
            by_wd[dt.weekday()].append(t)
    pairs = []
    for wd in range(7):
        pool = by_wd[wd]
        rng.shuffle(pool)
        for t in pool[:bpw]:
            dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
            sib = (dt + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
            pairs.append((t, sib))
    return pairs


def main():
    t0 = time.time()
    print("[start] SP vanilla with uniform teleportation prior")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    P_cs, links, lid_to_idx = build_P_cs(graph)
    N = len(links)
    print(f"  N = {N:,}, P_cs nnz = {P_cs.nnz:,}")

    u = np.full(N, 1.0 / N, dtype=np.float64)

    diff_rows = load_diffused(SMOOTH_NPZ, N, lid_to_idx)
    pairs = sample_pairs(sorted(diff_rows.keys()))
    print(f"  pairs: {len(pairs)}")

    results = {}
    for alpha in ALPHAS:
        print(f"\n  alpha = {alpha}")
        v_star = sp_iter(P_cs, u, alpha)
        per_pair = []
        for t, sib in pairs:
            tgt = diff_rows[sib]
            if tgt.sum() == 0:
                continue
            ov = float(np.mean(np.abs(tgt - v_star) / (tgt + EPS)))
            per_pair.append(ov)
        results[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "overall_mean":   float(np.mean(per_pair)),
            "overall_median": float(np.median(per_pair)),
            "n_pairs": len(per_pair),
        }
        r = results[f"alpha_{alpha}"]
        print(f"    Overall MRE mean   = {r['overall_mean']:.4f}")
        print(f"    Overall MRE median = {r['overall_median']:.4f}")
        print(f"    n pairs used       = {r['n_pairs']}")

    summary = {
        "model": "vanilla SP with uniform teleportation prior u=1/N",
        "n_pairs": len(pairs), "seed": SEED, "bpw": BPW,
        "alphas": ALPHAS,
        "graph": "city_graph_full.json",
        "target": "popularity_results_smoothed_osm_gamma020.npz",
        "results": results,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {time.time() - t0:.0f}s saved {OUT}")


if __name__ == "__main__":
    main()
