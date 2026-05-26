"""Old-form Two-Phase PageRank: vanilla evaluation.

Old form:
  M_{2N} = [(1-beta) P     beta I  ]      (row-stochastic, no rho)
           [   0           P       ]
  v^{(k+1)} = (1 - alpha) * M^T v^{(k)} + alpha * E_{2N}
  E_{2N}     = concat(E_b, 0_N)
  Dangling-mass correction: v += (1 - sum(v)) * E_{2N} each step

This script evaluates the chain (no tuning) on the same 840 pairs as
the new-form TP rows. Sweeps (alpha, beta) over a small grid so we can
read off the vanilla numbers for the Table 10 ladder.

Output: results/tp_old_form_eval.json
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
OUT = RESULTS / "tp_old_form_eval.json"

EPS = 1e-6
TOP_K = 100
SEED = 7
BPW = 120
TOL, MAX_ITER = 1e-6, 200

ALPHAS = [0.05, 0.15]
BETAS  = [0.05, 0.10, 0.15]


def build_P_row(graph, links, lid_to_idx):
    """Row-stochastic transition: P[i, j] = prob of going from i to j."""
    adj = graph.get("adjacency", {})
    rows, cols, data = [], [], []
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ:
            continue
        w = 1.0 / len(succ)
        for j in succ:
            rows.append(i); cols.append(j); data.append(w)
    return csr_matrix((data, (rows, cols)),
                       shape=(len(links), len(links)), dtype=np.float64)


def tp_old_iter(P, E_b, alpha, beta):
    """Iterate the old-form two-phase chain.

    v_up_{k+1}   = (1-alpha)*(1-beta) * P^T v_up_k + alpha * E_b
    v_down_{k+1} = (1-alpha)*beta * v_up_k + (1-alpha) * P^T v_down_k
    """
    N = E_b.shape[0]
    Pt = P.T
    v_up = E_b.copy()
    v_down = np.zeros(N)
    d = 1.0 - alpha
    for _ in range(MAX_ITER):
        v_up_n   = d * (1.0 - beta) * (Pt @ v_up)   + alpha * E_b
        v_down_n = d * beta * v_up + d * (Pt @ v_down)
        # Dangling-mass correction: any mass lost via dead-end columns goes
        # back through the teleportation distribution (E_b on up side only).
        s = float(v_up_n.sum() + v_down_n.sum())
        if s < 1.0:
            v_up_n += (1.0 - s) * E_b
        if float(np.abs(v_up_n - v_up).sum() + np.abs(v_down_n - v_down).sum()) < TOL:
            v_up, v_down = v_up_n, v_down_n
            break
        v_up, v_down = v_up_n, v_down_n
    return v_up + v_down


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
    print(f"[start] OLD-form TP eval, alphas={ALPHAS}, betas={BETAS}")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    P = build_P_row(graph, links, lid_to_idx)
    print(f"  N = {N:,}")
    rows = load_diffused(SMOOTH_NPZ, N, lid_to_idx)
    pairs = sample_pairs(sorted(rows.keys()))
    print(f"  pairs: {len(pairs)}")

    results = {}
    for alpha in ALPHAS:
        for beta in BETAS:
            key = f"alpha_{alpha}_beta_{beta}"
            print(f"\n  >> {key}")
            ts = time.time()
            top_list, ov_list = [], []
            for i, (t, sib) in enumerate(pairs):
                v = tp_old_iter(P, rows[t], alpha, beta)
                target = rows[sib]
                rel = np.abs(target - v) / (target + EPS)
                top_idx = np.argsort(target)[::-1][:TOP_K]
                top_list.append(float(rel[top_idx].mean()))
                ov_list.append(float(rel.mean()))
                if (i + 1) % 200 == 0:
                    print(f"    pair {i+1}/{len(pairs)}  ({time.time()-ts:.0f}s)",
                          flush=True)
            results[key] = {
                "alpha": alpha, "beta": beta,
                "top100_mean":   float(np.mean(top_list)),
                "top100_median": float(np.median(top_list)),
                "overall_mean":   float(np.mean(ov_list)),
                "overall_median": float(np.median(ov_list)),
            }
            r = results[key]
            print(f"    OV={r['overall_mean']:.4f}  T100={r['top100_mean']:.4f}",
                  flush=True)

    summary = {
        "model": "old-form (separate damping + E_{2N} teleportation)",
        "n_pairs": len(pairs), "seed": SEED, "bpw": BPW,
        "alphas": ALPHAS, "betas": BETAS,
        "data": "popularity_results_smoothed_osm_gamma020.npz",
        "results": results,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {time.time() - t0:.0f}s   saved {OUT}")


if __name__ == "__main__":
    main()
