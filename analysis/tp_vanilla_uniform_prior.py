"""Two-phase chains with UNIFORM teleportation prior u = 1/N.

Both chains run on the same 99,716-link column-stochastic P_cs (uniform
out-edges, no road-type weights). Each chain converges to a single v*
that does not depend on the time bin t; we then compare v* against each
of the 840 (t, t+7d) target slots and average Overall MRE.

We report each chain at TWO settings:
  - the geometric-fit values from the trip-length distribution
  - the upper-bound values that the tuned ladder pinned to

For both runs, the prior used inside the chain is u (no E_b), so this is
a strict "no traffic data, only network topology" baseline.

Output: results/tp_vanilla_uniform.json
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
OUT = RESULTS / "tp_vanilla_uniform.json"

EPS = 1e-6
SEED = 7
BPW = 120
TOL, MAX_ITER = 1e-9, 500


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


def tp_new_iter(P_cs, u, beta, rho, tol=TOL, max_iter=MAX_ITER):
    """New TP (main body): mass-conserving, restart via rho * u * 1^T v_down."""
    N = u.shape[0]
    v_up = u.copy()
    v_down = np.zeros(N)
    for _ in range(max_iter):
        s_down = float(v_down.sum())
        v_up_new   = (1.0 - beta) * (P_cs @ v_up) + rho * u * s_down
        v_down_new = beta * v_up + (1.0 - rho) * (P_cs @ v_down)
        s = float(v_up_new.sum() + v_down_new.sum())
        if s > 0:
            v_up_new /= s; v_down_new /= s
        diff = float(np.abs(v_up_new - v_up).sum() + np.abs(v_down_new - v_down).sum())
        v_up, v_down = v_up_new, v_down_new
        if diff < tol:
            break
    return v_up + v_down


def tp_old_iter(P_cs, u, alpha, beta, tol=TOL, max_iter=MAX_ITER):
    """Old TP (classical PageRank style): global damping alpha + E_{2N}=[u,0]."""
    N = u.shape[0]
    v_up = u.copy()
    v_down = np.zeros(N)
    d = 1.0 - alpha
    for _ in range(max_iter):
        v_up_new   = d * (1.0 - beta) * (P_cs @ v_up) + alpha * u
        v_down_new = d * beta * v_up + d * (P_cs @ v_down)
        s = float(v_up_new.sum() + v_down_new.sum())
        if s < 1.0:
            v_up_new += (1.0 - s) * u
        diff = float(np.abs(v_up_new - v_up).sum() + np.abs(v_down_new - v_down).sum())
        v_up, v_down = v_up_new, v_down_new
        if diff < tol:
            break
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


def eval_v_star(v_star, pairs, diff_rows):
    per_pair = []
    for t, sib in pairs:
        tgt = diff_rows[sib]
        if tgt.sum() == 0:
            continue
        per_pair.append(float(np.mean(np.abs(tgt - v_star) / (tgt + EPS))))
    return {
        "overall_mean":   float(np.mean(per_pair)),
        "overall_median": float(np.median(per_pair)),
        "n_pairs": len(per_pair),
    }


def main():
    t0 = time.time()
    print("[start] TP chains with uniform teleportation prior u = 1/N")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    P_cs, links, lid_to_idx = build_P_cs(graph)
    N = len(links)
    print(f"  N = {N:,}, P_cs nnz = {P_cs.nnz:,}")

    u = np.full(N, 1.0 / N, dtype=np.float64)

    diff_rows = load_diffused(SMOOTH_NPZ, N, lid_to_idx)
    pairs = sample_pairs(sorted(diff_rows.keys()))
    print(f"  pairs: {len(pairs)}")

    out = {}

    # --- new-form TP (body): two settings ---
    new_settings = [
        ("calibrated", 0.102, 0.101),
        ("upper_bound", 0.15, 0.15),
    ]
    for label, beta, rho in new_settings:
        ts = time.time()
        print(f"\n  new-form TP, {label}: beta={beta}, rho={rho}")
        v_star = tp_new_iter(P_cs, u, beta, rho)
        r = eval_v_star(v_star, pairs, diff_rows)
        print(f"    Overall MRE mean   = {r['overall_mean']:.4f}  ({time.time()-ts:.0f}s)")
        print(f"    Overall MRE median = {r['overall_median']:.4f}")
        out[f"new_{label}"] = dict(beta=beta, rho=rho, **r)

    # --- old-form TP (appendix): two settings ---
    old_settings = [
        ("calibrated", 0.0508, 0.102),
        ("upper_bound", 0.15, 0.15),
    ]
    for label, alpha, beta in old_settings:
        ts = time.time()
        print(f"\n  old-form TP, {label}: alpha={alpha}, beta={beta}")
        v_star = tp_old_iter(P_cs, u, alpha, beta)
        r = eval_v_star(v_star, pairs, diff_rows)
        print(f"    Overall MRE mean   = {r['overall_mean']:.4f}  ({time.time()-ts:.0f}s)")
        print(f"    Overall MRE median = {r['overall_median']:.4f}")
        out[f"old_{label}"] = dict(alpha=alpha, beta=beta, **r)

    summary = {
        "model": "TP chains (new + old forms) with uniform teleportation prior u=1/N",
        "n_pairs": len(pairs), "seed": SEED, "bpw": BPW,
        "graph": "city_graph_full.json",
        "target": "popularity_results_smoothed_osm_gamma020.npz",
        "results": out,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {time.time() - t0:.0f}s saved {OUT}")


if __name__ == "__main__":
    main()
