"""Vanilla single-phase PR, two alphas (0.15 and 0.05),
on 840 calendar slots (120 per weekday, seed=7), first ~23 days.

For each slot t:
  - prior = diffused E_b(t)
  - run PR with uniform out-edges:  v_{n+1} = alpha * prior + (1-alpha) * P_cs v_n
  - target = diffused E_b(t + 7 days)
  - compute MRE per slot (Top-100 and Overall)

Output: mean MRE across the 840 slots, for each alpha.
"""
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
import os
from core.io import load_popularity_npz

SMOOTH_NPZ = str(Path(os.environ.get("TPPR_INPUTS", str(config.DATA_DIR))) / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON = str(Path(os.environ.get("TPPR_INPUTS", str(config.DATA_DIR))) / "city_graph_full.json")
OUT = Path(os.environ.get("TPPR_RESULTS", str(config.PROJECT_ROOT / "results"))) / "vanilla_pr_two_alphas.json"

EPS = 1e-6
TOP_K = 100
SEED = 7
BPW = 120
ALPHAS = [0.15, 0.05]
TOL, MAX_ITER = 1e-6, 200


def build_P_cs(graph):
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    adj = graph.get("adjacency", {})
    rows, cols, data = [], [], []
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ: continue
        w = 1.0 / len(succ)
        for j in succ:
            rows.append(j); cols.append(i); data.append(w)
    return csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64), links, lid_to_idx


def sp_iter(P_cs, prior, alpha, tol=TOL, max_iter=MAX_ITER):
    v = prior.copy()
    for _ in range(max_iter):
        v_new = alpha * prior + (1.0 - alpha) * (P_cs @ v)
        s = float(v_new.sum())
        if s > 0:
            v_new /= s
        if float(np.abs(v_new - v).sum()) < tol:
            return v_new
        v = v_new
    return v


def load_prior(npz_path, N, lid_to_idx):
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
        if s > 0: E /= s
        rows[t] = E
    return rows


def sample_pairs(times, seed, bpw):
    """120 random t per weekday such that t+7d also exists in the corpus."""
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
    print("[start] vanilla PR, two alphas, 840 slots (120/weekday)")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    P_cs, links, lid_to_idx = build_P_cs(graph)
    N = len(links)
    print(f"  N={N:,}")

    rows = load_prior(SMOOTH_NPZ, N, lid_to_idx)
    print(f"  bins={len(rows):,}")

    pairs = sample_pairs(sorted(rows.keys()), SEED, BPW)
    print(f"  pairs: {len(pairs)}")

    summary = {"n_pairs": len(pairs), "seed": SEED, "bpw": BPW,
               "data": "popularity_results_smoothed_osm_gamma020.npz",
               "alphas": ALPHAS, "results": {}}

    for alpha in ALPHAS:
        print(f"\n  alpha = {alpha}")
        top_list, ov_list = [], []
        for t, sib in pairs:
            prior = rows[t]
            target = rows[sib]
            v = sp_iter(P_cs, prior, alpha)
            rel = np.abs(target - v) / (target + EPS)
            top_idx = np.argsort(target)[::-1][:TOP_K]
            top_list.append(float(rel[top_idx].mean()))
            ov_list.append(float(rel.mean()))
        top_arr = np.array(top_list); ov_arr = np.array(ov_list)
        r = {"top100_mean": float(top_arr.mean()),
             "top100_median": float(np.median(top_arr)),
             "overall_mean": float(ov_arr.mean()),
             "overall_median": float(np.median(ov_arr)),
             "top100_per_slot": top_list,
             "overall_per_slot": ov_list}
        summary["results"][f"alpha_{alpha:.2f}"] = r
        print(f"    Top-100 MRE  mean = {r['top100_mean']:.4f}   "
              f"median = {r['top100_median']:.4f}")
        print(f"    Overall MRE  mean = {r['overall_mean']:.4f}   "
              f"median = {r['overall_median']:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
