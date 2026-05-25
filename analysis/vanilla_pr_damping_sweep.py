"""Vanilla single-phase PageRank damping sweep.

For each alpha in {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}, eval the
single-phase chain
        v_{n+1} = alpha * prior + (1 - alpha) * (P_cs @ v_n)
with uniform out-edges P_cs, under TWO priors:
  (a) uniform 1/N (textbook PageRank)
  (b) popularity prior E_b at the input time

This shows that no choice of teleportation rate with uniform-out-edges
PR can match the data-driven popularity-prior result. Establishes
that the *prior* (not the *rate*) is what drives the v2->v3 jump in
the ladder.

840 evaluation pairs, seed=7.

Saves: results/vanilla_pr_damping_sweep.json
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config
from core.io import load_popularity_npz

INPUTS = Path(os.environ.get("TPPR_INPUTS", config.DATA_DIR))
RESULTS = Path(os.environ.get("TPPR_RESULTS", config.PROJECT_ROOT / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)

RAW_NPZ    = str(INPUTS / "popularity_results_osm.npz")
SMOOTH_NPZ = str(INPUTS / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON = str(INPUTS / "city_graph_full.json")

TOL, MAX_ITER = 1e-6, 200
EPS = 1e-6
TOP_K = 100
SEED = 7
BPW = 120

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


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


def sp_iter(P_cs, prior, alpha, tol, max_iter):
    v = prior.copy()
    for _ in range(max_iter):
        v_new = alpha * prior + (1.0 - alpha) * (P_cs @ v)
        s = float(v_new.sum())
        if s > 0:
            v_new /= s
        diff = float(np.abs(v_new - v).sum())
        v = v_new
        if diff < tol:
            return v
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
    print(f"[start] vanilla PR damping sweep: alphas = {ALPHAS}")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    P_cs, links, lid_to_idx = build_P_cs(graph)
    N = len(links)
    print(f"  N = {N:,}")

    print("  loading raw prior ...")
    raw = load_prior(RAW_NPZ, N, lid_to_idx)
    print("  loading smoothed prior ...")
    smooth = load_prior(SMOOTH_NPZ, N, lid_to_idx)

    pairs = sample_pairs(sorted(smooth.keys()), SEED, BPW)
    pairs = [(t, s) for t, s in pairs if t in smooth and s in smooth and s in raw]
    print(f"  pairs: {len(pairs)}")

    Es_in     = [smooth[t] for t, _ in pairs]
    Es_target = [raw[s]    for _, s in pairs]
    top_idx   = [np.argsort(E)[::-1][:TOP_K] for E in Es_target]

    summary = {}
    for alpha in ALPHAS:
        for prior_name in ["uniform", "popularity"]:
            tag = f"a{alpha:.2f}_{prior_name}"
            print(f"  [{tag}]", flush=True)
            ts = time.time()
            top_list, ov_list = [], []
            if prior_name == "uniform":
                v_fixed = sp_iter(P_cs, np.full(N, 1.0/N), alpha, TOL, MAX_ITER)
                for i in range(len(pairs)):
                    rel = np.abs(Es_target[i] - v_fixed) / (Es_target[i] + EPS)
                    top_list.append(float(rel[top_idx[i]].mean()))
                    ov_list.append(float(rel.mean()))
            else:
                for i in range(len(pairs)):
                    v = sp_iter(P_cs, Es_in[i], alpha, TOL, MAX_ITER)
                    rel = np.abs(Es_target[i] - v) / (Es_target[i] + EPS)
                    top_list.append(float(rel[top_idx[i]].mean()))
                    ov_list.append(float(rel.mean()))
            summary[tag] = {
                "alpha": alpha, "prior": prior_name,
                "top100_mre_mean": float(np.mean(top_list)),
                "top100_mre_median": float(np.median(top_list)),
                "overall_mre_mean": float(np.mean(ov_list)),
                "overall_mre_median": float(np.median(ov_list)),
                "n_bins": len(top_list),
                "elapsed_s": time.time() - ts,
            }
            print(f"    top100={summary[tag]['top100_mre_mean']:.4f} "
                  f"overall={summary[tag]['overall_mre_mean']:.4f} "
                  f"({summary[tag]['elapsed_s']:.0f}s)", flush=True)
            with open(RESULTS / "vanilla_pr_damping_sweep.json", "w") as f:
                json.dump({"alphas": ALPHAS, "summary": summary,
                           "elapsed_total": time.time() - t0}, f, indent=2)

    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
