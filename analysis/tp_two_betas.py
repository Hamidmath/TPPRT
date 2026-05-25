"""Two-phase PR vanilla, beta = rho in {0.15, 0.05}, alpha_s = alpha_l = 0,
on the SAME 840 (t, t+7d) pairs used by vanilla_pr_two_alphas.py.

TP iteration:
  v_up'   = (1 - beta) P_up v_up + rho * E_b * sum(v_down)
  v_down' = beta v_up + (1 - rho) P_down v_down
Vanilla means alpha_s = alpha_l = 0, so P_up = P_down = uniform out-edges P_cs.

For each pair:
  - E_b at slot     = diffused E_b(t)              (teleport target)
  - target          = diffused E_b(t+7d)
  - prediction      = v_up + v_down at convergence
  - MRE per pair    = Top-100 + Overall

Output: results/tp_two_betas.json
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
OUT = RESULTS / "tp_two_betas.json"

EPS = 1e-6
TOP_K = 100
SEED = 7
BPW = 120
BETAS = [0.15, 0.05]      # beta = rho for each
TOL, MAX_ITER = 1e-6, 200


def build_P_cs(graph, links, lid_to_idx):
    adj = graph.get("adjacency", {})
    rows, cols, data = [], [], []
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ: continue
        w = 1.0 / len(succ)
        for j in succ:
            rows.append(j); cols.append(i); data.append(w)
    return csr_matrix((data, (rows, cols)),
                       shape=(len(links), len(links)), dtype=np.float64)


def tp_iter(P_up_cs, P_down_cs, E_b, beta, rho, tol=TOL, max_iter=MAX_ITER):
    N = E_b.shape[0]
    v_up = E_b.copy()
    v_down = np.zeros(N)
    for _ in range(max_iter):
        s_down = float(v_down.sum())
        v_up_n   = (1.0 - beta) * (P_up_cs   @ v_up) + rho * E_b * s_down
        v_down_n = beta * v_up + (1.0 - rho) * (P_down_cs @ v_down)
        s = float(v_up_n.sum() + v_down_n.sum())
        if s > 0:
            v_up_n /= s; v_down_n /= s
        diff = float(np.abs(v_up_n - v_up).sum() + np.abs(v_down_n - v_down).sum())
        v_up, v_down = v_up_n, v_down_n
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
        if s > 0: E /= s
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
    print(f"[start] TP vanilla, beta=rho in {BETAS}, alpha_s=alpha_l=0, "
          f"840 (t,t+7d) pairs")

    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    P_cs = build_P_cs(graph, links, lid_to_idx)
    print(f"  N = {N:,}  P_cs nnz = {P_cs.nnz:,}")

    rows = load_diffused(SMOOTH_NPZ, N, lid_to_idx)
    pairs = sample_pairs(sorted(rows.keys()))
    print(f"  pairs: {len(pairs)}", flush=True)

    summary = {
        "n_pairs": len(pairs), "seed": SEED, "bpw": BPW,
        "betas":  BETAS,
        "rule":   "beta = rho, alpha_s = alpha_l = 0",
        "input":  "diffused E_b(t)",  "target": "diffused E_b(t+7d)",
        "data":   "popularity_results_smoothed_osm_gamma020.npz",
        "results": {},
    }
    for beta in BETAS:
        rho = beta
        print(f"\n  beta = rho = {beta}", flush=True)
        ts = time.time()
        top_list, ov_list = [], []
        for i, (t, sib) in enumerate(pairs):
            E_b = rows[t]
            target = rows[sib]
            v = tp_iter(P_cs, P_cs, E_b, beta, rho)
            rel = np.abs(target - v) / (target + EPS)
            top_idx = np.argsort(target)[::-1][:TOP_K]
            top_list.append(float(rel[top_idx].mean()))
            ov_list.append(float(rel.mean()))
            if (i + 1) % 100 == 0:
                print(f"    pair {i+1}/{len(pairs)}  "
                      f"({time.time() - ts:.0f}s)", flush=True)
        top_arr = np.array(top_list); ov_arr = np.array(ov_list)
        r = {"beta": beta, "rho": rho,
             "top100_mean":    float(top_arr.mean()),
             "top100_median":  float(np.median(top_arr)),
             "overall_mean":   float(ov_arr.mean()),
             "overall_median": float(np.median(ov_arr)),
             "elapsed_s":      time.time() - ts,
             "top100_per_pair":  top_list,
             "overall_per_pair": ov_list}
        summary["results"][f"beta_{beta:.2f}"] = r
        print(f"    Top-100 MRE  mean = {r['top100_mean']:.4f}   "
              f"median = {r['top100_median']:.4f}", flush=True)
        print(f"    Overall MRE  mean = {r['overall_mean']:.4f}   "
              f"median = {r['overall_median']:.4f}", flush=True)
        OUT.write_text(json.dumps(summary, indent=2))

    print(f"\n[done] {time.time() - t0:.0f}s   saved {OUT}")


if __name__ == "__main__":
    main()
