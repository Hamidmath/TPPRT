"""Three two-phase tuning variants on the SAME 840 (t, t+7d) pairs
used by vanilla_pr_two_alphas.py and sp_tune_three.py.

Variants (run sequentially in one SLURM job):
  1. lanes_only : tune (beta, rho, alpha_l) ; alpha_s = 0
  2. speed_only : tune (beta, rho, alpha_s) ; alpha_l = 0
  3. both       : tune (beta, rho, alpha_s, alpha_l)

Bounds (professor's spec):
  beta, rho  in [0.01, 0.15]
  alpha_s, alpha_l : unbounded

Sampling, target, data: same as the SP twin script.
  - 840 calendar (t, t+7d) pairs, 120/weekday, seed=7, t in first ~23 days
  - prior = diffused E_b(t),  target = diffused E_b(t+7d)
  - diffused = popularity_results_smoothed_osm_gamma020.npz

Each variant runs L-BFGS-B with 5 starts; we report the best
operating point plus per-pair Top-100 + Overall MRE.

Output: results/tp_tune_three_ov.json
"""
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz

INPUTS  = Path(os.environ.get("TPPR_INPUTS",  str(config.DATA_DIR)))
RESULTS = Path(os.environ.get("TPPR_RESULTS", str(config.PROJECT_ROOT / "results")))
RESULTS.mkdir(parents=True, exist_ok=True)

SMOOTH_NPZ = str(INPUTS / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON = str(INPUTS / "city_graph_full.json")
OUT = RESULTS / "tp_tune_three_ov.json"

EPS = 1e-6
TOP_K = 100
SEED = 7
BPW = 120
TOL, MAX_ITER = 1e-5, 100

BETA_LO, BETA_HI = 0.01, 0.15
RHO_LO,  RHO_HI  = 0.01, 0.15


def build_P_weighted(graph, links, lid_to_idx, weights):
    adj = graph.get("adjacency", {})
    rows, cols, data = [], [], []
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ: continue
        out_w = np.array([weights[j] for j in succ], dtype=np.float64)
        total = out_w.sum()
        if total <= 0: continue
        norm = out_w / total
        for k, j in enumerate(succ):
            rows.append(j); cols.append(i); data.append(float(norm[k]))
    return csr_matrix((data, (rows, cols)),
                       shape=(len(links), len(links)), dtype=np.float64)


def get_speed_lane(graph, links):
    s, l = [], []
    for lid in links:
        ed = graph["links"].get(lid, {})
        s.append(ed.get("speed", 11.17))
        l.append(ed.get("lanes", 1.0))
    s = np.array(s, dtype=np.float64); l = np.array(l, dtype=np.float64)
    if s.mean() > 0: s /= s.mean()
    if l.mean() > 0: l /= l.mean()
    return s, l


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
    print(f"[start] TP three-variant tuning, 840 (t,t+7d) pairs, "
          f"beta,rho in [{BETA_LO}, {BETA_HI}], alpha_s/alpha_l unbounded")

    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes = get_speed_lane(graph, links)
    print(f"  N = {N:,}", flush=True)

    rows = load_diffused(SMOOTH_NPZ, N, lid_to_idx)
    pairs = sample_pairs(sorted(rows.keys()))
    print(f"  pairs: {len(pairs)}", flush=True)
    inputs  = [rows[t]   for t, _   in pairs]
    targets = [rows[sib] for _, sib in pairs]
    top_idx_list = [np.argsort(t)[::-1][:TOP_K] for t in targets]

    cache_up_down = {}
    n_calls = [0]

    def get_kernels(a_s, a_l):
        key = (round(a_s, 4), round(a_l, 4))
        if key in cache_up_down:
            return cache_up_down[key]
        w_up   = np.ones(N, dtype=np.float64)
        w_down = np.ones(N, dtype=np.float64)
        if a_s != 0:
            w_up   = w_up   * np.power(speeds, +a_s)
            w_down = w_down * np.power(speeds, -a_s)
        if a_l != 0:
            w_up   = w_up   * np.power(lanes, +a_l)
            w_down = w_down * np.power(lanes, -a_l)
        P_up   = build_P_weighted(graph, links, lid_to_idx, w_up)
        P_down = build_P_weighted(graph, links, lid_to_idx, w_down)
        cache_up_down[key] = (P_up, P_down)
        if len(cache_up_down) > 64:
            for k in list(cache_up_down.keys())[:32]:
                del cache_up_down[k]
        return cache_up_down[key]

    def loss_full(beta, rho, a_s, a_l):
        P_up, P_down = get_kernels(a_s, a_l)
        top_list, ov_list = [], []
        for i in range(len(pairs)):
            v = tp_iter(P_up, P_down, inputs[i], beta, rho)
            rel = np.abs(targets[i] - v) / (targets[i] + EPS)
            top_list.append(float(rel[top_idx_list[i]].mean()))
            ov_list.append(float(rel.mean()))
        n_calls[0] += 1
        if n_calls[0] % 5 == 0:
            print(f"    eval {n_calls[0]:4d}  beta={beta:.4f} rho={rho:.4f} "
                  f"a_s={a_s:+.3f} a_l={a_l:+.3f}  "
                  f"top100_mean={np.mean(top_list):.4f}  "
                  f"overall_mean={np.mean(ov_list):.4f}", flush=True)
        return float(np.mean(ov_list)), top_list, ov_list

    def loss(x, mode):
        if mode == "lanes_only":
            beta, rho, a_l = float(x[0]), float(x[1]), float(x[2])
            a_s = 0.0
        elif mode == "speed_only":
            beta, rho, a_s = float(x[0]), float(x[1]), float(x[2])
            a_l = 0.0
        elif mode == "both":
            beta, rho, a_s, a_l = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        out, _, _ = loss_full(beta, rho, a_s, a_l)
        return out

    variants = {
        "lanes_only": {
            "bounds":  [(BETA_LO, BETA_HI), (RHO_LO, RHO_HI), (None, None)],
            "starts": [[0.10, 0.10,  0.0], [0.05, 0.05,  1.0],
                        [0.15, 0.15, -1.0], [0.05, 0.15,  2.0],
                        [0.15, 0.05, -2.0]],
            "param_names": ["beta", "rho", "alpha_l"],
        },
        "speed_only": {
            "bounds":  [(BETA_LO, BETA_HI), (RHO_LO, RHO_HI), (None, None)],
            "starts": [[0.10, 0.10,  0.0], [0.05, 0.05,  1.0],
                        [0.15, 0.15, -1.0], [0.05, 0.15,  2.0],
                        [0.15, 0.05, -2.0]],
            "param_names": ["beta", "rho", "alpha_s"],
        },
        "both": {
            "bounds":  [(BETA_LO, BETA_HI), (RHO_LO, RHO_HI), (None, None), (None, None)],
            "starts": [[0.10, 0.10, 0.0, 0.0],
                        [0.05, 0.05, 1.0, 1.0],
                        [0.15, 0.15, -1.0, -1.0],
                        [0.05, 0.15, 2.0, 0.0],
                        [0.15, 0.05, 0.0, 2.0]],
            "param_names": ["beta", "rho", "alpha_s", "alpha_l"],
        },
    }

    summary = {
        "spec": {
            "n_pairs": len(pairs), "seed": SEED, "bpw": BPW,
            "input": "diffused E_b(t)", "target": "diffused E_b(t+7d)",
            "data": "popularity_results_smoothed_osm_gamma020.npz",
            "beta_bounds": [BETA_LO, BETA_HI],
            "rho_bounds": [RHO_LO, RHO_HI],
            "a_s_bounds": [None, None], "a_l_bounds": [None, None],
        },
        "variants": {},
    }

    for mode, cfg in variants.items():
        print(f"\n========== variant: {mode} ==========", flush=True)
        t_mode = time.time()
        res_list = []
        for i, x0 in enumerate(cfg["starts"]):
            x0_arr = np.array(x0, dtype=np.float64)
            ts = time.time()
            print(f"\n  [start {i}] x0 = {x0}", flush=True)
            res = minimize(lambda x: loss(x, mode), x0_arr,
                            method="L-BFGS-B", bounds=cfg["bounds"],
                            options=dict(eps=2e-3, ftol=1e-6, gtol=1e-4, maxiter=20))
            print(f"    done in {time.time()-ts:.0f}s nfev={res.nfev} "
                  f"f={res.fun:.4f} x={list(map(float, res.x))}", flush=True)
            res_list.append({
                "start_index": i,
                "x0": list(map(float, x0)),
                "x_opt": list(map(float, res.x)),
                "f_opt": float(res.fun),
                "nfev": int(res.nfev),
                "success": bool(res.success),
                "message": str(res.message),
            })
        best = min(res_list, key=lambda r: r["f_opt"])
        if mode == "lanes_only":
            beta, rho, a_l = best["x_opt"]; a_s = 0.0
        elif mode == "speed_only":
            beta, rho, a_s = best["x_opt"]; a_l = 0.0
        else:
            beta, rho, a_s, a_l = best["x_opt"]
        _, top_list, ov_list = loss_full(beta, rho, a_s, a_l)
        top_arr = np.array(top_list); ov_arr = np.array(ov_list)
        summary["variants"][mode] = {
            "param_names": cfg["param_names"],
            "starts": res_list,
            "best": best,
            "beta":    float(beta),
            "rho":     float(rho),
            "alpha_s": float(a_s),
            "alpha_l": float(a_l),
            "top100_mean":    float(top_arr.mean()),
            "top100_median":  float(np.median(top_arr)),
            "overall_mean":   float(ov_arr.mean()),
            "overall_median": float(np.median(ov_arr)),
            "top100_per_pair":  top_list,
            "overall_per_pair": ov_list,
            "elapsed_s": time.time() - t_mode,
        }
        OUT.write_text(json.dumps(summary, indent=2))
        print(f"  [{mode}] best: beta={beta:.4f} rho={rho:.4f} "
              f"a_s={a_s:+.3f} a_l={a_l:+.3f}  "
              f"top100_mean={top_arr.mean():.4f}  "
              f"overall_mean={ov_arr.mean():.4f}", flush=True)

    print("\n========== summary ==========", flush=True)
    for mode, r in summary["variants"].items():
        print(f"  {mode:<12s}  beta={r['beta']:.4f}  rho={r['rho']:.4f}  "
              f"a_s={r['alpha_s']:+.3f}  a_l={r['alpha_l']:+.3f}  "
              f"top100={r['top100_mean']:.4f}  overall={r['overall_mean']:.4f}")
    print(f"\n[done] {time.time() - t0:.0f}s\nsaved {OUT}")


if __name__ == "__main__":
    main()
