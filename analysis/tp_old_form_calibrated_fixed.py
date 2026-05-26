"""Old-form TP tuning with both chain parameters FIXED at the
geometric-fit values from the trip-length distribution:

  alpha = 0.0508  (1 / (1 + E[K_total]))   -- total trip damping
  beta  = 0.102   (1 / (1 + E[L_up]))      -- up-phase commit rate

Three variants, only road-type weights tuned:
  lanes_only   tune alpha_l only
  speed_only   tune alpha_s only
  both         tune (alpha_s, alpha_l)

5 L-BFGS-B starts each, eps=2e-3, maxiter=20. alpha_s, alpha_l unbounded.

Output: results/tp_old_form_calibrated_fixed.json
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
OUT = RESULTS / "tp_old_form_calibrated_fixed.json"

EPS = 1e-6
TOP_K = 100
SEED = 7
BPW = 120
TOL, MAX_ITER = 1e-6, 200
ALPHA_FIXED = 0.0508
BETA_FIXED  = 0.102


def build_P_row_weighted(graph, links, lid_to_idx, weights):
    """Row-stochastic transition with weighted edges."""
    adj = graph.get("adjacency", {})
    rows, cols, data = [], [], []
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ:
            continue
        out_w = np.array([weights[j] for j in succ], dtype=np.float64)
        total = out_w.sum()
        if total <= 0:
            continue
        norm = out_w / total
        for k, j in enumerate(succ):
            rows.append(i); cols.append(j); data.append(float(norm[k]))
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


def tp_old_iter(P, E_b, alpha, beta):
    N = E_b.shape[0]
    Pt = P.T
    v_up = E_b.copy()
    v_down = np.zeros(N)
    d = 1.0 - alpha
    for _ in range(MAX_ITER):
        v_up_n   = d * (1.0 - beta) * (Pt @ v_up)   + alpha * E_b
        v_down_n = d * beta * v_up + d * (Pt @ v_down)
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
    print(f"[start] OLD-form TP tuning, alpha={ALPHA_FIXED} and beta={BETA_FIXED} "
          f"FIXED (geometric-fit), OV objective")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes = get_speed_lane(graph, links)
    print(f"  N = {N:,}")
    rows = load_diffused(SMOOTH_NPZ, N, lid_to_idx)
    pairs = sample_pairs(sorted(rows.keys()))
    print(f"  pairs: {len(pairs)}")
    inputs  = [rows[t]   for t, _   in pairs]
    targets = [rows[sib] for _, sib in pairs]
    top_idx_list = [np.argsort(t)[::-1][:TOP_K] for t in targets]

    kernel_cache = {}
    n_calls = [0]

    def get_kernel(a_s, a_l):
        key = (round(a_s, 4), round(a_l, 4))
        if key in kernel_cache:
            return kernel_cache[key]
        w = np.ones(N, dtype=np.float64)
        if a_s != 0: w = w * np.power(speeds, a_s)
        if a_l != 0: w = w * np.power(lanes, a_l)
        P = build_P_row_weighted(graph, links, lid_to_idx, w)
        kernel_cache[key] = P
        if len(kernel_cache) > 96:
            for k in list(kernel_cache.keys())[:48]:
                del kernel_cache[k]
        return P

    def loss_full(beta, a_s, a_l):
        P = get_kernel(a_s, a_l)
        top_list, ov_list = [], []
        for i in range(len(pairs)):
            v = tp_old_iter(P, inputs[i], ALPHA_FIXED, beta)
            rel = np.abs(targets[i] - v) / (targets[i] + EPS)
            top_list.append(float(rel[top_idx_list[i]].mean()))
            ov_list.append(float(rel.mean()))
        n_calls[0] += 1
        if n_calls[0] % 5 == 0:
            print(f"    eval {n_calls[0]:4d}  beta={beta:.3f} "
                  f"a_s={a_s:+.3f} a_l={a_l:+.3f}  "
                  f"top100={np.mean(top_list):.4f}  Overall={np.mean(ov_list):.4f}",
                  flush=True)
        return float(np.mean(ov_list)), top_list, ov_list

    def mk_loss_lanes_only():
        def loss(x):
            a_l = float(x[0])
            return loss_full(BETA_FIXED, 0.0, a_l)[0]
        return loss
    def mk_loss_speed_only():
        def loss(x):
            a_s = float(x[0])
            return loss_full(BETA_FIXED, a_s, 0.0)[0]
        return loss
    def mk_loss_both():
        def loss(x):
            a_s, a_l = float(x[0]), float(x[1])
            return loss_full(BETA_FIXED, a_s, a_l)[0]
        return loss

    VARIANTS = {
        "lanes_only": dict(
            loss=mk_loss_lanes_only(),
            bounds=[(None, None)],
            starts=[[0.0], [0.5], [1.0], [-0.5], [2.0]],
        ),
        "speed_only": dict(
            loss=mk_loss_speed_only(),
            bounds=[(None, None)],
            starts=[[0.0], [1.0], [2.0], [-1.0], [3.0]],
        ),
        "both": dict(
            loss=mk_loss_both(),
            bounds=[(None, None), (None, None)],
            starts=[[0.0, 0.0], [1.0, 0.5], [2.65, 0.17],
                    [-1.0, -1.0], [3.0, 1.0]],
        ),
    }

    out_variants = {}
    for mode, spec in VARIANTS.items():
        print(f"\n========== variant: {mode} ==========")
        starts = spec["starts"]
        bounds = spec["bounds"]
        loss = spec["loss"]
        runs = []
        best = None
        for j, x0 in enumerate(starts):
            ts = time.time()
            print(f"\n  [start {j}] x0 = {x0}", flush=True)
            res = minimize(loss, np.array(x0, dtype=np.float64),
                            method="L-BFGS-B", bounds=bounds,
                            options=dict(eps=2e-3, ftol=1e-6, gtol=1e-4, maxiter=20))
            print(f"    done in {time.time()-ts:.0f}s nfev={res.nfev} "
                  f"f={res.fun:.4f} x={list(map(float, res.x))}", flush=True)
            runs.append({"start": j, "x0": list(map(float, x0)),
                          "x_opt": list(map(float, res.x)),
                          "f_opt": float(res.fun), "nfev": int(res.nfev)})
            if best is None or res.fun < best["f_opt"]:
                best = runs[-1]
        x_opt = best["x_opt"]
        beta = BETA_FIXED
        if mode == "lanes_only":
            a_s, a_l = 0.0, x_opt[0]
        elif mode == "speed_only":
            a_s, a_l = x_opt[0], 0.0
        else:
            a_s, a_l = x_opt[0], x_opt[1]
        _, top_list, ov_list = loss_full(beta, a_s, a_l)
        out_variants[mode] = {
            "alpha": ALPHA_FIXED,
            "beta": float(beta),
            "alpha_s": float(a_s), "alpha_l": float(a_l),
            "starts": runs, "best": best,
            "top100_mean":   float(np.mean(top_list)),
            "top100_median": float(np.median(top_list)),
            "overall_mean":   float(np.mean(ov_list)),
            "overall_median": float(np.median(ov_list)),
        }
        r = out_variants[mode]
        print(f"  [{mode}] best: beta={beta:.4f} a_s={a_s:+.3f} a_l={a_l:+.3f}  "
              f"top100={r['top100_mean']:.4f}  Overall={r['overall_mean']:.4f}",
              flush=True)

    summary = {
        "model": "old-form (separate damping + E_{2N} teleportation)",
        "alpha_fixed": ALPHA_FIXED,
        "beta_fixed": BETA_FIXED,
        "n_pairs": len(pairs), "seed": SEED, "bpw": BPW,
        "data": "popularity_results_smoothed_osm_gamma020.npz",
        "variants": out_variants,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {time.time()-t0:.0f}s   saved {OUT}")


if __name__ == "__main__":
    main()
