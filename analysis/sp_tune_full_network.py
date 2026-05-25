"""Single-phase (alpha, alpha_s, alpha_l) tuning on full network.

Analog of tp_tune_v2.py for the single-phase chain. Same training
sample (420 pairs, seed=11), same bounds on alpha_s/alpha_l, but
optimises alpha (teleport probability) in [0.01, 0.30] instead of
(beta, rho).

Used to populate v4, v5, v6 of the ladder with real tuned values.

Saves: results/sp_tune.json
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config
from core.io import load_popularity_npz

INPUTS = Path(os.environ.get("TPPR_INPUTS", config.DATA_DIR))
RESULTS = Path(os.environ.get("TPPR_RESULTS", config.PROJECT_ROOT / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)

SMOOTH_NPZ = str(INPUTS / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON = str(INPUTS / "city_graph_full.json")

TOL, MAX_ITER = 1e-5, 100
EPS = 1e-6
TOP_K = 100
SEED = 11
BINS_PER_WEEKDAY = 60

ALPHA_LO, ALPHA_HI = 0.01, 0.30
A_LO,     A_HI     = -5.0, 5.0

N_STARTS = 10


def build_P_cs(graph_data, links, lid_to_idx, weights):
    adj = graph_data.get("adjacency", {})
    rows, cols, data = [], [], []
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ:
            continue
        out_w = np.array([weights[j] for j in succ], dtype=np.float64)
        total = out_w.sum()
        if total <= 0:
            continue
        norm_w = out_w / total
        for k, j in enumerate(succ):
            rows.append(j); cols.append(i); data.append(float(norm_w[k]))
    N = len(links)
    return csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64)


def get_speed_lane(graph_data, links):
    s, l = [], []
    for lid in links:
        ed = graph_data["links"].get(lid, {})
        s.append(ed.get("speed", 11.17))
        l.append(ed.get("lanes", 1.0))
    s = np.array(s, dtype=np.float64); l = np.array(l, dtype=np.float64)
    if s.mean() > 0: s /= s.mean()
    if l.mean() > 0: l /= l.mean()
    return s, l


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


def main():
    t0 = time.time()
    print(f"[start] SP tune, {N_STARTS} starts, alpha in [{ALPHA_LO},{ALPHA_HI}], "
          f"a_s,a_l in [{A_LO},{A_HI}]")

    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes = get_speed_lane(graph, links)
    print(f"  N = {N:,}")

    rows = load_prior(SMOOTH_NPZ, N, lid_to_idx)
    pairs = sample_pairs(sorted(rows.keys()), SEED, BINS_PER_WEEKDAY)
    pairs = [(t, s) for t, s in pairs if t in rows and s in rows]
    print(f"  pairs: {len(pairs)}")

    Es_t   = [rows[t] for t, _ in pairs]
    Es_tp1 = [rows[s] for _, s in pairs]
    top_idx_list = [np.argsort(E)[::-1][:TOP_K] for E in Es_tp1]

    cache = {}
    n_calls = [0]

    def get_kernel(a_s, a_l):
        key = (round(a_s, 4), round(a_l, 4))
        if key in cache:
            return cache[key]
        w = np.ones(N, dtype=np.float64)
        if a_s != 0: w = w * np.power(speeds, a_s)
        if a_l != 0: w = w * np.power(lanes, a_l)
        P = build_P_cs(graph, links, lid_to_idx, w)
        cache[key] = P
        if len(cache) > 96:
            for k in list(cache.keys())[:48]:
                del cache[k]
        return P

    def loss(x):
        alpha, a_s, a_l = float(x[0]), float(x[1]), float(x[2])
        P_cs = get_kernel(a_s, a_l)
        per = []
        for i in range(len(pairs)):
            v = sp_iter(P_cs, Es_t[i], alpha, TOL, MAX_ITER)
            E_tp = Es_tp1[i]
            top = top_idx_list[i]
            rel = np.abs(E_tp[top] - v[top]) / (E_tp[top] + EPS)
            per.append(float(rel.mean()))
        out = float(np.mean(per))
        n_calls[0] += 1
        if n_calls[0] % 5 == 0:
            print(f"    eval {n_calls[0]:4d}  alpha={alpha:.4f} "
                  f"a_s={a_s:+.3f} a_l={a_l:+.3f}  loss={out:.4f}", flush=True)
        return out

    rng = np.random.default_rng(SEED)
    starts = [
        [0.05, 0.0, 0.0],     # geometric-fit alpha
        [0.15, 0.0, 0.0],     # Google alpha
        [0.05, 2.0, 2.0],
        [0.10, -1.0, -1.0],
        [0.20, 3.0, 0.0],
    ]
    while len(starts) < N_STARTS:
        starts.append([
            float(rng.uniform(ALPHA_LO, ALPHA_HI)),
            float(rng.uniform(A_LO,     A_HI)),
            float(rng.uniform(A_LO,     A_HI)),
        ])
    starts = [np.array(s, dtype=np.float64) for s in starts]

    bounds = [(ALPHA_LO, ALPHA_HI), (A_LO, A_HI), (A_LO, A_HI)]
    results = []
    for i, x0 in enumerate(starts):
        ts = time.time()
        print(f"\n[start {i:02d}]  x0 = alpha={x0[0]:.4f} a_s={x0[1]:+.3f} a_l={x0[2]:+.3f}",
              flush=True)
        res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds,
                       options=dict(eps=1e-3, ftol=1e-6, gtol=1e-4, maxiter=30))
        elapsed = time.time() - ts
        print(f"  done in {elapsed:.0f}s  nfev={res.nfev}  f={res.fun:.4f}  "
              f"x={[float(z) for z in res.x]}", flush=True)
        results.append({
            "start_index": i,
            "x0": [float(z) for z in x0],
            "x_opt": [float(z) for z in res.x],
            "f_opt": float(res.fun),
            "nfev": int(res.nfev),
            "success": bool(res.success),
            "message": str(res.message),
            "elapsed_s": elapsed,
        })
        with open(RESULTS / "sp_tune.json", "w") as f:
            json.dump({
                "bounds": dict(alpha=[ALPHA_LO, ALPHA_HI],
                               a_s=[A_LO, A_HI], a_l=[A_LO, A_HI]),
                "n_starts": N_STARTS, "pairs": len(pairs),
                "tune_seed": SEED, "bins_per_weekday_train": BINS_PER_WEEKDAY,
                "results": results,
                "elapsed_total": time.time() - t0,
            }, f, indent=2)

    best = min(results, key=lambda r: r["f_opt"])
    print(f"\n=== BEST: alpha={best['x_opt'][0]:.4f} "
          f"a_s={best['x_opt'][1]:+.4f} a_l={best['x_opt'][2]:+.4f} "
          f"loss={best['f_opt']:.4f}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
