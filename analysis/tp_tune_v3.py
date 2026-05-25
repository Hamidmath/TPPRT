"""Two-phase (beta, rho, alpha_s, alpha_l) tuning on full network, v2.

Changes from v1:
  - alpha_s, alpha_l now BOUNDED to [-5, 5] (v1 was unbounded and the
    L-BFGS-B line search sent alpha_s -> 44, which is a degenerate
    "max-speed indicator" rather than a meaningful weighting).
  - beta, rho widened to [0.01, 0.40] (v1's [0.01, 0.20] pinned the
    upper bound and the model wanted more).
  - 420 training pairs (60 per weekday, seed=11), still independent
    of the 840 evaluation pairs (seed=7).
  - 10 starts, maxiter=30.

Saves: results/tp_tune_v3.json
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

BETA_LO, BETA_HI = 0.01, 0.60
RHO_LO,  RHO_HI  = 0.01, 0.60
A_LO,    A_HI    = -5.0, 5.0

N_STARTS = 10


def build_phase_matrix(graph_data, links, lid_to_idx, weights):
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


def get_speed_lane_arrays(graph_data, links):
    speeds, lanes = [], []
    for lid in links:
        ed = graph_data["links"].get(lid, {})
        speeds.append(ed.get("speed", 11.17))
        lanes.append(ed.get("lanes", 1.0))
    speeds = np.array(speeds, dtype=np.float64)
    lanes = np.array(lanes, dtype=np.float64)
    if speeds.mean() > 0:
        speeds /= speeds.mean()
    if lanes.mean() > 0:
        lanes /= lanes.mean()
    return speeds, lanes


def tp_iter(P_up_cs, P_down_cs, E_b, beta, rho, tol, max_iter):
    N = E_b.shape[0]
    v_up = E_b.copy()
    v_down = np.zeros(N)
    for k in range(max_iter):
        s_down = float(v_down.sum())
        v_up_n = (1.0 - beta) * (P_up_cs @ v_up) + rho * E_b * s_down
        v_down_n = beta * v_up + (1.0 - rho) * (P_down_cs @ v_down)
        s = float(v_up_n.sum() + v_down_n.sum())
        if s > 0:
            v_up_n /= s
            v_down_n /= s
        diff = float(np.abs(v_up_n - v_up).sum() + np.abs(v_down_n - v_down).sum())
        v_up, v_down = v_up_n, v_down_n
        if diff < tol:
            break
    return v_up + v_down


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
        if s > 0:
            E /= s
        rows[t] = E
    return rows


def main():
    t0 = time.time()
    print(f"[start] TP tune v3, {N_STARTS} starts, "
          f"beta in [{BETA_LO},{BETA_HI}], rho in [{RHO_LO},{RHO_HI}], "
          f"a_s,a_l in [{A_LO},{A_HI}]")

    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes = get_speed_lane_arrays(graph, links)
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

    def get_kernels(a_s, a_l):
        key = (round(a_s, 4), round(a_l, 4))
        if key in cache:
            return cache[key]
        w_up = np.ones(N, dtype=np.float64)
        w_down = np.ones(N, dtype=np.float64)
        if a_s != 0.0:
            w_up   = w_up   * np.power(speeds, +a_s)
            w_down = w_down * np.power(speeds, -a_s)
        if a_l != 0.0:
            w_up   = w_up   * np.power(lanes, +a_l)
            w_down = w_down * np.power(lanes, -a_l)
        P_up   = build_phase_matrix(graph, links, lid_to_idx, w_up)
        P_down = build_phase_matrix(graph, links, lid_to_idx, w_down)
        cache[key] = (P_up, P_down)
        if len(cache) > 64:
            # Drop oldest entries to keep memory bounded.
            for k in list(cache.keys())[:32]:
                del cache[k]
        return cache[key]

    def loss(x):
        beta, rho, a_s, a_l = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        P_up_cs, P_down_cs = get_kernels(a_s, a_l)
        per = []
        for i in range(len(pairs)):
            v = tp_iter(P_up_cs, P_down_cs, Es_t[i], beta, rho, TOL, MAX_ITER)
            E_tp = Es_tp1[i]
            top = top_idx_list[i]
            rel = np.abs(E_tp[top] - v[top]) / (E_tp[top] + EPS)
            per.append(float(rel.mean()))
        out = float(np.mean(per))
        n_calls[0] += 1
        if n_calls[0] % 5 == 0:
            print(f"    eval {n_calls[0]:4d}  beta={beta:.4f} rho={rho:.4f} "
                  f"a_s={a_s:+.3f} a_l={a_l:+.3f}  loss={out:.4f}", flush=True)
        return out

    rng = np.random.default_rng(SEED)
    starts = [
        [0.1019, 0.1014, 0.0,  0.0],   # published
        [0.20,   0.20,   2.0,  2.0],   # near v1 endpoint
        [0.10,   0.10,  -1.0, -1.0],
        [0.30,   0.30,   3.0,  0.0],
        [0.05,   0.30,   0.0,  3.0],
    ]
    while len(starts) < N_STARTS:
        starts.append([
            float(rng.uniform(BETA_LO, BETA_HI)),
            float(rng.uniform(RHO_LO,  RHO_HI)),
            float(rng.uniform(A_LO,    A_HI)),
            float(rng.uniform(A_LO,    A_HI)),
        ])
    starts = [np.array(s, dtype=np.float64) for s in starts]

    bounds = [(BETA_LO, BETA_HI), (RHO_LO, RHO_HI), (A_LO, A_HI), (A_LO, A_HI)]
    results = []
    for i, x0 in enumerate(starts):
        ts = time.time()
        print(f"\n[start {i:02d}]  x0 = beta={x0[0]:.4f} rho={x0[1]:.4f} "
              f"a_s={x0[2]:+.3f} a_l={x0[3]:+.3f}", flush=True)
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
        out_path = RESULTS / "tp_tune_v3.json"
        partial = {
            "bounds": dict(beta=[BETA_LO, BETA_HI], rho=[RHO_LO, RHO_HI],
                           a_s=[A_LO, A_HI], a_l=[A_LO, A_HI]),
            "n_starts": N_STARTS, "pairs": len(pairs),
            "tune_seed": SEED, "bins_per_weekday_train": BINS_PER_WEEKDAY,
            "results": results,
            "elapsed_total": time.time() - t0,
        }
        with open(out_path, "w") as f:
            json.dump(partial, f, indent=2)

    best = min(results, key=lambda r: r["f_opt"])
    print(f"\n=== BEST: beta={best['x_opt'][0]:.4f} rho={best['x_opt'][1]:.4f} "
          f"a_s={best['x_opt'][2]:+.4f} a_l={best['x_opt'][3]:+.4f} "
          f"loss={best['f_opt']:.4f}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
