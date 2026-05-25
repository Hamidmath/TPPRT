"""Additive baseline ladder, full network, week-to-week prediction.

Reproduces the professor's headline experiment: start from the
simplest possible baseline and add one modeling choice at a time;
report Top-100 and Overall MRE on a fixed 840-bin sample so each row
adds exactly one feature.

Variants (each row in the output table):
  v0  climatology              -- mean of other-weeks E_b at same (weekday, time-of-day); no PR
  v1  SP uniform, alpha=0.15   -- vanilla PageRank, uniform 1/N teleport, Google damping
  v2  SP uniform, alpha=0.05   -- same, teleport rate from geometric fit to route lengths
  v3  SP uniform, popularity   -- alpha=0.05, uniform out-edges, data-driven teleport prior E_b
  v4  SP + speed tuning        -- alpha=0.05, speed-weighted out-edges, popularity prior
  v5  SP + lanes tuning        -- alpha=0.05, lane-weighted out-edges,  popularity prior
  v6  SP + speed + lanes       -- alpha=0.05, both speed and lane weights, popularity prior
  v7  TP + speed + lanes       -- two-phase chain, all of the above (our model)

Tuning of (alpha_s, alpha_l, beta, rho) reuses the operating point
from the full-network tuning (see tp_tune_full_network.py). If that
job has not finished, the script falls back to the published values
beta=0.1019, rho=0.1014, alpha_s=alpha_l=0 (i.e. v7 reduces to v3-ish
two-phase) and notes that in the output JSON.

Output: results/ladder_eval_full_network.json
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

INPUTS = Path(os.environ.get("TPPR_INPUTS",
                             config.DATA_DIR))
RESULTS = Path(os.environ.get("TPPR_RESULTS",
                              config.PROJECT_ROOT / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)

RAW_NPZ    = str(INPUTS / "popularity_results_osm.npz")
SMOOTH_NPZ = str(INPUTS / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON = str(INPUTS / "city_graph_full.json")

TOL, MAX_ITER = 1e-6, 200
EPS = 1e-6
TOP_K = 100
SEED = 7
BINS_PER_WEEKDAY = 120

ALPHA_GOOGLE = 0.15
ALPHA_GEOM   = 0.0508
P_GEOM       = 0.0508

# Fallback published params if the tuner hasn't finished yet.
FALLBACK_BETA, FALLBACK_RHO = 0.1019, 0.1014


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


def sp_iter(P_cs, prior, alpha, tol, max_iter):
    """v_{n+1} = alpha * prior + (1 - alpha) * (P_cs @ v_n)"""
    v = prior.copy()
    for k in range(max_iter):
        v_new = alpha * prior + (1.0 - alpha) * (P_cs @ v)
        s = float(v_new.sum())
        if s > 0:
            v_new /= s
        diff = float(np.abs(v_new - v).sum())
        v = v_new
        if diff < tol:
            return v
    return v


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


def mre_top_and_overall(truth, pred, top_idx, top_k=TOP_K, eps=EPS):
    rel = np.abs(truth - pred) / (truth + eps)
    return float(rel[top_idx].mean()), float(rel.mean())


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


def sample_pairs(times, seed=SEED, bpw=BINS_PER_WEEKDAY):
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


def climatology_pred(rows, sib_time, all_times_by_slot):
    """Predict E(sib_time) as the average of E at the SAME (weekday, time-of-day)
    across OTHER weeks in the corpus."""
    dt = datetime.strptime(sib_time, "%Y-%m-%d %H:%M:%S")
    key = (dt.weekday(), dt.hour, dt.minute)
    candidates = [t for t in all_times_by_slot.get(key, []) if t != sib_time]
    if not candidates:
        # fall back to any time, give uniform
        return None
    Es = [rows[t] for t in candidates if t in rows]
    if not Es:
        return None
    return np.mean(Es, axis=0)


def load_tuned_params():
    """If the tuner has saved a result, pick up its best."""
    p = RESULTS / "tp_tune_full_network.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        rs = data.get("results", [])
        if not rs:
            return None
        best = min(rs, key=lambda r: r["f_opt"])
        return {
            "beta": float(best["x_opt"][0]),
            "rho":  float(best["x_opt"][1]),
            "a_s":  float(best["x_opt"][2]),
            "a_l":  float(best["x_opt"][3]),
            "f_opt": float(best["f_opt"]),
        }
    except Exception:
        return None


def main():
    t0 = time.time()
    print("[start] ladder eval, full network, 840 bins")

    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes = get_speed_lane_arrays(graph, links)
    print(f"  N = {N:,}")

    print("  loading raw prior ...")
    raw_rows = load_prior(RAW_NPZ, N, lid_to_idx)
    print(f"  raw bins: {len(raw_rows):,}")
    print("  loading smoothed prior ...")
    smooth_rows = load_prior(SMOOTH_NPZ, N, lid_to_idx)
    print(f"  smooth bins: {len(smooth_rows):,}")

    pairs = sample_pairs(sorted(smooth_rows.keys()))
    pairs = [(t, s) for t, s in pairs if t in smooth_rows and s in smooth_rows]
    print(f"  pairs: {len(pairs)}")

    # Targets for the MRE: use the *raw* next-week E_b (the contract).
    Es_pred_in = [smooth_rows[t] for t, _ in pairs]    # diffused t  (model input)
    Es_target  = [raw_rows.get(s, smooth_rows[s])
                  for _, s in pairs]                   # raw next-week
    top_idx    = [np.argsort(E)[::-1][:TOP_K] for E in Es_target]

    # Index by (weekday, hour, minute) for climatology.
    all_times_by_slot = {}
    for t in smooth_rows.keys():
        dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        all_times_by_slot.setdefault(
            (dt.weekday(), dt.hour, dt.minute), []
        ).append(t)

    # Tuned params (if available).
    tuned = load_tuned_params()
    if tuned is not None:
        print(f"  using tuned params: beta={tuned['beta']:.4f} rho={tuned['rho']:.4f} "
              f"a_s={tuned['a_s']:+.4f} a_l={tuned['a_l']:+.4f}")
    else:
        print(f"  no tuned params yet; falling back to published "
              f"beta={FALLBACK_BETA} rho={FALLBACK_RHO} a_s=a_l=0")
        tuned = {"beta": FALLBACK_BETA, "rho": FALLBACK_RHO,
                 "a_s": 0.0, "a_l": 0.0, "f_opt": None}

    # Pre-build the uniform-weights kernels once -- needed by v1..v3 and v7.
    print("  building uniform-weight P_cs ...")
    w_uniform = np.ones(N, dtype=np.float64)
    P_cs_uniform = build_phase_matrix(graph, links, lid_to_idx, w_uniform)

    # Per-variant kernels built lazily on demand.
    kernel_cache = {("uniform",): P_cs_uniform}

    def get_sp_kernel(a_s, a_l):
        key = ("sp", round(a_s, 6), round(a_l, 6))
        if key in kernel_cache:
            return kernel_cache[key]
        w = np.ones(N, dtype=np.float64)
        if a_s != 0:
            w = w * np.power(speeds, a_s)
        if a_l != 0:
            w = w * np.power(lanes, a_l)
        K = build_phase_matrix(graph, links, lid_to_idx, w)
        kernel_cache[key] = K
        return K

    def get_tp_kernels(a_s, a_l):
        key = ("tp", round(a_s, 6), round(a_l, 6))
        if key in kernel_cache:
            return kernel_cache[key]
        w_up   = np.ones(N, dtype=np.float64)
        w_down = np.ones(N, dtype=np.float64)
        if a_s != 0:
            w_up   = w_up   * np.power(speeds, +a_s)
            w_down = w_down * np.power(speeds, -a_s)
        if a_l != 0:
            w_up   = w_up   * np.power(lanes, +a_l)
            w_down = w_down * np.power(lanes, -a_l)
        P_up   = build_phase_matrix(graph, links, lid_to_idx, w_up)
        P_down = build_phase_matrix(graph, links, lid_to_idx, w_down)
        kernel_cache[key] = (P_up, P_down)
        return (P_up, P_down)

    # ------------------------------------------------------------------
    # Variants
    # ------------------------------------------------------------------
    variants = [
        # name, builder (returns prediction for pair index i)
        ("v0_climatology",
         lambda i, ttest=pairs: climatology_pred(smooth_rows, ttest[i][1], all_times_by_slot)),
        ("v1_sp_uniform_alpha015_uniformprior",
         lambda i: sp_iter(P_cs_uniform, np.full(N, 1.0/N), ALPHA_GOOGLE, TOL, MAX_ITER)),
        ("v2_sp_uniform_alpha005_uniformprior",
         lambda i: sp_iter(P_cs_uniform, np.full(N, 1.0/N), ALPHA_GEOM,   TOL, MAX_ITER)),
        ("v3_sp_uniform_alpha005_popprior",
         lambda i: sp_iter(P_cs_uniform, Es_pred_in[i],     ALPHA_GEOM,   TOL, MAX_ITER)),
        ("v4_sp_speedonly_popprior",
         lambda i: sp_iter(get_sp_kernel(tuned["a_s"], 0.0),
                            Es_pred_in[i], ALPHA_GEOM, TOL, MAX_ITER)),
        ("v5_sp_lanesonly_popprior",
         lambda i: sp_iter(get_sp_kernel(0.0, tuned["a_l"]),
                            Es_pred_in[i], ALPHA_GEOM, TOL, MAX_ITER)),
        ("v6_sp_speed_lanes_popprior",
         lambda i: sp_iter(get_sp_kernel(tuned["a_s"], tuned["a_l"]),
                            Es_pred_in[i], ALPHA_GEOM, TOL, MAX_ITER)),
        ("v7_tp_speed_lanes_popprior",
         lambda i: tp_iter(*get_tp_kernels(tuned["a_s"], tuned["a_l"]),
                            Es_pred_in[i], tuned["beta"], tuned["rho"],
                            TOL, MAX_ITER)),
    ]

    # Pre-bake one fixed PageRank vector for v1/v2 (independent of the bin).
    v_v1 = sp_iter(P_cs_uniform, np.full(N, 1.0/N), ALPHA_GOOGLE, TOL, MAX_ITER * 2)
    v_v2 = sp_iter(P_cs_uniform, np.full(N, 1.0/N), ALPHA_GEOM,   TOL, MAX_ITER * 2)
    fixed_pred = {"v1_sp_uniform_alpha015_uniformprior": v_v1,
                  "v2_sp_uniform_alpha005_uniformprior": v_v2}

    summary = {}
    for vname, fn in variants:
        print(f"\n[run] {vname}")
        ts = time.time()
        top_list, ov_list = [], []
        for i in range(len(pairs)):
            E_target = Es_target[i]
            tidx = top_idx[i]
            if vname in fixed_pred:
                v = fixed_pred[vname]
            else:
                v = fn(i)
                if v is None:
                    continue
            top_mre, ov_mre = mre_top_and_overall(E_target, v, tidx)
            top_list.append(top_mre)
            ov_list.append(ov_mre)
            if (i + 1) % 100 == 0:
                print(f"  ... {i+1}/{len(pairs)}  ({time.time()-ts:.0f}s)",
                      flush=True)
        if top_list:
            summary[vname] = {
                "n_bins": len(top_list),
                "top100_mre_mean": float(np.mean(top_list)),
                "top100_mre_median": float(np.median(top_list)),
                "overall_mre_mean": float(np.mean(ov_list)),
                "overall_mre_median": float(np.median(ov_list)),
                "elapsed_s": time.time() - ts,
            }
        else:
            summary[vname] = {"n_bins": 0, "error": "no predictions"}
        print(f"  done: {vname}  top100={summary[vname].get('top100_mre_mean')}  "
              f"overall={summary[vname].get('overall_mre_mean')}  "
              f"({time.time()-ts:.0f}s)", flush=True)

        # Save partial after every variant.
        out = RESULTS / "ladder_eval_full_network.json"
        with open(out, "w") as f:
            json.dump({"tuned": tuned, "summary": summary,
                       "n_pairs_planned": len(pairs),
                       "elapsed_total": time.time() - t0}, f, indent=2)

    print("\n========================================")
    print(f"  {'variant':<42s}  {'top100':>8s}  {'overall':>8s}")
    for k, v in summary.items():
        if "top100_mre_mean" in v:
            print(f"  {k:<42s}  {v['top100_mre_mean']:>8.4f}  "
                  f"{v['overall_mre_mean']:>8.4f}")
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
