"""Ladder evaluation, pass 2: uses tuned parameters from tp_tune_v2.py
and sp_tune_full_network.py, plus a per-weekday breakdown, plus
three sample seeds for stability.

For each seed in {7, 17, 42}, evaluate the 8 ladder variants on 840
bins (120 per weekday). Report Top-100 / Overall MRE per variant
per seed, plus per-weekday breakdown for seed=7.

Saves: results/ladder_eval_v2.json
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
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
SEEDS = [7, 17, 42]
BPW = 120

ALPHA_GOOGLE = 0.15
ALPHA_GEOM   = 0.0508


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


def tp_iter(P_up_cs, P_down_cs, E_b, beta, rho, tol, max_iter):
    N = E_b.shape[0]
    v_up = E_b.copy()
    v_down = np.zeros(N)
    for _ in range(max_iter):
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


def load_tuned(path, default):
    if not Path(path).exists():
        print(f"  WARNING: {path} not found; using default {default}")
        return default
    try:
        with open(path) as f:
            data = json.load(f)
        rs = data.get("results", [])
        if not rs:
            return default
        best = min(rs, key=lambda r: r["f_opt"])
        return {"x_opt": best["x_opt"], "f_opt": best["f_opt"]}
    except Exception as e:
        print(f"  WARNING: failed to load {path}: {e}")
        return default


def main():
    t0 = time.time()
    print("[start] ladder v2 (multi-seed, per-weekday breakdown)")

    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes = get_speed_lane(graph, links)
    print(f"  N = {N:,}")

    print("  loading raw + smooth priors ...")
    raw_rows    = load_prior(RAW_NPZ, N, lid_to_idx)
    smooth_rows = load_prior(SMOOTH_NPZ, N, lid_to_idx)

    sp_tuned = load_tuned(RESULTS / "sp_tune.json",
                          {"x_opt": [ALPHA_GEOM, 0.0, 0.0], "f_opt": None})
    tp_tuned = load_tuned(RESULTS / "tp_tune_v2.json",
                          {"x_opt": [0.1019, 0.1014, 0.0, 0.0], "f_opt": None})
    print(f"  SP tuned: alpha={sp_tuned['x_opt'][0]:.4f} "
          f"a_s={sp_tuned['x_opt'][1]:+.3f} a_l={sp_tuned['x_opt'][2]:+.3f}")
    print(f"  TP tuned: beta={tp_tuned['x_opt'][0]:.4f} rho={tp_tuned['x_opt'][1]:.4f} "
          f"a_s={tp_tuned['x_opt'][2]:+.3f} a_l={tp_tuned['x_opt'][3]:+.3f}")

    # Build kernels we will reuse across seeds
    print("  building uniform P_cs ...")
    w_uniform = np.ones(N, dtype=np.float64)
    P_uniform = build_phase_matrix(graph, links, lid_to_idx, w_uniform)

    # Single-phase kernels (speed-only, lanes-only, both)
    def make_sp(a_s, a_l):
        w = np.ones(N, dtype=np.float64)
        if a_s != 0: w = w * np.power(speeds, a_s)
        if a_l != 0: w = w * np.power(lanes, a_l)
        return build_phase_matrix(graph, links, lid_to_idx, w)

    print("  building SP speed-only kernel ...")
    P_sp_s = make_sp(sp_tuned["x_opt"][1], 0.0)
    print("  building SP lanes-only kernel ...")
    P_sp_l = make_sp(0.0, sp_tuned["x_opt"][2])
    print("  building SP speed+lanes kernel ...")
    P_sp_sl = make_sp(sp_tuned["x_opt"][1], sp_tuned["x_opt"][2])

    # Two-phase kernels (using tuned a_s, a_l)
    print("  building TP kernels ...")
    a_s_tp = tp_tuned["x_opt"][2]
    a_l_tp = tp_tuned["x_opt"][3]
    w_up   = np.ones(N, dtype=np.float64)
    w_down = np.ones(N, dtype=np.float64)
    if a_s_tp != 0:
        w_up   = w_up   * np.power(speeds, +a_s_tp)
        w_down = w_down * np.power(speeds, -a_s_tp)
    if a_l_tp != 0:
        w_up   = w_up   * np.power(lanes, +a_l_tp)
        w_down = w_down * np.power(lanes, -a_l_tp)
    P_tp_up   = build_phase_matrix(graph, links, lid_to_idx, w_up)
    P_tp_down = build_phase_matrix(graph, links, lid_to_idx, w_down)

    # Fixed PR vectors that depend only on (alpha, uniform prior)
    print("  precomputing vanilla PR vectors ...")
    v_v1 = sp_iter(P_uniform, np.full(N, 1.0/N), ALPHA_GOOGLE, TOL, MAX_ITER * 2)
    v_v2 = sp_iter(P_uniform, np.full(N, 1.0/N), ALPHA_GEOM,   TOL, MAX_ITER * 2)

    # Climatology slot index
    slot_to_times = defaultdict(list)
    for t in raw_rows.keys():
        dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        slot_to_times[(dt.weekday(), dt.hour, dt.minute)].append(t)

    def climatology(sib):
        dt = datetime.strptime(sib, "%Y-%m-%d %H:%M:%S")
        slot = (dt.weekday(), dt.hour, dt.minute)
        Es = [raw_rows[t] for t in slot_to_times[slot] if t != sib and t in raw_rows]
        return np.mean(Es, axis=0) if Es else None

    all_results = {}
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        pairs = sample_pairs(sorted(smooth_rows.keys()), seed, BPW)
        pairs = [(t, s) for t, s in pairs if t in smooth_rows and s in raw_rows]
        print(f"  pairs: {len(pairs)}")

        Es_pred_in = [smooth_rows[t] for t, _ in pairs]
        Es_target  = [raw_rows[s]    for _, s in pairs]
        top_idx    = [np.argsort(E)[::-1][:TOP_K] for E in Es_target]
        wd_of_pair = [datetime.strptime(s, "%Y-%m-%d %H:%M:%S").weekday()
                      for _, s in pairs]

        def eval_v(name, pred_fn):
            ts = time.time()
            top, ov, wd_top, wd_ov = [], [], defaultdict(list), defaultdict(list)
            for i in range(len(pairs)):
                v = pred_fn(i)
                if v is None:
                    continue
                rel = np.abs(Es_target[i] - v) / (Es_target[i] + EPS)
                tm = float(rel[top_idx[i]].mean())
                om = float(rel.mean())
                top.append(tm); ov.append(om)
                wd_top[wd_of_pair[i]].append(tm)
                wd_ov[wd_of_pair[i]].append(om)
            return {
                "top100_mre_mean": float(np.mean(top)),
                "top100_mre_median": float(np.median(top)),
                "overall_mre_mean": float(np.mean(ov)),
                "overall_mre_median": float(np.median(ov)),
                "n_bins": len(top),
                "by_weekday": {
                    str(wd): {"top100": float(np.mean(wd_top[wd])),
                              "overall": float(np.mean(wd_ov[wd])),
                              "n": len(wd_top[wd])}
                    for wd in sorted(wd_top.keys())
                },
                "elapsed_s": time.time() - ts,
            }

        seed_summary = {}
        variants = [
            ("v0_climatology",       lambda i: climatology(pairs[i][1])),
            ("v1_sp_unif_a015_unif", lambda i: v_v1),
            ("v2_sp_unif_a005_unif", lambda i: v_v2),
            ("v3_sp_unif_a005_pop",
             lambda i: sp_iter(P_uniform, Es_pred_in[i], ALPHA_GEOM, TOL, MAX_ITER)),
            ("v4_sp_speed_pop",
             lambda i: sp_iter(P_sp_s, Es_pred_in[i], sp_tuned["x_opt"][0],
                                TOL, MAX_ITER)),
            ("v5_sp_lanes_pop",
             lambda i: sp_iter(P_sp_l, Es_pred_in[i], sp_tuned["x_opt"][0],
                                TOL, MAX_ITER)),
            ("v6_sp_speedlanes_pop",
             lambda i: sp_iter(P_sp_sl, Es_pred_in[i], sp_tuned["x_opt"][0],
                                TOL, MAX_ITER)),
            ("v7_tp_speedlanes_pop",
             lambda i: tp_iter(P_tp_up, P_tp_down, Es_pred_in[i],
                                tp_tuned["x_opt"][0], tp_tuned["x_opt"][1],
                                TOL, MAX_ITER)),
        ]
        for name, fn in variants:
            print(f"  [{name}] ...", flush=True)
            s = eval_v(name, fn)
            seed_summary[name] = s
            print(f"    top100={s['top100_mre_mean']:.4f}  "
                  f"overall={s['overall_mre_mean']:.4f}  "
                  f"({s['elapsed_s']:.0f}s)", flush=True)

        all_results[str(seed)] = seed_summary

        # Save partial after each seed.
        with open(RESULTS / "ladder_eval_v2.json", "w") as f:
            json.dump({
                "seeds": SEEDS, "bpw": BPW, "top_k": TOP_K, "eps": EPS,
                "sp_tuned": sp_tuned, "tp_tuned": tp_tuned,
                "results_by_seed": all_results,
                "elapsed_total": time.time() - t0,
            }, f, indent=2)

    # Cross-seed mean / std per variant
    print("\n=== cross-seed summary ===")
    variant_names = list(all_results[str(SEEDS[0])].keys())
    cross = {}
    for v in variant_names:
        tops = [all_results[str(s)][v]["top100_mre_mean"] for s in SEEDS]
        overs = [all_results[str(s)][v]["overall_mre_mean"] for s in SEEDS]
        cross[v] = {
            "top100_mean": float(np.mean(tops)),
            "top100_std":  float(np.std(tops)),
            "overall_mean": float(np.mean(overs)),
            "overall_std":  float(np.std(overs)),
        }
        print(f"  {v:<28s}  top100={cross[v]['top100_mean']:.4f} "
              f"+/- {cross[v]['top100_std']:.4f}   "
              f"overall={cross[v]['overall_mean']:.4f} "
              f"+/- {cross[v]['overall_std']:.4f}")

    with open(RESULTS / "ladder_eval_v2.json", "w") as f:
        json.dump({
            "seeds": SEEDS, "bpw": BPW, "top_k": TOP_K, "eps": EPS,
            "sp_tuned": sp_tuned, "tp_tuned": tp_tuned,
            "results_by_seed": all_results,
            "cross_seed": cross,
            "elapsed_total": time.time() - t0,
        }, f, indent=2)
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
