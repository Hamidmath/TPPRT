"""Top-10, top-20, top-30 busiest-cluster experiment with a
constrained 4D tuning grid:

  beta  in [0.05, 0.075, 0.102]
  rho   in [0.05, 0.075, 0.101]
  alpha_s in [-1.0, -0.5, 0.0, 0.5, 1.0]
  alpha_l in [-1.0, -0.5, 0.0, 0.5, 1.0]

= 3 x 3 x 5 x 5 = 225 combos, 12 event bins each, scored on all
three saved masks in one pass.

Reports: cluster | size | event d2d | d2m default | d2m tuned |
reduction | best params.
"""
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz
from core import pagerank
from core.pagerank import build_phase_kernels, power_iteration

OUT = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project/results/event_sept15")

GAMMA, ALPHA = 0.20, 0.01
BETA_DEF, RHO_DEF = 0.102, 0.101
TOL, MAX_ITER = 1e-6, 200
WINDOW_START, WINDOW_END = "19:00:00", "19:55:00"

GRID_BETA   = [0.05, 0.075, 0.102]
GRID_RHO    = [0.05, 0.075, 0.101]
GRID_ALPHA_S = [-1.0, -0.5, 0.0, 0.5, 1.0]
GRID_ALPHA_L = [-1.0, -0.5, 0.0, 0.5, 1.0]

TOP_KS = [10, 20, 30]


def build_diffusion_P(graph_data, links_order):
    id_to_idx = {lid: i for i, lid in enumerate(links_order)}
    adj = graph_data.get("adjacency", {})
    N = len(links_order)
    row, col, data = [], [], []
    for i, lid in enumerate(links_order):
        succ = [id_to_idx[ol] for ol in adj.get(lid, []) if ol in id_to_idx]
        if not succ: continue
        w = 1.0 / len(succ)
        for j in succ:
            row.append(i); col.append(j); data.append(w)
    return csr_matrix((data, (row, col)), shape=(N, N), dtype=np.float64)


def bins_window(day_str):
    start = datetime.strptime(f"{day_str} {WINDOW_START}", "%Y-%m-%d %H:%M:%S")
    end   = datetime.strptime(f"{day_str} {WINDOW_END}",   "%Y-%m-%d %H:%M:%S")
    out, cur = [], start
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d %H:%M:%S"))
        cur += timedelta(minutes=5)
    return out


def main():
    t0 = time.time()
    print("[load] graph + popularity ...")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links = list(graph_data["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    P_up_cs, P_down_cs = build_phase_kernels(graph_data)
    P_diff = build_diffusion_P(graph_data, links)

    b = load_popularity_npz(str(config.DATA_DIR / "popularity_results_osm.npz"))
    matrix = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_link_ids = b["link_ids"]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_link_ids], dtype=np.int64)
    valid = proj >= 0
    time_index = {t: i for i, t in enumerate(times)}

    def get_E(time_str):
        ti = time_index.get(time_str)
        if ti is None: return None
        row = matrix.getrow(ti).toarray().ravel().astype(np.float64)
        C = np.zeros(N, dtype=np.float64)
        np.add.at(C, proj[valid], row[valid])
        c = C + ALPHA
        c_diff = (1.0 - GAMMA) * c + GAMMA * (P_diff @ c)
        s = c_diff.sum()
        return c_diff / s if s > 0 else None

    print("[load] masks ...")
    masks = {k: np.load(OUT / f"top{k}_busiest_mask.npy") for k in TOP_KS}
    for k in TOP_KS:
        print(f"  top-{k}: {int(masks[k].sum())} links")

    eps = 1e-6
    bins_p = bins_window("2018-09-08")
    bins_t = bins_window("2018-09-15")

    # -------- default + d2d on event pair --------
    print("\n[default] running default chain (beta=0.102, rho=0.101) ...")
    default_d2d = {k: [] for k in TOP_KS}
    default_d2m = {k: [] for k in TOP_KS}
    for bp, bt in zip(bins_p, bins_t):
        E_in = get_E(bp); E_tg = get_E(bt)
        if E_in is None or E_tg is None: continue
        vu, vd, _ = power_iteration(P_up_cs, P_down_cs, E_in,
                                       BETA_DEF, RHO_DEF, TOL, MAX_ITER)
        v = vu + vd; s = v.sum()
        if s > 0: v /= s
        denom = E_tg + eps
        err_d2d = np.abs(E_tg - E_in) / denom
        err_d2m = np.abs(E_tg - v)    / denom
        for k in TOP_KS:
            default_d2d[k].append(float(err_d2d[masks[k]].mean()))
            default_d2m[k].append(float(err_d2m[masks[k]].mean()))
    d2d_mean = {k: float(np.mean(default_d2d[k])) for k in TOP_KS}
    d2m_def  = {k: float(np.mean(default_d2m[k])) for k in TOP_KS}
    for k in TOP_KS:
        print(f"  top-{k}: d2d={d2d_mean[k]:.4f}  d2m default={d2m_def[k]:.4f}")

    # -------- 4D constrained sweep --------
    print(f"\n[sweep] {len(GRID_BETA)} beta x {len(GRID_RHO)} rho x "
          f"{len(GRID_ALPHA_S)} a_s x {len(GRID_ALPHA_L)} a_l "
          f"= {len(GRID_BETA)*len(GRID_RHO)*len(GRID_ALPHA_S)*len(GRID_ALPHA_L)} combos")
    best = {k: dict(mre=float("inf"), params=None) for k in TOP_KS}
    grid_records = []
    done = 0; t_g = time.time()
    for a_s in GRID_ALPHA_S:
        for a_l in GRID_ALPHA_L:
            pagerank.PARAMS["alpha_s"] = a_s
            pagerank.PARAMS["alpha_l"] = a_l
            Pu, Pd = build_phase_kernels(graph_data)
            for beta in GRID_BETA:
                for rho in GRID_RHO:
                    per_k = {k: [] for k in TOP_KS}
                    for bp, bt in zip(bins_p, bins_t):
                        E_in = get_E(bp); E_tg = get_E(bt)
                        if E_in is None or E_tg is None: continue
                        vu, vd, _ = power_iteration(Pu, Pd, E_in,
                                                      beta, rho, TOL, MAX_ITER)
                        v = vu + vd; s = v.sum()
                        if s > 0: v /= s
                        err = np.abs(E_tg - v) / (E_tg + eps)
                        for k in TOP_KS:
                            per_k[k].append(float(err[masks[k]].mean()))
                    rec = dict(alpha_s=a_s, alpha_l=a_l, beta=beta, rho=rho)
                    for k in TOP_KS:
                        m = float(np.mean(per_k[k]))
                        rec[f"top{k}_d2m"] = m
                        if m < best[k]["mre"]:
                            best[k] = dict(mre=m, params=dict(
                                alpha_s=a_s, alpha_l=a_l, beta=beta, rho=rho))
                    grid_records.append(rec)
                    done += 1
                    if done % 25 == 0:
                        print(f"  {done}/225 t={time.time()-t_g:.0f}s "
                              f"best10={best[10]['mre']:.4f} "
                              f"best20={best[20]['mre']:.4f} "
                              f"best30={best[30]['mre']:.4f}", flush=True)

    # -------- summary --------
    print()
    print("=" * 100)
    print("Summary on Sept 8 -> Sept 15 event pair  (constrained 4D tune)")
    print("  beta  in [0.05, 0.075, 0.102]  ;  rho  in [0.05, 0.075, 0.101]")
    print(f"  alpha_s in {GRID_ALPHA_S}")
    print(f"  alpha_l in {GRID_ALPHA_L}")
    print("=" * 100)
    print(f"  {'cluster':<10s} {'size':>5s}  "
          f"{'evt d2d':>9s}  {'d2m def':>9s}  {'d2m tuned':>10s}  "
          f"{'reduction':>10s}  {'best (a_s, a_l, beta, rho)':>30s}")
    summary = []
    for k in TOP_KS:
        red = (1 - best[k]["mre"] / d2m_def[k]) * 100.0
        bp = best[k]["params"]
        ps = (f"({bp['alpha_s']:+.2f}, {bp['alpha_l']:+.2f}, "
              f"{bp['beta']:.3f}, {bp['rho']:.3f})")
        print(f"  top-{k:<6d} {k:>5d}  "
              f"{d2d_mean[k]:>9.4f}  {d2m_def[k]:>9.4f}  "
              f"{best[k]['mre']:>10.4f}  {red:>9.1f}%  {ps:>30s}")
        summary.append(dict(cluster=f"top-{k}", size=k,
                             event_d2d=d2d_mean[k],
                             d2m_default=d2m_def[k],
                             d2m_tuned=best[k]["mre"],
                             reduction_pct=red,
                             best_params=best[k]["params"]))

    with open(OUT / "topk_constrained_4d_results.json", "w") as f:
        json.dump(dict(
            grid_beta=GRID_BETA, grid_rho=GRID_RHO,
            grid_alpha_s=GRID_ALPHA_S, grid_alpha_l=GRID_ALPHA_L,
            default_d2d=d2d_mean, default_d2m=d2m_def,
            summary=summary, grid=grid_records,
        ), f, indent=2)
    print(f"\nsaved {OUT/'topk_constrained_4d_results.json'}")
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
