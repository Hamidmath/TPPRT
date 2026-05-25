"""Fine-tune (alpha_s, alpha_l) with beta = rho = 0.5 fixed,
scored on the top-10 and top-20 busiest clusters (saved earlier).

9 x 9 grid for the two alphas (-1.0, -0.75, -0.5, -0.25, 0,
0.25, 0.5, 0.75, 1.0). 81 combinations total. Chain runs are shared
across both masks per (alpha_s, alpha_l).
"""
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz
from core import pagerank
from core.pagerank import build_phase_kernels, power_iteration

OUT = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project/results/event_sept15")

GAMMA, ALPHA = 0.20, 0.01
import os as _os
BETA = float(_os.environ.get("BETA_RHO", 0.5))
RHO  = float(_os.environ.get("BETA_RHO", 0.5))
TOL, MAX_ITER = 1e-6, 200
TAG = f"b{int(round(BETA*100)):03d}"   # e.g. b050 or b005
WINDOW_START, WINDOW_END = "19:00:00", "19:55:00"

# 9 x 9 finer grid for the two alphas
GRID_ALPHA_S = np.linspace(-1.0, 1.0, 9).tolist()
GRID_ALPHA_L = np.linspace(-1.0, 1.0, 9).tolist()


def build_diffusion_P(graph_data, links_order):
    id_to_idx = {lid: i for i, lid in enumerate(links_order)}
    adj = graph_data.get('adjacency', {})
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

    print("[load] cluster masks ...")
    mask10 = np.load(OUT / "top10_busiest_mask.npy")
    mask20 = np.load(OUT / "top20_busiest_mask.npy")
    print(f"  top-10: {int(mask10.sum())} links")
    print(f"  top-20: {int(mask20.sum())} links")

    bins_p = bins_window("2018-09-08")
    bins_t = bins_window("2018-09-15")
    eps = 1e-6

    nA = len(GRID_ALPHA_S); nL = len(GRID_ALPHA_L)
    print(f"[sweep] {nA} x {nL} = {nA*nL} (alpha_s, alpha_l) combos "
          f"with beta = rho = {BETA}")
    grid10 = np.full((nA, nL), np.nan, dtype=np.float64)
    grid20 = np.full((nA, nL), np.nan, dtype=np.float64)
    best10 = dict(mre=float("inf"), params=None)
    best20 = dict(mre=float("inf"), params=None)
    t_g = time.time(); done = 0
    for i, a_s in enumerate(GRID_ALPHA_S):
        for j, a_l in enumerate(GRID_ALPHA_L):
            pagerank.PARAMS["alpha_s"] = a_s
            pagerank.PARAMS["alpha_l"] = a_l
            Pu, Pd = build_phase_kernels(graph_data)
            v10, v20 = [], []
            for bp, bt in zip(bins_p, bins_t):
                E_in = get_E(bp); E_tg = get_E(bt)
                if E_in is None or E_tg is None: continue
                vu, vd, _ = power_iteration(Pu, Pd, E_in, BETA, RHO,
                                               TOL, MAX_ITER)
                v = vu + vd; s = v.sum()
                if s > 0: v /= s
                err = np.abs(E_tg - v) / (E_tg + eps)
                v10.append(float(err[mask10].mean()))
                v20.append(float(err[mask20].mean()))
            m10 = float(np.mean(v10)); m20 = float(np.mean(v20))
            grid10[i, j] = m10
            grid20[i, j] = m20
            if m10 < best10["mre"]:
                best10 = dict(mre=m10, params=dict(alpha_s=a_s, alpha_l=a_l))
            if m20 < best20["mre"]:
                best20 = dict(mre=m20, params=dict(alpha_s=a_s, alpha_l=a_l))
            done += 1
            if done % 9 == 0:
                print(f"  {done}/{nA*nL} t={time.time()-t_g:.0f}s "
                      f"best10={best10['mre']:.4f} best20={best20['mre']:.4f}",
                      flush=True)

    print()
    print("Best with beta = rho = 0.5:")
    print(f"  top-10  : MRE = {best10['mre']:.4f}  at alpha_s = {best10['params']['alpha_s']:+.3f}, alpha_l = {best10['params']['alpha_l']:+.3f}")
    print(f"  top-20  : MRE = {best20['mre']:.4f}  at alpha_s = {best20['params']['alpha_s']:+.3f}, alpha_l = {best20['params']['alpha_l']:+.3f}")

    # Heatmap figure
    def plot_heatmap(grid, best, k, save_path):
        fig, ax = plt.subplots(figsize=(8, 6.5))
        im = ax.imshow(grid, origin="lower",
                        extent=[GRID_ALPHA_L[0], GRID_ALPHA_L[-1],
                                 GRID_ALPHA_S[0], GRID_ALPHA_S[-1]],
                        aspect="auto", cmap="viridis_r")
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(f"top-{k} cluster MRE")
        ax.set_xlabel("alpha_l")
        ax.set_ylabel("alpha_s")
        ax.set_title(f"Top-{k} cluster, beta = rho = {BETA}\n"
                      f"sweep over (alpha_s, alpha_l); best at "
                      f"({best['params']['alpha_s']:+.2f}, "
                      f"{best['params']['alpha_l']:+.2f}) "
                      f"MRE = {best['mre']:.4f}")
        # mark best
        ax.plot(best["params"]["alpha_l"], best["params"]["alpha_s"],
                "r*", markersize=18, markeredgecolor="white", markeredgewidth=1.0)
        # annotate cell values
        for i, a_s in enumerate(GRID_ALPHA_S):
            for j, a_l in enumerate(GRID_ALPHA_L):
                ax.text(a_l, a_s, f"{grid[i,j]:.2f}",
                         ha="center", va="center",
                         color=("white" if grid[i,j] > grid.mean() else "black"),
                         fontsize=7)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  saved {save_path}")

    plot_heatmap(grid10, best10, 10, OUT / f"fig_topk_alpha_tune_top10_{TAG}.png")
    plot_heatmap(grid20, best20, 20, OUT / f"fig_topk_alpha_tune_top20_{TAG}.png")

    # Save JSON
    with open(OUT / f"topk_alpha_tune_results_{TAG}.json", "w") as f:
        json.dump(dict(
            beta=BETA, rho=RHO, gamma=GAMMA, alpha_laplace=ALPHA,
            grid_alpha_s=GRID_ALPHA_S,
            grid_alpha_l=GRID_ALPHA_L,
            top10_grid=grid10.tolist(),
            top20_grid=grid20.tolist(),
            best_top10=best10,
            best_top20=best20,
        ), f, indent=2)
    print(f"saved {OUT/'topk_alpha_tune_results.json'}")
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
