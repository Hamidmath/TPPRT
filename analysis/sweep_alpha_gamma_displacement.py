"""Sweep gamma at multiple alpha values; for each (gamma, alpha) report
the median smoothing displacement over active raw cells, and compare
to the cross-week noise floor.

Goal: find (gamma, alpha) such that median displacement >= noise floor.
"""
import gc
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, diags

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz

OUT = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project/results/gamma_calibration")
OUT.mkdir(parents=True, exist_ok=True)
FIGDIR = OUT / "figs"
FIGDIR.mkdir(exist_ok=True)

N_BINS_WEEK = 7 * 288
T_TOTAL = 8640
NOISE_FLOOR = 2.646e-4
ALPHAS = [0.0, 0.001, 0.01, 0.1]
GAMMAS = np.round(np.linspace(0.0, 1.0, 11), 3)  # 11 values 0..1


def load():
    print("[load] graph + popularity")
    with open(config.GRAPH_FILE) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    adj = graph.get("adjacency", {})
    row, col, data = [], [], []
    for lid, out_lids in adj.items():
        if lid not in lid_to_idx:
            continue
        i = lid_to_idx[lid]
        valids = [ol for ol in out_lids if ol in lid_to_idx]
        if not valids:
            continue
        w = 1.0 / len(valids)
        for ol in valids:
            row.append(i); col.append(lid_to_idx[ol]); data.append(w)
    P = csr_matrix((data, (row, col)), shape=(N, N), dtype=np.float32)
    b = load_popularity_npz(str(config.DATA_DIR / "popularity_results_osm.npz"))
    matrix = b["matrix"]
    pop_link_ids = b["link_ids"]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_link_ids], dtype=np.int64)
    valid_mask = proj >= 0
    coo = matrix.tocoo()
    sel = valid_mask[coo.col]
    rows_sel = coo.row[sel]
    cols_sel = proj[coo.col[sel]]
    data_sel = coo.data[sel].astype(np.float32)
    T = len(b["times"])
    M_raw = csr_matrix((data_sel, (rows_sel, cols_sel)),
                       shape=(T, N), dtype=np.float32)
    return P, M_raw, T, N


def main():
    t0 = time.time()
    P, M_raw, T, N = load()
    n_full = 4 * N_BINS_WEEK  # 8064

    print("[norm] computing M_norm and its first 4 weeks as dense...")
    rs = np.asarray(M_raw.sum(axis=1)).ravel()
    inv = np.where(rs > 0, 1.0 / rs, 0.0).astype(np.float32)
    M_norm = diags(inv) @ M_raw  # sparse
    M_norm_4w = M_norm[:n_full].toarray().astype(np.float32).reshape(
        4, N_BINS_WEEK, N
    )
    # Active mask: cell had at least one positive week
    active_3d = (M_norm_4w > 0).any(axis=0)  # (2016, N)
    n_active = int(active_3d.sum())
    print(f"  n_active = {n_active:,}")
    del M_norm; gc.collect()

    curves = {}
    for alpha in ALPHAS:
        print(f"\n[alpha={alpha}] precomputing Ma, PMa...")
        Ma = M_raw.toarray() + np.float32(alpha)
        PMa = P @ Ma.T
        sweep = []
        for gamma in GAMMAS:
            t_s = time.time()
            if gamma == 0.0:
                S = Ma.copy()
            else:
                S = (1.0 - gamma) * Ma + gamma * PMa.T
                S = S.astype(np.float32)
            rs2 = S.sum(axis=1, keepdims=True)
            rs2 = np.where(rs2 > 0, rs2, 1.0)
            S /= rs2
            S_4w = S[:n_full].reshape(4, N_BINS_WEEK, N)

            # mean of |S - M_norm| over 4 weeks, then median over active cells
            disp_per_cell = np.zeros((N_BINS_WEEK, N), dtype=np.float32)
            for w in range(4):
                disp_per_cell += np.abs(S_4w[w] - M_norm_4w[w])
            disp_per_cell /= 4.0
            med_disp = float(np.median(disp_per_cell[active_3d]))
            sweep.append((float(gamma), med_disp))
            print(f"  gamma={gamma:.2f}  med_disp={med_disp:.3e}  "
                  f"med/NF={med_disp/NOISE_FLOOR:.3f}  "
                  f"({time.time()-t_s:.1f}s)")
            del S, S_4w, disp_per_cell; gc.collect()
        curves[float(alpha)] = sweep
        del Ma, PMa; gc.collect()

    # Save raw numbers
    out_json = OUT / "alpha_gamma_displacement.json"
    with open(out_json, "w") as f:
        json.dump({
            "noise_floor": NOISE_FLOOR,
            "n_active": n_active,
            "curves": {str(a): v for a, v in curves.items()},
        }, f, indent=2)
    print(f"saved {out_json}")

    # Plot all 4 curves on one figure (linear y), and again (log y)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {0.0: "#c0392b", 0.001: "#f39c12", 0.01: "#1f6fb4",
              0.1: "#2e8b57"}
    for alpha in ALPHAS:
        gs = [x[0] for x in curves[float(alpha)]]
        ds = [x[1] for x in curves[float(alpha)]]
        for ax in axes:
            ax.plot(gs, ds, "o-", color=colors[alpha], lw=1.5, ms=5,
                    label=fr"$\alpha={alpha}$")
    for ax in axes:
        ax.axhline(NOISE_FLOOR, color="red", ls="--", lw=1.2,
                   label=fr"$\sigma^{{NF}} = {NOISE_FLOOR:.2e}$")
        ax.axhline(NOISE_FLOOR / 2, color="red", ls=":", lw=0.8,
                   label=fr"$\sigma^{{NF}}/2$")
        ax.axhline(2 * NOISE_FLOOR, color="red", ls=":", lw=0.8,
                   label=fr"$2\sigma^{{NF}}$")
        ax.set_xlabel(r"diffusion strength $\gamma$")
        ax.set_ylabel("median smoothing displacement (active cells)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="best")
    axes[0].set_title("Linear y")
    axes[1].set_yscale("log")
    axes[1].set_title("Log y")
    fig.suptitle("Median displacement vs. $\\gamma$, for several $\\alpha$")
    fig.tight_layout()
    out_fig = FIGDIR / "fig_alpha_gamma_displacement_sweep.png"
    fig.savefig(out_fig, dpi=130)
    plt.close(fig)
    print(f"saved {out_fig}")

    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
