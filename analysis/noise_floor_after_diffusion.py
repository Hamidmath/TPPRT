"""Generate cross-week noise-floor histograms AFTER diffusion smoothing
for gamma in {0.1, 0.2, 0.26}, with alpha = 0.01.

Same construction as fig1_cross_week_noise_floor: per (link, bin-of-week)
compute std across the 4 weeks of the smoothed probability matrix,
histogram those stds. One figure per gamma + a 3-panel comparison.
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
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz

OUT = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project/results/gamma_calibration")
FIGDIR = OUT / "figs"
FIGDIR.mkdir(parents=True, exist_ok=True)

N_BINS_DAY = 288
N_BINS_WEEK = 7 * N_BINS_DAY
T_TOTAL = 8640

GAMMAS = [0.1, 0.2, 0.26]
ALPHA = 0.01


def load():
    print("[load] graph + popularity ...")
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
    proj = np.array(
        [lid_to_idx.get(lid, -1) for lid in pop_link_ids], dtype=np.int64
    )
    valid_mask = proj >= 0
    coo = matrix.tocoo()
    sel = valid_mask[coo.col]
    rows_sel = coo.row[sel]
    cols_sel = proj[coo.col[sel]]
    data_sel = coo.data[sel].astype(np.float32)
    T = len(b["times"])
    M_raw = csr_matrix((data_sel, (rows_sel, cols_sel)),
                       shape=(T, N), dtype=np.float32)
    print(f"  N={N:,}, T={T}, P nnz={P.nnz:,}, M_raw nnz={M_raw.nnz:,}")
    return P, M_raw, T, N


def smooth(Ma_dense, PMa_dense, gamma):
    S = (1.0 - gamma) * Ma_dense + gamma * PMa_dense.T
    S = S.astype(np.float32)
    row_sums = S.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    S /= row_sums
    return S


def cross_week_stds(M, n_weeks=4):
    """Return (stds_2d (n_per_week, N), active_mask (n_per_week, N))."""
    n_per_week = N_BINS_WEEK
    full = n_weeks * n_per_week
    M_3d = M[:full].reshape(n_weeks, n_per_week, M.shape[1])
    stds = M_3d.std(axis=0, ddof=1)
    active = (M_3d > 0).any(axis=0)
    return stds, active


def plot_hist(stds, active, title, out_path):
    s = stds[active]
    med = float(np.median(s))
    mean = float(s.mean())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.clip(s, 1e-9, 1e-2),
            bins=np.logspace(-9, -2, 80), color="#1f6fb4", alpha=0.85)
    ax.axvline(med, color="red", ls="--", lw=1.4,
               label=f"median = {med:.2e}")
    ax.axvline(mean, color="orange", ls=":", lw=1.4,
               label=f"mean = {mean:.2e}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("cross-week std (probability units, per (link, bin-of-week))")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return {"median": med, "mean": mean,
            "p90": float(np.percentile(s, 90)),
            "p95": float(np.percentile(s, 95)),
            "n_active": int(active.sum())}


def main():
    t0 = time.time()
    P, M_raw, T, N = load()

    print(f"[precompute] Ma = M_raw + {ALPHA} (dense float32)...")
    t_a = time.time()
    Ma = M_raw.toarray() + np.float32(ALPHA)
    print(f"  done in {time.time()-t_a:.0f}s")

    print(f"[precompute] PMa = P @ Ma.T (dense)...")
    t_a = time.time()
    PMa = P @ Ma.T
    print(f"  done in {time.time()-t_a:.0f}s")

    summaries = {}
    panels = []
    for gamma in GAMMAS:
        print(f"[smooth] gamma = {gamma}...")
        t_s = time.time()
        S = smooth(Ma, PMa, gamma)
        print(f"  smoothed in {time.time()-t_s:.0f}s")

        stds, active = cross_week_stds(S, 4)
        title = (f"Cross-week noise floor AFTER diffusion  "
                 f"($\\gamma={gamma:.2f}$, $\\alpha={ALPHA}$)")
        out_path = FIGDIR / f"fig_noise_floor_gamma{str(gamma).replace('.', 'p')}.png"
        info = plot_hist(stds, active, title, out_path)
        summaries[gamma] = info
        panels.append((gamma, stds, active, info))
        print(f"  median = {info['median']:.3e}, mean = {info['mean']:.3e}")
        print(f"  saved {out_path}")
        del S
        gc.collect()

    # Combined 1x3 panel
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (gamma, stds, active, info) in zip(axes, panels):
        s = stds[active]
        ax.hist(np.clip(s, 1e-9, 1e-2),
                bins=np.logspace(-9, -2, 80),
                color="#1f6fb4", alpha=0.85)
        ax.axvline(info["median"], color="red", ls="--", lw=1.4,
                   label=f"median = {info['median']:.2e}")
        ax.axvline(info["mean"], color="orange", ls=":", lw=1.4,
                   label=f"mean = {info['mean']:.2e}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("cross-week std (probability units)")
        ax.set_title(f"$\\gamma={gamma:.2f}, \\alpha={ALPHA}$")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("count")
    fig.suptitle(f"Cross-week noise floor of the diffused popularity matrix",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_noise_floor_diffused_3panel.png", dpi=130)
    plt.close(fig)

    print()
    print("Summary medians:")
    for g, info in summaries.items():
        print(f"  gamma = {g:.2f}: median = {info['median']:.3e}, "
              f"mean = {info['mean']:.3e}, 90th = {info['p90']:.3e}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
