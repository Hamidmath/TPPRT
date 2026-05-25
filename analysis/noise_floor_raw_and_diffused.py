"""4-panel comparison: raw cross-week noise floor + diffused at gamma 0.1, 0.2, 0.26.

Saves a single side-by-side figure for easy reading.
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

N_BINS_WEEK = 7 * 288
T_TOTAL = 8640
GAMMAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ALPHA = 0.1


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


def cross_week_stds(M, n_weeks=4):
    n_per_week = N_BINS_WEEK
    full = n_weeks * n_per_week
    M_3d = M[:full].reshape(n_weeks, n_per_week, M.shape[1])
    stds = M_3d.std(axis=0, ddof=1)
    active = (M_3d > 0).any(axis=0)
    return stds, active


def smooth(Ma, PMa, gamma):
    S = (1.0 - gamma) * Ma + gamma * PMa.T
    S = S.astype(np.float32)
    rs = S.sum(axis=1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    S /= rs
    return S


def main():
    t0 = time.time()
    P, M_raw, T, N = load()

    # RAW: normalize row-wise
    print("[raw] normalising row-wise and computing cross-week std...")
    rs = np.asarray(M_raw.sum(axis=1)).ravel()
    inv = np.where(rs > 0, 1.0 / rs, 0.0).astype(np.float32)
    from scipy.sparse import diags
    M_norm = diags(inv) @ M_raw
    M_norm = M_norm.toarray().astype(np.float32)  # dense for std
    stds_raw, active_raw = cross_week_stds(M_norm, 4)
    del M_norm; gc.collect()

    # Pre-compute Ma, PMa once for diffused
    print(f"[precompute] Ma = M_raw + {ALPHA} ...")
    Ma = M_raw.toarray() + np.float32(ALPHA)
    print(f"[precompute] PMa = P @ Ma.T ...")
    PMa = P @ Ma.T

    diffused_data = []
    for gamma in GAMMAS:
        print(f"[smooth] gamma={gamma}...")
        S = smooth(Ma, PMa, gamma)
        stds, _ = cross_week_stds(S, 4)
        # Reuse the raw active mask so diffused panels show only the
        # originally non-zero (link, bow) cells. This filters out the
        # 195M alpha-only cells.
        diffused_data.append((gamma, stds, active_raw))
        del S; gc.collect()

    # ----- N-panel plot (raw + len(GAMMAS) gammas) -----
    n_panels = 1 + len(GAMMAS)
    if n_panels <= 4:
        nrows, ncols = 1, n_panels
    else:
        ncols = 4
        nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4.7 * nrows),
                              sharey=True, squeeze=False)
    bins = np.logspace(-9, -2, 80)
    panels = [
        ("RAW (no diffusion)", stds_raw, active_raw, None),
    ] + [(f"$\\gamma={g}$, $\\alpha={ALPHA}$\n(raw-active cells only)",
          st, ac, g)
         for (g, st, ac) in diffused_data]
    axes_flat = axes.flatten()
    # hide unused axes
    for ax in axes_flat[len(panels):]:
        ax.axis("off")
    for ax, (title, st, ac, _) in zip(axes_flat, panels):
        s = st[ac]
        med = float(np.median(s))
        mn = float(s.mean())
        ax.hist(np.clip(s, 1e-9, 1e-2), bins=bins,
                color="#1f6fb4", alpha=0.85)
        ax.axvline(med, color="red", ls="--", lw=1.4,
                   label=f"median = {med:.2e}")
        ax.axvline(mn, color="orange", ls=":", lw=1.4,
                   label=f"mean   = {mn:.2e}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("cross-week std (prob. units)")
        ax.set_title(f"{title}\n(N active = {int(ac.sum()):,})", fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)
    for r in range(nrows):
        axes[r][0].set_ylabel("count")
    fig.suptitle(f"Cross-week noise floor: raw vs diffused "
                 f"(alpha = {ALPHA}, raw-active cells only)",
                 fontsize=13)
    fig.tight_layout()
    alpha_tag = str(ALPHA).replace('.', 'p')
    g_tag = f"g{int(min(GAMMAS)*100):03d}_to_g{int(max(GAMMAS)*100):03d}"
    out = FIGDIR / f"fig_noise_floor_raw_{g_tag}_alpha{alpha_tag}_rawactive.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")

    print()
    print("Medians:")
    print(f"  raw         : {float(np.median(stds_raw[active_raw])):.3e}")
    for (g, st, ac) in diffused_data:
        print(f"  gamma={g:<4}: {float(np.median(st[ac])):.3e}")
    print(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
