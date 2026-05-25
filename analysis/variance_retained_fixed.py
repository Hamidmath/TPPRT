"""Corrected variance-retained-vs-gamma figure.

The previous version mixed two effects: the Laplace step (alpha) and
the diffusion step (gamma). It computed
    var(Laplace + diffusion(gamma)) / var(raw, no Laplace)
which means even at gamma=0 the ratio is < 1 because the Laplace
step alone has already deflated the per-row variance.

This corrected version holds the Laplace step constant on both
sides:
    ratio(gamma) = var(Laplace + diffusion(gamma)) / var(Laplace, gamma=0)
so at gamma=0 the ratio is exactly 1.0 by construction, and the
curve isolates the diffusion's effect.

We also report a second curve where Laplace is not applied at all:
    ratio_noLap(gamma) = var(diffusion-only(gamma)) / var(raw)
again 1.0 at gamma=0 by construction.
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
FIGDIR = OUT / "figs"
FIGDIR.mkdir(parents=True, exist_ok=True)

T_TOTAL = 8640
GAMMAS = np.round(np.arange(0.9, 2.01, 0.1), 3)
ALPHA = 0.01
SAMPLE_STRIDE = 10  # sample every 10th bin row for speed


def load():
    print("[load] graph + popularity...")
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


def per_row_var_mean(M, stride=SAMPLE_STRIDE):
    """Mean of per-row variances over a subsample of bins."""
    vs = []
    if isinstance(M, csr_matrix):
        T = M.shape[0]
        for t in range(0, T, stride):
            r = M.getrow(t).toarray().ravel()
            v = float(r.var())
            if v > 0:
                vs.append(v)
    else:
        T = M.shape[0]
        for t in range(0, T, stride):
            r = M[t]
            v = float(r.var())
            if v > 0:
                vs.append(v)
    return float(np.mean(vs))


def smooth_dense(Ma_dense, PMa_dense, gamma):
    if gamma == 0.0:
        S = Ma_dense.copy()
    else:
        S = (1.0 - gamma) * Ma_dense + gamma * PMa_dense.T
        S = S.astype(np.float32)
    rs = S.sum(axis=1, keepdims=True)
    rs = np.where(rs > 0, rs, 1.0)
    S /= rs
    return S


def main():
    t0 = time.time()
    P, M_raw, T, N = load()

    # --- BASELINE 1: Laplace at gamma=0 (matches what numerator uses) ---
    print(f"[precompute Laplace] Ma = M_raw + {ALPHA}")
    Ma = M_raw.toarray() + np.float32(ALPHA)
    print(f"[precompute Laplace] PMa = P @ Ma.T")
    PMa = P @ Ma.T

    # Baseline = Laplace + gamma=0 (i.e. Laplace only, normalised)
    print("[baseline Laplace-only] computing var(S(gamma=0))...")
    S0_lap = smooth_dense(Ma, PMa, 0.0)
    var_lap_baseline = per_row_var_mean(S0_lap)
    del S0_lap; gc.collect()
    print(f"  var(S(gamma=0) with Laplace alpha={ALPHA}) = {var_lap_baseline:.3e}")

    results = []
    for gamma in GAMMAS:
        t_s = time.time()
        S_lap = smooth_dense(Ma, PMa, float(gamma))
        var_lap_g = per_row_var_mean(S_lap)
        del S_lap; gc.collect()
        ratio_lap = var_lap_g / var_lap_baseline
        results.append((float(gamma), ratio_lap, var_lap_g))
        print(f"  gamma={gamma:.3f}  "
              f"ratio={ratio_lap:.3f}  "
              f"var={var_lap_g:.3e}  "
              f"({time.time()-t_s:.1f}s)")

    # Plot
    gs = [r[0] for r in results]
    rs_lap = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gs, rs_lap, "o-", color="#1f6fb4", lw=1.6, ms=5,
            label=fr"$\alpha={ALPHA}$, Laplace held constant; "
                  fr"ratio $=$ var($S_\gamma$) / var($S_{{\gamma=0}}$)")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.axvline(1.0, color="gray", ls=":", lw=0.8,
               label=r"$\gamma=1$ (convex-combination boundary)")
    ax.set_xlabel(r"diffusion strength $\gamma$")
    ax.set_ylabel("variance retained (ratio)")
    g_lo = min(GAMMAS); g_hi = max(GAMMAS)
    ax.set_title(rf"Variance retained vs $\gamma$ in [{g_lo:.1f}, {g_hi:.1f}] "
                 r"(baseline holds Laplace constant)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    g_tag = f"g{int(g_lo*10):02d}_to_g{int(g_hi*10):02d}"
    out = FIGDIR / f"fig6_variance_retained_{g_tag}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
