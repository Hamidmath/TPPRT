"""Self-contained justification of the diffusion calibration.

Computes everything needed for a 1-3 page appendix that explains, from
scratch and without referencing the downstream chain:

1. Sparsity of the raw popularity matrix
   - per-bin zero fraction
   - per (link, bin-of-week) cell: how many weeks had zero?

2. Cross-week noise floor on the raw probability matrix
   - distribution and median (already in fig1; recompute for the report)
   - stratify cells into bands by raw cross-week std:
     SIGNAL  -- raw_std > 2 x noise_floor
     MIXED   -- noise_floor/2 <= raw_std <= 2 x noise_floor
     SUBNOISE-- raw_mean > 0 but raw_std < noise_floor/2
     ZERO    -- raw_mean == 0 across all 4 weeks

3. For each gamma in a small set, apply Laplace+diffusion and report
   per-band statistics:
     - cells in band
     - mean raw value
     - mean smoothed value
     - mean smoothing displacement |smooth - raw|
     - ratio (smoothed value / raw value) -- signal preservation
     - ratio (smoothing displacement / cell's natural cross-week std)

4. Plots
   - hist of raw_std per cell, with bands shaded
   - scatter (or 2D hist) raw_std vs smoothing displacement for one gamma
   - bar chart per-band displacement at each gamma
   - density of smoothed values for the ZERO-cell stratum

Saves results/justification/ tables and figs.
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

OUT = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project/results/justification")
OUT.mkdir(parents=True, exist_ok=True)
FIGDIR = OUT / "figs"
FIGDIR.mkdir(exist_ok=True)

N_BINS_WEEK = 7 * 288
T_TOTAL = 8640
ALPHA = 0.01
NOISE_FLOOR = 2.646e-4
GAMMAS = [0.0, 0.10, 0.26, 0.50, 0.70]


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
    print(f"  N={N:,} T={T} P_nnz={P.nnz:,} M_raw nnz={M_raw.nnz:,}")

    print("[raw] normalising and slicing to 4 full weeks...")
    rs = np.asarray(M_raw.sum(axis=1)).ravel()
    inv = np.where(rs > 0, 1.0 / rs, 0.0).astype(np.float32)
    M_norm = diags(inv) @ M_raw  # sparse
    n_full = 4 * N_BINS_WEEK
    M_norm_dense = M_norm[:n_full].toarray().astype(np.float32)  # 3.4 GB
    M_4w = M_norm_dense.reshape(4, N_BINS_WEEK, N)
    print(f"  M_4w shape={M_4w.shape}, mem={M_4w.nbytes/1e9:.2f} GB")

    # --- A. Per-cell raw statistics (4 weeks) ---
    print("[stats] raw_mean, raw_std per (bow, link)...")
    raw_mean = M_4w.mean(axis=0).astype(np.float32)            # (2016, N)
    raw_std  = M_4w.std(axis=0, ddof=1).astype(np.float32)     # (2016, N)
    # How many of 4 weeks were zero at each cell:
    zeros_in_4w = (M_4w == 0).sum(axis=0).astype(np.int8)      # (2016, N)
    # Bands
    EMPTY_M  = (raw_mean == 0)
    SUBN_M   = (~EMPTY_M) & (raw_std <  NOISE_FLOOR / 2)
    MIXED_M  = (~EMPTY_M) & (raw_std >= NOISE_FLOOR / 2) & (raw_std <= 2 * NOISE_FLOOR)
    SIGNAL_M = (~EMPTY_M) & (raw_std >  2 * NOISE_FLOOR)
    print(f"  EMPTY  cells: {int(EMPTY_M.sum()):>12,}")
    print(f"  SUBN   cells: {int(SUBN_M.sum()):>12,}")
    print(f"  MIXED  cells: {int(MIXED_M.sum()):>12,}")
    print(f"  SIGNAL cells: {int(SIGNAL_M.sum()):>12,}")
    print(f"  median raw_std (active) = "
          f"{float(np.median(raw_std[~EMPTY_M])):.3e}")

    # --- B. Per-bin zero fraction ---
    bin_zero_frac = (M_norm_dense == 0).mean(axis=1)  # (n_full,)
    print(f"[sparsity] per-bin zero fraction:")
    print(f"  min   = {float(bin_zero_frac.min()):.3f}")
    print(f"  median= {float(np.median(bin_zero_frac)):.3f}")
    print(f"  max   = {float(bin_zero_frac.max()):.3f}")
    print(f"  mean  = {float(bin_zero_frac.mean()):.3f}")

    # --- Distribution of "zeros_in_4w" ---
    print("[sparsity] zeros_in_4w distribution per cell:")
    for k in range(5):
        mask = (zeros_in_4w == k)
        print(f"  k={k}: {int(mask.sum()):>12,} cells "
              f"({100.0 * mask.sum() / mask.size:.2f}%)")

    # Save table A
    table_A = {
        "N_links": int(N),
        "T_bins": int(T_TOTAL),
        "n_full_4w_bins": int(n_full),
        "per_bin_zero_fraction": {
            "min": float(bin_zero_frac.min()),
            "median": float(np.median(bin_zero_frac)),
            "max": float(bin_zero_frac.max()),
            "mean": float(bin_zero_frac.mean()),
        },
        "zeros_in_4w_distribution_per_cell": {
            str(k): int((zeros_in_4w == k).sum()) for k in range(5)
        },
        "band_counts": {
            "EMPTY":  int(EMPTY_M.sum()),
            "SUBNOISE": int(SUBN_M.sum()),
            "MIXED":  int(MIXED_M.sum()),
            "SIGNAL": int(SIGNAL_M.sum()),
        },
        "median_raw_std_active": float(np.median(raw_std[~EMPTY_M])),
        "noise_floor_median": NOISE_FLOOR,
    }
    with open(OUT / "table_A_data_landscape.json", "w") as f:
        json.dump(table_A, f, indent=2)

    # Drop M_norm_dense (3.4 GB)
    del M_norm_dense, M_4w
    gc.collect()

    # --- C. Pre-compute Ma, PMa for diffusion sweep ---
    print(f"[precompute] Ma = M_raw + alpha (alpha={ALPHA})")
    Ma = M_raw.toarray() + np.float32(ALPHA)
    print(f"[precompute] PMa = P @ Ma.T")
    PMa = P @ Ma.T
    # Free M_raw (still have it via Ma)
    # M_raw is sparse so cheap to keep

    # Re-derive a SMALL dense copy of M_norm for displacement calc
    # We only need the 4-week slice, computed week-by-week to bound memory.
    # Build it once now (since we already freed it above):
    print("[build] M_norm_4w (4-week slice, dense)")
    M_norm_4w = M_norm[:n_full].toarray().astype(np.float32).reshape(
        4, N_BINS_WEEK, N
    )

    # --- D. Per-gamma sweep, stratified statistics ---
    per_gamma = {}
    for gamma in GAMMAS:
        t_s = time.time()
        print(f"\n[gamma={gamma}]")
        if gamma == 0.0:
            S_dense = Ma.copy()
        else:
            S_dense = (1.0 - gamma) * Ma + gamma * PMa.T
            S_dense = S_dense.astype(np.float32)
        # row-normalise
        rsm = S_dense.sum(axis=1, keepdims=True)
        rsm = np.where(rsm > 0, rsm, 1.0)
        S_dense /= rsm

        S_4w = S_dense[:n_full].reshape(4, N_BINS_WEEK, N)
        smooth_mean = S_4w.mean(axis=0).astype(np.float32)             # (2016, N)
        smooth_std  = S_4w.std(axis=0, ddof=1).astype(np.float32)      # (2016, N)
        # displacement = mean over weeks of |S - M_norm|
        # Compute week-by-week to keep memory bounded
        displacement = np.zeros_like(raw_mean)
        for w in range(4):
            displacement += np.abs(
                S_4w[w] - M_norm_4w[w]
            )
        displacement /= 4.0

        # Free S_dense, keep small stats
        del S_dense; gc.collect()

        # Per-band stats
        bands = [("EMPTY", EMPTY_M),
                 ("SUBNOISE", SUBN_M),
                 ("MIXED", MIXED_M),
                 ("SIGNAL", SIGNAL_M)]
        gstats = {}
        print(f"  band         n      med raw_mean med raw_std med smooth_mean "
              f"med smooth_std med disp  med disp/raw_std")
        for band_name, mask in bands:
            if not mask.any():
                continue
            n_in = int(mask.sum())
            rm = float(np.median(raw_mean[mask]))
            rs2 = float(np.median(raw_std[mask]))
            sm = float(np.median(smooth_mean[mask]))
            ss2 = float(np.median(smooth_std[mask]))
            d_ = float(np.median(displacement[mask]))
            # disp/raw_std (where raw_std > 0); for EMPTY we skip this ratio
            if band_name == "EMPTY":
                dr = float("nan")
            else:
                ratios = displacement[mask] / np.maximum(raw_std[mask], 1e-12)
                dr = float(np.median(ratios))
            gstats[band_name] = {
                "n": n_in,
                "med_raw_mean": rm,
                "med_raw_std": rs2,
                "med_smooth_mean": sm,
                "med_smooth_std": ss2,
                "med_displacement": d_,
                "med_disp_over_raw_std": dr,
            }
            print(f"  {band_name:8s} {n_in:>11,}  {rm:>11.3e}  "
                  f"{rs2:>11.3e}  {sm:>11.3e}  {ss2:>11.3e}  "
                  f"{d_:>9.3e}  {dr:>15.3f}")
        per_gamma[float(gamma)] = gstats
        print(f"  ({time.time()-t_s:.0f}s)")

        # ---- Per-gamma plot: smoothing impact stratified by raw_std ----
        # Bin raw_std on log scale, compute mean displacement in each bin
        # (active cells only).
        rs_active = raw_std[~EMPTY_M]
        disp_active = displacement[~EMPTY_M]
        bins_log = np.logspace(-7, -2, 60)
        bin_idx = np.digitize(rs_active, bins_log) - 1
        bin_medians = np.zeros(len(bins_log) - 1)
        for i in range(len(bin_medians)):
            sel = (bin_idx == i)
            if sel.any():
                bin_medians[i] = np.median(disp_active[sel])
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        centers = 0.5 * (bins_log[1:] + bins_log[:-1])
        ax.plot(centers, bin_medians, "o-", color="#1f6fb4", ms=4, lw=1.2,
                label=fr"median smoothing displacement, $\gamma={gamma}$")
        # noise-floor reference
        ax.axvline(NOISE_FLOOR, color="red", ls="--", lw=1.0,
                   label=f"noise floor = {NOISE_FLOOR:.2e}")
        ax.axvline(NOISE_FLOOR / 2, color="red", ls=":", lw=0.8)
        ax.axvline(NOISE_FLOOR * 2, color="red", ls=":", lw=0.8)
        ax.plot([1e-7, 1e-2], [1e-7, 1e-2], color="gray", ls="--", lw=0.6,
                label="y = x (perfect preservation: disp == raw_std)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("cell's raw cross-week std (per (link, bow))")
        ax.set_ylabel("median smoothing displacement")
        ax.set_title(rf"Smoothing impact stratified by raw cross-week std "
                     rf"($\gamma={gamma}$, $\alpha={ALPHA}$)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9)
        fig.tight_layout()
        out_fig = FIGDIR / f"fig_displacement_vs_rawstd_g{int(gamma*100):03d}.png"
        fig.savefig(out_fig, dpi=130)
        plt.close(fig)
        print(f"  saved {out_fig}")
        del displacement; gc.collect()

    # Save the per_gamma table
    with open(OUT / "table_B_per_band_per_gamma.json", "w") as f:
        json.dump(per_gamma, f, indent=2)

    # --- E. Combined band-displacement bar plot ---
    band_names = ["EMPTY", "SUBNOISE", "MIXED", "SIGNAL"]
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(band_names))
    width = 0.15
    for i, gamma in enumerate(GAMMAS):
        ys = []
        for bn in band_names:
            v = per_gamma[float(gamma)].get(bn, {}).get("med_displacement", 0.0)
            ys.append(v)
        ax.bar(xs + (i - 2) * width, ys, width, label=fr"$\gamma={gamma}$")
    ax.axhline(NOISE_FLOOR, color="red", ls="--", lw=1.0,
               label=f"noise floor = {NOISE_FLOOR:.2e}")
    ax.set_xticks(xs); ax.set_xticklabels(band_names)
    ax.set_yscale("log")
    ax.set_ylabel("median smoothing displacement (probability units)")
    ax.set_title("Median smoothing displacement per cell band, by $\\gamma$")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_band_displacement_bars.png", dpi=130)
    plt.close(fig)

    # --- F. Distribution of EMPTY-cell smoothed value (sparsity fix) ---
    # For the best/intermediate gamma=0.26, plot the distribution of
    # smoothed mean values on EMPTY cells.
    fig, axes = plt.subplots(1, len(GAMMAS), figsize=(4 * len(GAMMAS), 4),
                              sharey=True)
    if len(GAMMAS) == 1:
        axes = [axes]
    for ax, gamma in zip(axes, GAMMAS):
        # We need smooth_mean for empty cells at this gamma.
        # Re-smooth (small extra cost).
        if gamma == 0.0:
            S_dense = Ma.copy()
        else:
            S_dense = (1.0 - gamma) * Ma + gamma * PMa.T
            S_dense = S_dense.astype(np.float32)
        rsm = S_dense.sum(axis=1, keepdims=True)
        rsm = np.where(rsm > 0, rsm, 1.0)
        S_dense /= rsm
        S_4w = S_dense[:n_full].reshape(4, N_BINS_WEEK, N)
        sm = S_4w.mean(axis=0).astype(np.float32)
        del S_dense; gc.collect()
        vals = sm[EMPTY_M]
        ax.hist(np.clip(vals, 1e-10, 1e-2),
                bins=np.logspace(-10, -2, 80),
                color="#2e8b57", alpha=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(fr"$\gamma={gamma}$,  median = {float(np.median(vals)):.2e}",
                     fontsize=10)
        ax.set_xlabel("smoothed value on EMPTY cell")
        ax.grid(True, alpha=0.3, which="both")
        del sm
    axes[0].set_ylabel("count")
    fig.suptitle(f"How much mass diffusion deposits on always-empty cells "
                 f"(N = {int(EMPTY_M.sum()):,})")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_empty_cell_fill.png", dpi=130)
    plt.close(fig)

    print(f"\n[done] {time.time()-t0:.0f}s")
    print(f"results in: {OUT}")


if __name__ == "__main__":
    main()
