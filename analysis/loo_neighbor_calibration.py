"""Leave-one-out (LOO) calibration of gamma on non-zero cells.

Idea (from H.A.): take observed non-zero cells as ground truth. Hide
each one in turn and ask the smoother to predict it from its
outgoing neighbors using

    T_gamma = (1 - gamma) I + gamma P,    P_{ii} = 0.

In raw count units, the held-out prediction at i is

    C_hat_{b,i} = gamma (P C_b)_i + alpha,

and the truth is C_{b,i} + alpha, so the residual is

    r_{b,i} = gamma (P C_b)_i - C_{b,i}.

The alpha contributions cancel (alpha regularizes empty cells; on
observed cells it is silent). The least-squares optimum in closed
form is

    gamma_star = sum (PC_b)_i C_{b,i} / sum (PC_b)_i^2.

We compute this:
  (a) on 1000 random non-zero (bin, link) cells (the user's idea),
  (b) on ALL non-zero cells in the corpus (full sample, for vibes),
  (c) sweep gamma in [0, 1] and report the loss curve for plotting.

We also write a per-link gamma_star_i = sum_b (PC_b)_i C_{b,i} / sum_b (PC_b)_i^2
to show how the optimal gamma varies across the road network.
"""
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

OUT = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project/results/loo_calibration")
OUT.mkdir(parents=True, exist_ok=True)
FIGDIR = OUT / "figs"
FIGDIR.mkdir(exist_ok=True)

N_SAMPLES = 1000
RNG_SEED = 42


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


def main():
    t0 = time.time()
    P, M_raw, T, N = load()
    print(f"  N = {N:,}, T = {T}")
    n_nonzero = M_raw.nnz
    print(f"  total non-zero cells in corpus = {n_nonzero:,}")

    rng = np.random.default_rng(RNG_SEED)

    # -------------------------------------------------------------------
    # (a) 1000 random non-zero (bin, link) cells
    # -------------------------------------------------------------------
    coo = M_raw.tocoo()
    idx = rng.choice(coo.nnz, size=N_SAMPLES, replace=False)
    samp_bins   = coo.row[idx]
    samp_links  = coo.col[idx]
    samp_counts = coo.data[idx].astype(np.float64)
    print(f"\n[(a) sample] {N_SAMPLES} random non-zero cells. "
          f"Distinct bins covered: {len(np.unique(samp_bins))}.")
    print(f"  count C_i  range = [{samp_counts.min():.0f}, "
          f"{samp_counts.max():.0f}], median = {np.median(samp_counts):.1f}")

    # Group sampled cells by bin (we need to compute P C_b once per bin)
    pc_at_i = np.zeros(N_SAMPLES, dtype=np.float64)
    unique_bins = np.unique(samp_bins)
    bin_to_k = {int(b): [] for b in unique_bins}
    for k, b in enumerate(samp_bins):
        bin_to_k[int(b)].append(k)
    print(f"  computing (P C_b)_i over {len(unique_bins)} bins...")
    for b in unique_bins:
        C_b = M_raw.getrow(int(b)).toarray().ravel().astype(np.float64)
        PCb = P @ C_b
        for k in bin_to_k[int(b)]:
            pc_at_i[k] = PCb[samp_links[k]]
    print(f"  (PC_b)_i: range [{pc_at_i.min():.4f}, {pc_at_i.max():.4f}], "
          f"median {np.median(pc_at_i):.4f}, "
          f"fraction at 0 = {float((pc_at_i == 0).mean()):.3f}")

    # Closed form gamma*
    numer = float(np.sum(pc_at_i * samp_counts))
    denom = float(np.sum(pc_at_i ** 2))
    gamma_star_unclipped = numer / denom if denom > 0 else float("inf")
    gamma_star_clipped = max(0.0, min(1.0, gamma_star_unclipped))
    print(f"\n  numer = sum (PC)_i C_i      = {numer:.4e}")
    print(f"  denom = sum (PC)_i^2        = {denom:.4e}")
    print(f"  gamma_star (unclipped)      = {gamma_star_unclipped:.4f}")
    print(f"  gamma_star (in [0,1])       = {gamma_star_clipped:.4f}")

    # Loss curve over the 1000 sample
    gammas = np.linspace(0.0, 1.5, 61)
    losses_1k = np.array(
        [float(np.sum((g * pc_at_i - samp_counts) ** 2)) for g in gammas]
    )

    # -------------------------------------------------------------------
    # (b) Full-corpus gamma* on ALL non-zero cells
    # -------------------------------------------------------------------
    print(f"\n[(b) full corpus] computing gamma* on all {n_nonzero:,} "
          f"non-zero cells (loops over {T} bins)...")
    full_numer = 0.0
    full_denom = 0.0
    # Per-link accumulators for plot (c)
    per_link_numer = np.zeros(N, dtype=np.float64)
    per_link_denom = np.zeros(N, dtype=np.float64)
    per_link_count = np.zeros(N, dtype=np.int64)
    t_s = time.time()
    for b in range(T):
        C_b = M_raw.getrow(b).toarray().ravel().astype(np.float64)
        if C_b.sum() == 0:
            continue
        nz = np.nonzero(C_b)[0]
        PCb = P @ C_b
        # Restrict to cells where (PCb)_i > 0 (otherwise contribution = 0)
        nz_pc = PCb[nz]
        nz_c  = C_b[nz]
        full_numer += float(np.sum(nz_pc * nz_c))
        full_denom += float(np.sum(nz_pc ** 2))
        # Per-link accumulators (only over non-zero cells at this bin)
        per_link_numer[nz] += nz_pc * nz_c
        per_link_denom[nz] += nz_pc ** 2
        per_link_count[nz] += 1
        if (b + 1) % 1000 == 0:
            print(f"  bin {b+1}/{T}  partial gamma* = "
                  f"{full_numer/full_denom:.4f}  "
                  f"({time.time()-t_s:.0f}s)", flush=True)
    gamma_full_unclipped = full_numer / full_denom if full_denom > 0 else float("inf")
    gamma_full_clipped = max(0.0, min(1.0, gamma_full_unclipped))
    print(f"\n  numer (full)  = {full_numer:.4e}")
    print(f"  denom (full)  = {full_denom:.4e}")
    print(f"  gamma_full (unclipped) = {gamma_full_unclipped:.4f}")
    print(f"  gamma_full (in [0,1])  = {gamma_full_clipped:.4f}")

    # Per-link gamma*_i (only on links with at least 10 active bins and
    # positive denominator)
    mask = (per_link_count >= 10) & (per_link_denom > 0)
    gamma_per_link = np.full(N, np.nan, dtype=np.float64)
    gamma_per_link[mask] = per_link_numer[mask] / per_link_denom[mask]
    gpl = gamma_per_link[mask]
    print(f"\n  per-link gamma*_i (links with >= 10 active bins, "
          f"n = {int(mask.sum()):,}):")
    print(f"    median   = {float(np.median(gpl)):.3f}")
    print(f"    mean     = {float(np.mean(gpl)):.3f}")
    print(f"    p05      = {float(np.percentile(gpl, 5)):.3f}")
    print(f"    p95      = {float(np.percentile(gpl, 95)):.3f}")
    print(f"    frac > 1 = {float((gpl > 1.0).mean()):.3f}")

    # -------------------------------------------------------------------
    # Plot: loss curve (1k sample) + key gamma points + per-link histogram
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(gammas, losses_1k / losses_1k.max(), "o-",
            color="#1f6fb4", lw=1.6, ms=4,
            label="LOO loss (normalised)")
    ax.axvline(gamma_star_clipped, color="red", ls="--", lw=1.4,
               label=fr"$\gamma^\star$ (1000-cell sample) = {gamma_star_clipped:.3f}"
                     + (f" (unclipped {gamma_star_unclipped:.3f})"
                        if gamma_star_unclipped > 1.0 else ""))
    ax.axvline(gamma_full_clipped, color="orange", ls="--", lw=1.4,
               label=fr"$\gamma^\star$ (full corpus) = {gamma_full_clipped:.3f}"
                     + (f" (unclipped {gamma_full_unclipped:.3f})"
                        if gamma_full_unclipped > 1.0 else ""))
    ax.axvline(0.26, color="gray", ls=":", lw=1.2,
               label=r"paper's previous $\gamma = 0.26$")
    ax.axvline(1.0, color="black", ls="-", lw=0.5,
               label=r"$\gamma=1$ boundary")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$\sum_i (\gamma\,(P C_b)_i - C_{b,i})^2$ (normalised)")
    ax.set_title(r"LOO loss on 1000 random non-zero cells")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    # Per-link histogram, clipped at gamma in [0, 3] for readability
    gpl_clip = np.clip(gpl, 0.0, 3.0)
    ax.hist(gpl_clip, bins=np.linspace(0, 3.0, 61),
            color="#2e8b57", alpha=0.85, edgecolor="white")
    ax.axvline(1.0, color="black", ls="-", lw=0.6)
    ax.axvline(float(np.median(gpl)), color="red", ls="--", lw=1.4,
               label=fr"median per-link $\gamma_i^\star = {float(np.median(gpl)):.3f}$")
    ax.axvline(gamma_full_unclipped, color="orange", ls="--", lw=1.4,
               label=fr"corpus $\gamma^\star = {gamma_full_unclipped:.3f}$")
    ax.axvline(0.26, color="gray", ls=":", lw=1.2,
               label=r"paper's $\gamma=0.26$")
    ax.set_xlabel(r"per-link $\gamma_i^\star$ (clipped at 3 for display)")
    ax.set_ylabel("number of links")
    ax.set_title(rf"Per-link LOO-optimal $\gamma_i^\star$ "
                 rf"(n={int(mask.sum()):,} links with $\geq$10 active bins)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_loo_calibration.png", dpi=130)
    plt.close(fig)

    with open(OUT / "loo_results.json", "w") as f:
        json.dump({
            "method": ("LOO on non-zero cells: gamma minimises "
                       "sum_{i in S} (gamma (P C_b)_i - C_{b,i})^2; "
                       "alpha cancels in the residual."),
            "N_samples": N_SAMPLES,
            "rng_seed": RNG_SEED,
            "total_nonzero_cells_corpus": int(n_nonzero),
            "sample_1000_cells": {
                "gamma_star_unclipped": float(gamma_star_unclipped),
                "gamma_star_clipped":   float(gamma_star_clipped),
                "numer":                float(numer),
                "denom":                float(denom),
            },
            "full_corpus": {
                "gamma_star_unclipped": float(gamma_full_unclipped),
                "gamma_star_clipped":   float(gamma_full_clipped),
                "numer":                float(full_numer),
                "denom":                float(full_denom),
                "n_nonzero":            int(n_nonzero),
            },
            "per_link": {
                "n_links_with_ge10_active_bins": int(mask.sum()),
                "median": float(np.median(gpl)),
                "mean":   float(np.mean(gpl)),
                "p05":    float(np.percentile(gpl, 5)),
                "p25":    float(np.percentile(gpl, 25)),
                "p75":    float(np.percentile(gpl, 75)),
                "p95":    float(np.percentile(gpl, 95)),
                "frac_gt_1": float((gpl > 1.0).mean()),
            },
            "note_on_alpha": (
                "alpha contributions cancel exactly in the LOO residual "
                "on observed cells (because P_{ii}=0 and observed C >> alpha); "
                "alpha is calibrated separately, by Appendix B's variance "
                "criterion on empty cells."
            ),
        }, f, indent=2)
    print(f"\nSaved {OUT / 'loo_results.json'}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
