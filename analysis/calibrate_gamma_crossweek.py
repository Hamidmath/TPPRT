"""Comprehensive gamma (and alpha) calibration via cross-week analysis.

Memory-efficient version: precomputes Ma = M + alpha and PMa = P @ Ma once
per (alpha), then S(gamma) = (1-gamma)*Ma + gamma*PMa.

Implements the professor's recipe:
  1. For each (link, bin-of-week) compute the std across weeks.
  2. Take the median across all (link, bin-of-week) to get the noise floor.
  3. Set diffusion so the smoothing scale roughly matches that level.

Then validates against many cross-week prediction comparisons over a wide
gamma sweep:
  - 1-week to next-week
  - 2-week average to next-week
  - 3-week average to next-week
  - day-of-week to same-day next-week (per weekday)
  - weekday (Mon-Fri) bin to same weekday slot next week
  - weekend (Sat/Sun) bin to same weekend slot next week
"""
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz

OUT = config.RESULTS_DIR / "gamma_calibration"
OUT.mkdir(parents=True, exist_ok=True)
FIGDIR = OUT / "figs"
FIGDIR.mkdir(exist_ok=True)

BIN_MIN = 5
N_BINS_DAY = 24 * 60 // BIN_MIN      # 288
N_BINS_WEEK = 7 * N_BINS_DAY          # 2016
T_TOTAL = 8640                        # 30 days
DOW_OF_BIN0 = 4  # corpus starts Fri 2018-08-31 19:00

# Sweep settings (kept smaller than initial draft to control runtime)
GAMMAS = np.round(np.concatenate([
    np.linspace(0.00, 0.50, 21),
    np.array([0.55, 0.60, 0.70, 0.80, 0.90]),
]), 3)
ALPHAS = [0.0, 0.001, 0.01, 0.1, 1.0]
TOP_K = 100
EPS = 1e-6


def load_data():
    print("[load] graph...")
    with open(config.GRAPH_FILE) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    adj = graph.get("adjacency", {})

    print("[load] building row-stochastic P...")
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

    print("[load] raw popularity (as sparse CSR)...")
    t0 = time.time()
    b = load_popularity_npz(str(config.DATA_DIR / "popularity_results_osm.npz"))
    matrix = b["matrix"]
    pop_link_ids = b["link_ids"]
    times = b["times"]
    proj = np.array(
        [lid_to_idx.get(lid, -1) for lid in pop_link_ids], dtype=np.int64
    )
    valid_mask = proj >= 0

    # Build M_raw as float32 sparse, shape (T, N), aligned to graph link order
    T = len(times)
    csr_data = matrix.tocsr() if not isinstance(matrix, csr_matrix) else matrix
    # Re-project columns
    src_cols = np.arange(len(pop_link_ids))
    # Build a column permutation: for each src column j (valid), new col proj[j]
    valid_src = src_cols[valid_mask]
    new_cols = proj[valid_mask]
    # Build (T, N) by remapping
    # Easiest: for each row t, get columns from CSR and remap
    # But faster: use scipy. We'll convert csr by remap.
    # Use COO then construct
    coo = matrix.tocoo()
    sel = valid_mask[coo.col]
    rows_sel = coo.row[sel]
    cols_sel = proj[coo.col[sel]]
    data_sel = coo.data[sel].astype(np.float32)
    M_raw = csr_matrix((data_sel, (rows_sel, cols_sel)),
                       shape=(T, N), dtype=np.float32)
    print(f"  M_raw: shape={M_raw.shape}, nnz={M_raw.nnz:,}  "
          f"({time.time()-t0:.1f}s)")

    # Pre-normalize per row: M_norm = diag(1/row_sum) @ M
    row_sums = np.asarray(M_raw.sum(axis=1)).ravel()
    inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0).astype(np.float32)
    from scipy.sparse import diags
    M_norm = diags(inv) @ M_raw  # CSR @ diag stays sparse-ish; this is dense diags
    M_norm = M_norm.tocsr()

    return graph, P, M_raw, M_norm, T, N, links


def cross_week_noise_floor(M_norm, T, N):
    """Compute cross-week std for each (link, bin-of-week) using sparse rows.

    Returns (stds_2d: (n_per_week, N) std-per-pair, active_mask: bool).
    """
    n_weeks = 4
    n_per_week = N_BINS_WEEK
    full = n_weeks * n_per_week  # 8064
    full = min(full, T)
    # Build a dense (n_weeks, N) per-bow buffer
    print(f"[noise] computing cross-week std per (link, bow) "
          f"on {n_weeks} weeks x {n_per_week} bins x {N} links ...")
    t0 = time.time()
    stds = np.zeros((n_per_week, N), dtype=np.float32)
    active = np.zeros((n_per_week, N), dtype=bool)
    # We process bow-by-bow to keep memory bounded
    # Each bow has up to n_weeks rows of M_norm
    M_lil = M_norm.tolil()
    for bow in range(n_per_week):
        rows = []
        any_pos = np.zeros(N, dtype=bool)
        for w in range(n_weeks):
            t = w * n_per_week + bow
            if t >= T:
                break
            rrow = M_norm.getrow(t).toarray().ravel()
            rows.append(rrow)
            any_pos |= rrow > 0
        if not rows:
            continue
        rows_arr = np.stack(rows, axis=0)  # (n_weeks, N)
        stds[bow] = rows_arr.std(axis=0, ddof=1) if len(rows) > 1 else 0.0
        active[bow] = any_pos
        if bow % 200 == 0:
            print(f"  bow {bow}/{n_per_week}  ({time.time()-t0:.0f}s)")
    print(f"[noise] done in {time.time()-t0:.0f}s")
    return stds, active


def precompute_PMa(M_raw, P, alpha):
    """Return (Ma_dense, PMa_dense) of shape (T, N) and (N, T).

    Ma_dense = M_raw + alpha (dense float32).
    PMa_dense = P @ Ma_dense.T (sparse-dense matmul, dense float32 (N, T)).

    Together: ~6.8 GB at fp32 for our dimensions. Done once per alpha.
    """
    print(f"[precompute] Ma_dense = M_raw + {alpha:.4g} ...")
    t0 = time.time()
    # Build Ma as dense float32
    Ma = M_raw.toarray() + np.float32(alpha)  # (T, N) float32
    print(f"  Ma: shape={Ma.shape}, dtype={Ma.dtype}, mem~{Ma.nbytes/1e9:.2f} GB "
          f"({time.time()-t0:.0f}s)")
    print(f"[precompute] PMa = P @ Ma.T ...")
    t0 = time.time()
    PMa = P @ Ma.T  # (N, T) dense float32
    print(f"  PMa: shape={PMa.shape}, mem~{PMa.nbytes/1e9:.2f} GB "
          f"({time.time()-t0:.0f}s)")
    return Ma, PMa


def smooth_from_precomputed(Ma, PMa, gamma):
    """S = (1-gamma)*Ma + gamma*PMa.T, then row-normalize. Returns (T, N) fp32."""
    if gamma == 0.0:
        S = Ma.copy()
    else:
        S = (1.0 - gamma) * Ma + gamma * PMa.T
        S = S.astype(np.float32)
    row_sums = S.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    S /= row_sums
    return S


def mre_pair(truth, pred, eps=EPS, top_k=TOP_K):
    rel = np.abs(truth - pred) / (truth + eps)
    overall = float(rel.mean())
    top = np.argpartition(-truth, top_k)[:top_k]
    return overall, float(rel[top].mean())


def comparison_metrics(S, n_weeks_full=4, seed=42):
    """Run all the cross-week comparisons on a smoothed matrix S of shape (T, N).

    Returns dict cmp_name -> {n, top_mean, overall_mean, top_median, overall_median}.
    """
    N = S.shape[1]
    n_per_week = N_BINS_WEEK
    M_3d = S[:n_weeks_full * n_per_week].reshape(
        n_weeks_full, n_per_week, N
    )
    avgs = {2: M_3d[:2].mean(axis=0), 3: M_3d[:3].mean(axis=0)}

    rng = np.random.default_rng(seed)
    out = defaultdict(list)

    # 1-week to next-week (sample 1000 pairs)
    pairs_1w = [(w, b) for w in range(n_weeks_full - 1)
                       for b in range(n_per_week)]
    idx = rng.choice(len(pairs_1w), size=min(1000, len(pairs_1w)), replace=False)
    for j in idx:
        w, bow = pairs_1w[j]
        ov, top = mre_pair(M_3d[w + 1, bow], M_3d[w, bow])
        out["1w_next"].append((top, ov))

    # 2-week average to next-week (week 2 is target)
    bows = rng.choice(n_per_week, size=min(500, n_per_week), replace=False)
    for bow in bows:
        ov, top = mre_pair(M_3d[2, bow], avgs[2][bow])
        out["2w_avg_to_next"].append((top, ov))

    # 3-week average to next-week (week 3 is target)
    bows = rng.choice(n_per_week, size=min(500, n_per_week), replace=False)
    for bow in bows:
        ov, top = mre_pair(M_3d[3, bow], avgs[3][bow])
        out["3w_avg_to_next"].append((top, ov))

    # Day-of-week stratification: for each weekday, compute 1w-next for
    # bins that fall on that weekday.
    for w in range(n_weeks_full - 1):
        bow_sample = rng.choice(n_per_week, size=min(700, n_per_week), replace=False)
        for bow in bow_sample:
            dow_within = (bow // N_BINS_DAY)
            real_dow = (DOW_OF_BIN0 + dow_within) % 7
            ov, top = mre_pair(M_3d[w + 1, bow], M_3d[w, bow])
            label = "weekday_1w" if real_dow < 5 else "weekend_1w"
            out[label].append((top, ov))
            out[f"dow{real_dow}_1w"].append((top, ov))

    result = {}
    for k, lst in out.items():
        arr = np.asarray(lst)
        result[k] = {
            "n": len(lst),
            "top_mean": float(arr[:, 0].mean()),
            "top_median": float(np.median(arr[:, 0])),
            "overall_mean": float(arr[:, 1].mean()),
            "overall_median": float(np.median(arr[:, 1])),
        }
    return result


def smoothing_displacement_metrics(S, M_norm, n_weeks_full=4):
    """Median and mean of |S - M_norm| over (link, bow) for active pairs only,
    computed per-bow to avoid materializing a (T, N) delta."""
    n_per_week = N_BINS_WEEK
    full = n_weeks_full * n_per_week
    # Build active mask: any row in 4 weeks has positive value (from sparse M_norm)
    deltas = []
    active_count = 0
    for bow in range(n_per_week):
        # Get four week values for this bow as dense (n_weeks_full, N)
        rows_norm = np.zeros((n_weeks_full, S.shape[1]), dtype=np.float32)
        rows_S = np.zeros((n_weeks_full, S.shape[1]), dtype=np.float32)
        for w in range(n_weeks_full):
            t = w * n_per_week + bow
            rows_norm[w] = M_norm.getrow(t).toarray().ravel()
            rows_S[w] = S[t]
        any_pos = (rows_norm > 0).any(axis=0)
        if not any_pos.any():
            continue
        # mean over weeks of |delta|
        delta_per_week = np.abs(rows_S - rows_norm)
        delta_mean_over_weeks = delta_per_week.mean(axis=0)
        deltas.extend(delta_mean_over_weeks[any_pos].tolist())
        active_count += int(any_pos.sum())
    deltas = np.asarray(deltas)
    return {
        "delta_median": float(np.median(deltas)),
        "delta_mean": float(deltas.mean()),
        "n_active": active_count,
    }


def main():
    t_global = time.time()
    graph, P, M_raw, M_norm, T, N, links = load_data()

    print()
    print("=" * 78)
    print("STEP 1: Cross-week noise floor (raw normalized probability matrix)")
    print("=" * 78)
    stds, active = cross_week_noise_floor(M_norm, T, N)
    stds_active = stds[active]
    noise_floor = {
        "median": float(np.median(stds_active)),
        "mean":   float(stds_active.mean()),
        "p10":    float(np.percentile(stds_active, 10)),
        "p25":    float(np.percentile(stds_active, 25)),
        "p75":    float(np.percentile(stds_active, 75)),
        "p90":    float(np.percentile(stds_active, 90)),
        "p95":    float(np.percentile(stds_active, 95)),
        "p99":    float(np.percentile(stds_active, 99)),
        "n_active": int(active.sum()),
        "weeks_used": 4,
    }
    print(f"  N active (bow, link) pairs : {noise_floor['n_active']:,}")
    print(f"  cross-week std median      : {noise_floor['median']:.3e}")
    print(f"  cross-week std mean        : {noise_floor['mean']:.3e}")
    print(f"  cross-week std 90th pct    : {noise_floor['p90']:.3e}")
    print(f"  cross-week std 95th pct    : {noise_floor['p95']:.3e}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.clip(stds_active, 1e-9, 1e-2),
            bins=np.logspace(-9, -2, 80), color="#1f6fb4", alpha=0.85)
    ax.axvline(noise_floor['median'], color="red", ls="--", lw=1.4,
               label=f"median = {noise_floor['median']:.2e}")
    ax.axvline(noise_floor['mean'], color="orange", ls=":", lw=1.4,
               label=f"mean = {noise_floor['mean']:.2e}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("cross-week std (probability units, per (link, bin-of-week))")
    ax.set_ylabel("count")
    ax.set_title(f"Cross-week noise floor (4 weeks, "
                 f"{noise_floor['n_active']:,} active pairs)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_cross_week_noise_floor.png", dpi=130)
    plt.close(fig)
    print(f"  saved {FIGDIR / 'fig1_cross_week_noise_floor.png'}")

    with open(OUT / "cross_week_noise_floor.json", "w") as f:
        json.dump(noise_floor, f, indent=2)

    # Free stds + active
    del stds, active, stds_active
    gc.collect()

    print()
    print("=" * 78)
    print(f"STEP 2: gamma sweep ({len(GAMMAS)} values, alpha=0.01 fixed)")
    print("=" * 78)
    alpha_fixed = 0.01
    Ma, PMa = precompute_PMa(M_raw, P, alpha_fixed)

    # Pre-compute baseline raw variance per row (for variance-retained plot)
    var_raw_per_row = []
    for t in range(0, T, 10):
        r = M_norm.getrow(t).toarray().ravel()
        if r.var() > 0:
            var_raw_per_row.append(r.var())
    var_raw_mean = float(np.mean(var_raw_per_row))

    sweep = {}
    smoothing_change = []
    var_retained_list = []
    for gi, gamma in enumerate(GAMMAS):
        t0 = time.time()
        S = smooth_from_precomputed(Ma, PMa, float(gamma))

        # Smoothing displacement (against raw normalized probability matrix)
        disp = smoothing_displacement_metrics(S, M_norm, 4)
        smoothing_change.append((float(gamma),
                                 disp["delta_median"], disp["delta_mean"]))

        # Variance retained (sample every 10th row)
        vs = []
        for t in range(0, T, 10):
            r = S[t]
            if r.var() > 0:
                vs.append(r.var())
        var_ret = float(np.mean(vs)) / var_raw_mean if vs else 0.0
        var_retained_list.append((float(gamma), var_ret))

        # Cross-week prediction metrics
        metrics = comparison_metrics(S, 4, seed=42)
        sweep[float(gamma)] = {
            "smoothing_delta_median": disp["delta_median"],
            "smoothing_delta_mean":   disp["delta_mean"],
            "var_retained":           var_ret,
            **metrics,
        }
        print(f"  gamma = {gamma:.3f}  delta_med = {disp['delta_median']:.3e}  "
              f"var_ret = {var_ret:.3f}  "
              f"1w T-100 = {metrics['1w_next']['top_mean']:.4f}  "
              f"3w T-100 = {metrics['3w_avg_to_next']['top_mean']:.4f}  "
              f"({time.time()-t0:.1f}s)")
        del S
        gc.collect()

    with open(OUT / "gamma_sweep_results.json", "w") as f:
        json.dump({"alpha": alpha_fixed,
                   "noise_floor": noise_floor,
                   "sweep": sweep}, f, indent=2)

    # ----- Best gammas -----
    comparisons = ["1w_next", "2w_avg_to_next", "3w_avg_to_next",
                   "weekday_1w", "weekend_1w"]
    best_gammas = {}
    print()
    print("Best gamma per comparison (Top-100 MRE):")
    for cmp in comparisons:
        gs = [(g, sweep[g][cmp]["top_mean"]) for g in sorted(sweep)
              if cmp in sweep[g]]
        g_best = min(gs, key=lambda x: x[1])
        best_gammas[cmp] = g_best
        print(f"  {cmp:20s}: gamma = {g_best[0]:.3f}, T-100 = {g_best[1]:.4f}")

    # ----- Figure 2: gamma vs Top-100 MRE -----
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {"1w_next": "#1f6fb4", "2w_avg_to_next": "#2e8b57",
               "3w_avg_to_next": "#7e57c2", "weekday_1w": "#c0392b",
               "weekend_1w": "#f39c12"}
    for cmp in comparisons:
        ys = [sweep[g][cmp]["top_mean"] for g in GAMMAS]
        ax.plot(GAMMAS, ys, "o-", color=palette[cmp], lw=1.4, ms=4, label=cmp)
        gbest, ybest = best_gammas[cmp]
        ax.scatter([gbest], [ybest], color=palette[cmp], s=140, marker="*",
                   edgecolor="black", zorder=5)
    ax.set_xlabel(r"diffusion strength $\gamma$")
    ax.set_ylabel("Top-100 MRE (mean across pairs)")
    ax.set_title(r"Cross-week prediction Top-100 MRE vs $\gamma$ ($\alpha=0.01$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_gamma_top100.png", dpi=130)
    plt.close(fig)
    print(f"  saved {FIGDIR / 'fig2_gamma_top100.png'}")

    # ----- Figure 3: gamma vs Overall MRE -----
    fig, ax = plt.subplots(figsize=(9, 5))
    for cmp in comparisons:
        ys = [sweep[g][cmp]["overall_mean"] for g in GAMMAS]
        ax.plot(GAMMAS, ys, "o-", color=palette[cmp], lw=1.4, ms=4, label=cmp)
    ax.set_xlabel(r"diffusion strength $\gamma$")
    ax.set_ylabel("Overall MRE")
    ax.set_yscale("log")
    ax.set_title(r"Cross-week prediction Overall MRE vs $\gamma$ ($\alpha=0.01$)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3_gamma_overall.png", dpi=130)
    plt.close(fig)

    # ----- Figure 4: per-day-of-week -----
    fig, ax = plt.subplots(figsize=(9, 5))
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for dow in range(7):
        cmp = f"dow{dow}_1w"
        if cmp not in sweep[GAMMAS[0]]:
            continue
        ys = [sweep[g][cmp]["top_mean"] for g in GAMMAS]
        ax.plot(GAMMAS, ys, "-", lw=1.4, label=dow_names[dow])
    ax.set_xlabel(r"diffusion strength $\gamma$")
    ax.set_ylabel("Top-100 MRE")
    ax.set_title(r"Same-day-next-week Top-100 vs $\gamma$, by day of week")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_gamma_by_dow.png", dpi=130)
    plt.close(fig)

    # ----- Figure 5: smoothing displacement vs noise floor -----
    gs = [x[0] for x in smoothing_change]
    deltas = np.asarray([x[1] for x in smoothing_change])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gs, deltas, "o-", color="#1f6fb4", lw=1.5, ms=4,
            label="median |M - smooth(M)|")
    ax.axhline(noise_floor['median'], color="red", ls="--", lw=1.4,
               label=f"cross-week noise floor median = {noise_floor['median']:.2e}")
    ax.axhline(noise_floor['mean'], color="orange", ls=":", lw=1.0,
               label=f"cross-week noise floor mean   = {noise_floor['mean']:.2e}")
    cross_idx = int(np.argmin(np.abs(deltas - noise_floor['median'])))
    g_match = gs[cross_idx]
    ax.scatter([g_match], [deltas[cross_idx]], color="red", s=150,
               marker="*", edgecolor="black", zorder=5,
               label=f"smoothing = noise at gamma = {g_match:.3f}")
    ax.set_xlabel(r"diffusion strength $\gamma$")
    ax.set_ylabel("smoothing displacement (median over active (bow, link))")
    ax.set_yscale("log")
    ax.set_title(r"Smoothing magnitude vs $\gamma$, against cross-week noise floor")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig5_smoothing_vs_noise_floor.png", dpi=130)
    plt.close(fig)
    print(f"  saved {FIGDIR / 'fig5_smoothing_vs_noise_floor.png'}")
    print(f"  smoothing matches noise floor at gamma = {g_match:.3f}")

    # ----- Figure 6: variance retained -----
    fig, ax = plt.subplots(figsize=(8, 5))
    gs = [x[0] for x in var_retained_list]
    vs = [x[1] for x in var_retained_list]
    ax.plot(gs, vs, "o-", color="#7e57c2", lw=1.4, ms=4)
    ax.axhline(1.0, color="gray", ls=":", lw=0.7)
    ax.set_xlabel(r"diffusion strength $\gamma$")
    ax.set_ylabel(r"variance retained: $\mathrm{mean\,Var}(c_\mathrm{smooth}) / \mathrm{mean\,Var}(c)$")
    ax.set_title(r"Variance retained vs $\gamma$ (the old plot, for reference)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_variance_retained.png", dpi=130)
    plt.close(fig)

    # Free Ma, PMa before alpha sweep — each alpha rebuilds them anyway
    del Ma, PMa
    gc.collect()

    print()
    print("=" * 78)
    print("STEP 3: alpha sweep at best gamma (from 1w_next)")
    print("=" * 78)
    g_best_1w = best_gammas["1w_next"][0]
    print(f"  using gamma = {g_best_1w}")
    alpha_sweep = {}
    for alpha in ALPHAS:
        t0 = time.time()
        Ma_a, PMa_a = precompute_PMa(M_raw, P, float(alpha))
        S = smooth_from_precomputed(Ma_a, PMa_a, float(g_best_1w))
        metrics = comparison_metrics(S, 4, seed=42)
        disp = smoothing_displacement_metrics(S, M_norm, 4)
        alpha_sweep[float(alpha)] = {**metrics,
                                      "smoothing_delta_median": disp["delta_median"]}
        print(f"  alpha = {alpha:.4f}  1w T-100 = {metrics['1w_next']['top_mean']:.4f}  "
              f"3w T-100 = {metrics['3w_avg_to_next']['top_mean']:.4f}  "
              f"({time.time()-t0:.1f}s)")
        del Ma_a, PMa_a, S; gc.collect()

    with open(OUT / "alpha_sweep_results.json", "w") as f:
        json.dump({"gamma_fixed": float(g_best_1w),
                   "sweep": alpha_sweep}, f, indent=2)

    # ----- Figure 7: alpha sensitivity -----
    fig, ax = plt.subplots(figsize=(8, 5))
    for cmp in comparisons:
        ys = [alpha_sweep[a][cmp]["top_mean"] for a in ALPHAS]
        ax.plot(ALPHAS, ys, "o-", lw=1.4, ms=5, color=palette[cmp], label=cmp)
    # log-x; alpha=0 needs replacing
    xs_plot = [a if a > 0 else 1e-5 for a in ALPHAS]
    ax.set_xscale("log")
    ax.set_xlabel(r"Laplace pseudo-count $\alpha$  (1e-5 represents $\alpha=0$)")
    ax.set_ylabel("Top-100 MRE")
    ax.set_title(f"Alpha sensitivity at $\\gamma$ = {g_best_1w:.3f}")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig7_alpha_sensitivity.png", dpi=130)
    plt.close(fig)

    print()
    print("=" * 78)
    print(f"DONE in {time.time()-t_global:.0f}s")
    print("=" * 78)
    print(f"\nFINAL RESULTS:")
    print(f"  Cross-week noise floor (median std): {noise_floor['median']:.3e}")
    print(f"  Cross-week noise floor (mean std)  : {noise_floor['mean']:.3e}")
    print(f"  gamma where smoothing matches noise: {g_match:.3f}")
    print(f"  best gamma by 1-week pred T-100    : {best_gammas['1w_next'][0]:.3f}"
          f"  ({best_gammas['1w_next'][1]:.4f})")
    print(f"  best gamma by 2-week-avg T-100     : {best_gammas['2w_avg_to_next'][0]:.3f}"
          f"  ({best_gammas['2w_avg_to_next'][1]:.4f})")
    print(f"  best gamma by 3-week-avg T-100     : {best_gammas['3w_avg_to_next'][0]:.3f}"
          f"  ({best_gammas['3w_avg_to_next'][1]:.4f})")
    print(f"  best gamma by weekday Top-100      : {best_gammas['weekday_1w'][0]:.3f}"
          f"  ({best_gammas['weekday_1w'][1]:.4f})")
    print(f"  best gamma by weekend Top-100      : {best_gammas['weekend_1w'][0]:.3f}"
          f"  ({best_gammas['weekend_1w'][1]:.4f})")


if __name__ == "__main__":
    main()
