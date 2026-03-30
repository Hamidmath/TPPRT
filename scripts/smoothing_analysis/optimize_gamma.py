"""
Optimal smoothing parameter (gamma) search via Monte Carlo cross-validation.

Splits raw trajectory data into held-out halves, applies graph-diffusion
smoothing at various gamma levels to one half, and measures how well it
predicts the other half -- both directly (MSE, correlation) and through
the downstream PageRank pipeline.

Includes built-in verification with a second independent seed to confirm
the optimum is robust, and generates publication-quality plots.

Usage:
    python scripts/smoothing_analysis/optimize_gamma.py
    python scripts/smoothing_analysis/optimize_gamma.py --num-splits 20   # faster
    python scripts/smoothing_analysis/optimize_gamma.py --skip-verify      # skip 2nd seed
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core import build_two_phase_matrix, PARAMS
import config

OUTPUT_DIR = config.FIGURES_DIR / 'smoothing_analysis'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_directed_adjacency(city_graph, link_ids):
    """Build row-stochastic adjacency matrix for diffusion."""
    n = len(link_ids)
    id_to_idx = {lid: i for i, lid in enumerate(link_ids)}
    row_ind, col_ind, data = [], [], []
    adj = city_graph.get('adjacency', {})
    for lid, out_links in adj.items():
        if lid not in id_to_idx:
            continue
        i = id_to_idx[lid]
        valid_outs = [nid for nid in out_links if nid in id_to_idx]
        if not valid_outs:
            continue
        for out_lid in valid_outs:
            row_ind.append(i)
            col_ind.append(id_to_idx[out_lid])
            data.append(1.0 / len(valid_outs))
    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))


def load_popularity_matrix(filepath):
    loader = np.load(str(filepath), allow_pickle=True)
    matrix = sparse.csr_matrix(
        (loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']),
        shape=loader['matrix_shape']
    )
    return {
        'matrix': matrix,
        'times': list(loader['times']),
        'link_ids': list(loader['link_ids'])
    }


def identify_splits(times):
    """Group timeframes by (weekday, time-of-day) for paired splits."""
    tf = defaultdict(list)
    for t_idx, t_str in enumerate(times):
        dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
        tf[(dt.weekday(), dt.strftime("%H:%M:%S"))].append(t_idx)
    return {k: v for k, v in tf.items() if len(v) >= 2}


# ---------------------------------------------------------------------------
# Core cross-validation engine
# ---------------------------------------------------------------------------

def run_cross_validation(raw_data, city_graph, M_2N_T, P_dir,
                         gammas, num_splits, seed, alpha, damping,
                         N_raw, N_graph, raw_to_graph, valid_raw,
                         valid_splits, label=""):
    """
    Run Monte Carlo cross-validation for a set of gamma values.

    For each split:
      1. Divide timeframes into Set A (train) and Set B (test)
      2. For each gamma, smooth Set A and measure:
         - Direct prediction: MSE and correlation vs Set B
         - Downstream: PageRank on smoothed A vs PageRank on B
         - Variance retention: how much signal variance survives smoothing
    """
    print(f"\n  Cross-validation ({label}): seed={seed}, {num_splits} splits, {len(gammas)} gamma values")

    random.seed(seed)
    selected_keys = random.sample(list(valid_splits.keys()), min(num_splits, len(valid_splits)))

    n_g = len(gammas)
    avg_mse_raw = np.zeros(n_g)
    avg_corr_raw = np.zeros(n_g)
    avg_mse_pr = np.zeros(n_g)
    avg_corr_pr = np.zeros(n_g)
    avg_var_ret = np.zeros(n_g)

    valid_count = 0
    batch_size = 10

    for b_start in range(0, len(selected_keys), batch_size):
        b_keys = selected_keys[b_start:b_start + batch_size]
        pct = (b_start + len(b_keys)) / len(selected_keys) * 100
        print(f"    Processing splits {b_start+1}-{b_start+len(b_keys)}/{len(selected_keys)} ({pct:.0f}%)...", flush=True)

        target_B_list = []
        c_A_list = []

        for k in b_keys:
            t_indices = valid_splits[k]
            np.random.seed(hash(k) % (2**32))
            idx_copy = list(t_indices)
            random.shuffle(idx_copy)
            half = len(idx_copy) // 2
            set_A, set_B = idx_copy[:half], idx_copy[half:]

            raw_A = np.zeros(N_raw)
            for idx in set_A:
                row = raw_data['matrix'].getrow(idx)
                for i, val in zip(row.indices, row.data):
                    raw_A[i] += float(val)

            raw_B = np.zeros(N_raw)
            for idx in set_B:
                row = raw_data['matrix'].getrow(idx)
                for i, val in zip(row.indices, row.data):
                    raw_B[i] += float(val)

            if raw_B.sum() <= 0:
                continue

            target_B_list.append(raw_B / raw_B.sum())
            c_A_list.append(raw_A + alpha)

        if not c_A_list:
            continue

        bs = len(c_A_list)
        valid_count += bs

        # PageRank for target B
        E_2N_B = np.zeros((2 * N_graph, bs))
        for b in range(bs):
            E_2N_B[raw_to_graph, b] = target_B_list[b][valid_raw]
            s = E_2N_B[:N_graph, b].sum()
            if s > 0:
                E_2N_B[:N_graph, b] /= s
            else:
                E_2N_B[:N_graph, b] = 1.0 / N_graph

        v_2N_B = E_2N_B.copy()
        for _ in range(60):
            v_next = damping * M_2N_T.dot(v_2N_B) + (1 - damping) * E_2N_B
            S = np.sum(v_next, axis=0)
            missing = 1.0 - S
            missing[missing < 0] = 0
            v_next += E_2N_B * missing
            v_2N_B = v_next

        tpr_full = v_2N_B[:N_graph, :] + v_2N_B[N_graph:, :]
        tpr = np.zeros((N_raw, bs))
        for b in range(bs):
            tpr[valid_raw, b] = tpr_full[raw_to_graph, b]
            s = tpr[:, b].sum()
            if s > 0:
                tpr[:, b] /= s

        # Process Set A across all gammas
        total_cols = bs * n_g
        E_2N_A = np.zeros((2 * N_graph, total_cols))
        dist_A = np.zeros((N_raw, total_cols))
        base_vars = []

        for b in range(bs):
            cA = c_A_list[b]
            P_cA = P_dir.dot(cA)
            base_vars.append(np.var(cA / cA.sum()))

            for g_i, g in enumerate(gammas):
                c_diff = (1 - g) * cA + g * P_cA
                dist = c_diff / c_diff.sum()
                col = b * n_g + g_i
                dist_A[:, col] = dist

                E_2N_A[raw_to_graph, col] = dist[valid_raw]
                s = E_2N_A[:N_graph, col].sum()
                if s > 0:
                    E_2N_A[:N_graph, col] /= s
                else:
                    E_2N_A[:N_graph, col] = 1.0 / N_graph

        v_2N_A = E_2N_A.copy()
        for _ in range(60):
            v_next = damping * M_2N_T.dot(v_2N_A) + (1 - damping) * E_2N_A
            S = np.sum(v_next, axis=0)
            missing = 1.0 - S
            missing[missing < 0] = 0
            v_next += E_2N_A * missing
            v_2N_A = v_next

        dpr_full = v_2N_A[:N_graph, :] + v_2N_A[N_graph:, :]
        dpr = np.zeros((N_raw, total_cols))
        for col in range(total_cols):
            dpr[valid_raw, col] = dpr_full[raw_to_graph, col]
            s = dpr[:, col].sum()
            if s > 0:
                dpr[:, col] /= s

        # Aggregate metrics
        for b in range(bs):
            tB = target_B_list[b]
            tprB = tpr[:, b]
            bv = base_vars[b]
            for g_i in range(n_g):
                col = b * n_g + g_i
                mse_r = np.mean((dist_A[:, col] - tB) ** 2)
                r_r, _ = pearsonr(dist_A[:, col], tB)
                mse_p = np.mean((dpr[:, col] - tprB) ** 2)
                r_p, _ = pearsonr(dpr[:, col], tprB)

                avg_mse_raw[g_i] += mse_r
                if not np.isnan(r_r):
                    avg_corr_raw[g_i] += r_r
                avg_mse_pr[g_i] += mse_p
                if not np.isnan(r_p):
                    avg_corr_pr[g_i] += r_p
                avg_var_ret[g_i] += (np.var(dist_A[:, col]) / bv)

    if valid_count > 0:
        avg_mse_raw /= valid_count
        avg_corr_raw /= valid_count
        avg_mse_pr /= valid_count
        avg_corr_pr /= valid_count
        avg_var_ret = (avg_var_ret / valid_count) * 100

    # Find optima
    opt_mse = gammas[np.argmin(avg_mse_raw)]
    opt_corr = gammas[np.argmax(avg_corr_raw)]
    opt_pr = gammas[np.argmin(avg_mse_pr)]

    return {
        'seed': seed,
        'splits_used': valid_count,
        'gammas': gammas,
        'mse_raw': avg_mse_raw.tolist(),
        'corr_raw': avg_corr_raw.tolist(),
        'mse_pr': avg_mse_pr.tolist(),
        'corr_pr': avg_corr_pr.tolist(),
        'variance_ret': avg_var_ret.tolist(),
        'optimal_gamma_mse': opt_mse,
        'optimal_gamma_corr': opt_corr,
        'optimal_gamma_pr': opt_pr,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results_table(results, label):
    gammas = results['gammas']
    print(f"\n  {'gamma':>6} | {'MSE (raw)':>14} | {'Correlation':>12} | {'MSE (PR)':>14} | {'Var Ret %':>9}")
    print(f"  {'-' * 6}-+-{'-' * 14}-+-{'-' * 12}-+-{'-' * 14}-+-{'-' * 9}")
    for i, g in enumerate(gammas):
        star = " <--" if g == results['optimal_gamma_mse'] else ""
        print(f"  {g:6.2f} | {results['mse_raw'][i]:.6e} | {results['corr_raw'][i]:>12.6f} | "
              f"{results['mse_pr'][i]:.6e} | {results['variance_ret'][i]:>7.1f}%{star}")

    print(f"\n  Optimal gamma (min predictive MSE):  {results['optimal_gamma_mse']}")
    print(f"  Optimal gamma (max correlation):     {results['optimal_gamma_corr']}")
    print(f"  Optimal gamma (min downstream MSE):  {results['optimal_gamma_pr']}")
    print(f"  Splits evaluated: {results['splits_used']}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_plots(primary, verification=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use('ggplot')

    gammas = primary['gammas']

    # ---------- Figure 1: Predictive Error & Correlation ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Cross-Validated Optimal Smoothing Parameter', fontsize=16, fontweight='bold')

    # MSE
    ax = axes[0]
    ax.plot(gammas, primary['mse_raw'], 'o-', color='#e63946', linewidth=2, markersize=6, label='Predictive MSE')
    min_idx = np.argmin(primary['mse_raw'])
    ax.axvline(gammas[min_idx], color='#e63946', linestyle='--', alpha=0.5)
    ax.annotate(f"Optimal: $\\gamma$={gammas[min_idx]}",
                xy=(gammas[min_idx], primary['mse_raw'][min_idx]),
                xytext=(gammas[min_idx] + 0.08, primary['mse_raw'][min_idx] * 1.1),
                fontsize=11, fontweight='bold', color='#e63946',
                arrowprops=dict(arrowstyle='->', color='#e63946'))
    if verification:
        ax.plot(verification['gammas'], verification['mse_raw'], 's--', color='#e63946',
                alpha=0.5, linewidth=1.5, markersize=4, label=f"Verification (seed={verification['seed']})")
    ax.set_xlabel('Smoothing Parameter ($\\gamma$)', fontsize=12)
    ax.set_ylabel('Held-out Predictive MSE', fontsize=12)
    ax.set_title('Prediction Error (Lower = Better)')
    ax.legend()

    # Correlation
    ax = axes[1]
    ax.plot(gammas, primary['corr_raw'], 'o-', color='#457b9d', linewidth=2, markersize=6, label='Pearson Correlation')
    max_idx = np.argmax(primary['corr_raw'])
    ax.axvline(gammas[max_idx], color='#457b9d', linestyle='--', alpha=0.5)
    ax.annotate(f"Peak: $\\gamma$={gammas[max_idx]}",
                xy=(gammas[max_idx], primary['corr_raw'][max_idx]),
                xytext=(gammas[max_idx] + 0.08, primary['corr_raw'][max_idx] - 0.002),
                fontsize=11, fontweight='bold', color='#457b9d',
                arrowprops=dict(arrowstyle='->', color='#457b9d'))
    if verification:
        ax.plot(verification['gammas'], verification['corr_raw'], 's--', color='#457b9d',
                alpha=0.5, linewidth=1.5, markersize=4, label=f"Verification (seed={verification['seed']})")
    ax.set_xlabel('Smoothing Parameter ($\\gamma$)', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('Correlation with Held-out Data (Higher = Better)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'gamma_cv_predictive.png', dpi=200, bbox_inches='tight')
    plt.close()

    # ---------- Figure 2: Downstream & Variance ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Downstream Impact of Smoothing', fontsize=16, fontweight='bold')

    ax = axes[0]
    ax.plot(gammas, primary['mse_pr'], '^-', color='#2a9d8f', linewidth=2, markersize=6, label='PageRank MSE')
    min_pr_idx = np.argmin(primary['mse_pr'])
    ax.axvline(gammas[min_pr_idx], color='#2a9d8f', linestyle='--', alpha=0.5,
               label=f"Optimal $\\gamma$={gammas[min_pr_idx]}")
    if verification:
        ax.plot(verification['gammas'], verification['mse_pr'], 's--', color='#2a9d8f',
                alpha=0.5, linewidth=1.5, markersize=4, label=f"Verification (seed={verification['seed']})")
    ax.set_xlabel('Smoothing Parameter ($\\gamma$)', fontsize=12)
    ax.set_ylabel('Downstream PageRank MSE', fontsize=12)
    ax.set_title('PageRank Prediction Error (Lower = Better)')
    ax.legend()

    ax = axes[1]
    ax.plot(gammas, primary['variance_ret'], 'd-', color='#e9c46a', linewidth=2, markersize=6, label='Variance Retained')
    ax.axhline(100, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(gammas[min_idx], color='#e63946', linestyle='--', alpha=0.3, label=f"MSE optimum ($\\gamma$={gammas[min_idx]})")
    ax.set_xlabel('Smoothing Parameter ($\\gamma$)', fontsize=12)
    ax.set_ylabel('Signal Variance Retained (%)', fontsize=12)
    ax.set_title('Variance Retention (Higher = More Signal Preserved)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'gamma_cv_downstream.png', dpi=200, bbox_inches='tight')
    plt.close()

    # ---------- Figure 3: Bias-Variance Tradeoff ----------
    fig, ax = plt.subplots(figsize=(10, 6))

    def norm(arr):
        mn, mx = min(arr), max(arr)
        rng = mx - mn if mx > mn else 1
        return [(x - mn) / rng for x in arr]

    ax.plot(gammas, norm(primary['mse_raw']), 'o-', color='#e63946', linewidth=2.5,
            markersize=7, label='Prediction Error (normalized)')
    ax.plot(gammas, [1 - v for v in norm(primary['variance_ret'])], 'd-', color='#457b9d',
            linewidth=2.5, markersize=7, label='Signal Loss (normalized)')

    # Shade optimal region
    opt_g = gammas[min_idx]
    ax.axvspan(opt_g - 0.03, opt_g + 0.03, alpha=0.15, color='green', label=f'Optimal region ($\\gamma$={opt_g})')
    ax.axvline(opt_g, color='green', linestyle='--', alpha=0.5)

    ax.set_xlabel('Smoothing Parameter ($\\gamma$)', fontsize=13)
    ax.set_ylabel('Normalized Score (0 = best)', fontsize=13)
    ax.set_title('Bias-Variance Tradeoff: Finding the Sweet Spot', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'gamma_cv_tradeoff.png', dpi=200, bbox_inches='tight')
    plt.close()

    # ---------- Figure 4: Verification comparison (if available) ----------
    if verification:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(primary['gammas'], primary['mse_raw'], 'o-', color='#e63946', linewidth=2,
                markersize=7, label=f"Primary (seed={primary['seed']}, n={primary['splits_used']})")
        ax.plot(verification['gammas'], verification['mse_raw'], 's-', color='#457b9d', linewidth=2,
                markersize=7, label=f"Verification (seed={verification['seed']}, n={verification['splits_used']})")

        ax.axvline(primary['optimal_gamma_mse'], color='#e63946', linestyle='--', alpha=0.5,
                   label=f"Primary opt: $\\gamma$={primary['optimal_gamma_mse']}")
        ax.axvline(verification['optimal_gamma_mse'], color='#457b9d', linestyle='--', alpha=0.5,
                   label=f"Verify opt: $\\gamma$={verification['optimal_gamma_mse']}")

        ax.set_xlabel('Smoothing Parameter ($\\gamma$)', fontsize=13)
        ax.set_ylabel('Held-out Predictive MSE', fontsize=13)
        ax.set_title('Reproducibility: Two Independent Cross-Validations', fontsize=15, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'gamma_cv_verification.png', dpi=200, bbox_inches='tight')
        plt.close()

    print(f"\n  Plots saved to {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Find optimal smoothing gamma via cross-validation')
    parser.add_argument('--num-splits', type=int, default=49,
                        help='Number of Monte Carlo splits (default: 49)')
    parser.add_argument('--skip-verify', action='store_true',
                        help='Skip independent verification with second seed')
    args = parser.parse_args()

    start_all = time.time()

    # Load data
    print("Loading data...")
    with open(config.GRAPH_FILE) as f:
        city_graph = json.load(f)
    raw_data = load_popularity_matrix(config.POPULARITY_RAW_NPZ)

    links_raw = raw_data['link_ids']
    N_raw = len(links_raw)
    links_graph = list(city_graph['links'].keys())
    N_graph = len(links_graph)

    # Map between raw and graph link indices
    graph_idx_map = {lid: i for i, lid in enumerate(links_graph)}
    raw_to_graph = []
    valid_raw = []
    for i, lid in enumerate(links_raw):
        if lid in graph_idx_map:
            raw_to_graph.append(graph_idx_map[lid])
            valid_raw.append(i)
    raw_to_graph = np.array(raw_to_graph)
    valid_raw = np.array(valid_raw)

    print(f"  Raw links: {N_raw}, Graph links: {N_graph}, Mapped: {len(raw_to_graph)}")

    # Build adjacency for diffusion
    print("Building adjacency matrix...")
    P_dir = build_directed_adjacency(city_graph, links_raw)

    # Build Two-Phase matrix for downstream evaluation
    print("Building Two-Phase Matrix for downstream evaluation...")
    PARAMS['alpha_s'] = 0.0
    PARAMS['alpha_l'] = 0.0
    PARAMS['beta'] = 0.2
    M_2N = build_two_phase_matrix(city_graph)
    M_2N_T = M_2N.transpose().tocsr()

    # Identify valid splits
    valid_splits = identify_splits(raw_data['times'])
    print(f"  Valid split groups: {len(valid_splits)}")

    # Gamma grid (coarse + fine around expected optimum)
    gammas = [0.0, 0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.24,
              0.25, 0.26, 0.27, 0.28, 0.30, 0.35, 0.40, 0.50, 0.60]

    alpha = 0.1   # Laplace smoothing
    damping = 0.89

    # ---- PRIMARY CROSS-VALIDATION ----
    print("\n" + "=" * 75)
    print("  PRIMARY CROSS-VALIDATION")
    print("=" * 75)
    primary = run_cross_validation(
        raw_data, city_graph, M_2N_T, P_dir,
        gammas, args.num_splits, seed=42, alpha=alpha, damping=damping,
        N_raw=N_raw, N_graph=N_graph, raw_to_graph=raw_to_graph,
        valid_raw=valid_raw, valid_splits=valid_splits, label="Primary"
    )
    print_results_table(primary, "Primary")

    # ---- VERIFICATION (independent seed) ----
    verification = None
    if not args.skip_verify:
        print("\n" + "=" * 75)
        print("  VERIFICATION (Independent Seed)")
        print("=" * 75)
        verify_splits = min(80, len(valid_splits))
        verification = run_cross_validation(
            raw_data, city_graph, M_2N_T, P_dir,
            gammas, verify_splits, seed=123, alpha=alpha, damping=damping,
            N_raw=N_raw, N_graph=N_graph, raw_to_graph=raw_to_graph,
            valid_raw=valid_raw, valid_splits=valid_splits, label="Verification"
        )
        print_results_table(verification, "Verification")

    # ---- FINAL VERDICT ----
    print("\n" + "=" * 75)
    print("  FINAL VERDICT")
    print("=" * 75)

    opt = primary['optimal_gamma_mse']
    print(f"\n  Primary optimal gamma (min MSE):        {opt}")
    print(f"  Primary optimal gamma (max correlation): {primary['optimal_gamma_corr']}")
    print(f"  Primary optimal gamma (min PR MSE):      {primary['optimal_gamma_pr']}")

    if verification:
        v_opt = verification['optimal_gamma_mse']
        print(f"\n  Verification optimal gamma (min MSE):   {v_opt}")
        print(f"  Verification optimal gamma (max corr):  {verification['optimal_gamma_corr']}")
        print(f"  Verification optimal gamma (min PR MSE):{verification['optimal_gamma_pr']}")

        if opt == v_opt:
            print(f"\n  CONFIRMED: Both seeds agree on gamma* = {opt}")
        else:
            print(f"\n  Primary and verification optima differ ({opt} vs {v_opt})")
            print(f"  This suggests the optimum lies in the range [{min(opt, v_opt)}, {max(opt, v_opt)}]")

    # Improvement at optimal gamma vs no smoothing
    g0_idx = gammas.index(0.0)
    opt_idx = gammas.index(opt)
    mse_reduction = (1 - primary['mse_raw'][opt_idx] / primary['mse_raw'][g0_idx]) * 100
    corr_improvement = primary['corr_raw'][opt_idx] - primary['corr_raw'][g0_idx]
    var_retained = primary['variance_ret'][opt_idx]

    print(f"\n  At gamma* = {opt} vs no smoothing (gamma=0):")
    print(f"    Predictive MSE reduced by:  {mse_reduction:.1f}%")
    print(f"    Correlation improved by:     +{corr_improvement:.4f}")
    print(f"    Signal variance retained:    {var_retained:.1f}%")
    print(f"    Downstream PR MSE reduced:   {(1 - primary['mse_pr'][opt_idx] / primary['mse_pr'][g0_idx]) * 100:.1f}%")

    total_time = time.time() - start_all
    print(f"\n  Total time: {total_time:.0f}s ({total_time / 60:.1f} min)")

    # ---- Generate plots ----
    print("\n  Generating plots...")
    generate_plots(primary, verification)

    # ---- Save results ----
    results = {
        'primary': primary,
        'verification': verification,
        'conclusion': {
            'optimal_gamma': opt,
            'mse_reduction_pct': mse_reduction,
            'correlation_improvement': corr_improvement,
            'variance_retained_pct': var_retained,
        }
    }

    json_file = config.RESULTS_DIR / 'gamma_optimization.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {json_file}")


if __name__ == '__main__':
    main()
