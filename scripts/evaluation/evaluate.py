"""
Comprehensive evaluation of Two-Phase PageRank.

Runs PageRank on random timeframes and produces detailed metrics,
per-timeframe breakdowns, temporal analysis, and visualizations.

Usage:
    python scripts/evaluation/evaluate.py                    # 49 timeframes (default)
    python scripts/evaluation/evaluate.py --num-frames 10    # quick evaluation
    python scripts/evaluation/evaluate.py --seed 123         # different seed
"""
import argparse
import json
import logging
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
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core import build_two_phase_matrix, run_power_iteration, PARAMS
import config

logging.basicConfig(level=logging.WARNING)


def load_data(npz_path):
    """Load graph and popularity data."""
    with open(config.GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)

    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    loader = np.load(str(npz_path), allow_pickle=True)
    matrix = csr_matrix(
        (loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']),
        shape=loader['matrix_shape']
    )
    times = list(loader['times'])
    pop_link_ids = list(loader['link_ids'])

    return graph_data, links, N, lid_to_idx, matrix, times, pop_link_ids


def extract_teleportation(matrix, t_idx, pop_link_ids, lid_to_idx, N):
    """Extract and normalize teleportation vector for a timeframe."""
    row = matrix.getrow(t_idx)
    E_N = np.zeros(N)
    for i, val in zip(row.indices, row.data):
        lid = pop_link_ids[i]
        if lid in lid_to_idx:
            E_N[lid_to_idx[lid]] = float(val)
    s = E_N.sum()
    if s > 0:
        E_N /= s
    else:
        E_N = np.ones(N) / N
    return E_N


def compute_metrics(E_N, v_final, top_k=100):
    """Compute comprehensive metrics between ground truth and prediction."""
    # MRE metrics
    rel_errors = np.abs(E_N - v_final) / (E_N + 1e-9)
    overall_mre = np.mean(rel_errors)

    top_idx = np.argsort(E_N)[::-1][:top_k]
    top100_mre = np.mean(rel_errors[top_idx])

    # Bottom links MRE (to check if we over-predict low-traffic links)
    bottom_idx = np.argsort(E_N)[:top_k]
    bottom_mre = np.mean(rel_errors[bottom_idx])

    # Absolute error
    abs_errors = np.abs(E_N - v_final)
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(abs_errors ** 2))

    # Correlation
    mask = E_N > 0
    if np.sum(mask) > 100:
        pearson_r, _ = pearsonr(E_N[mask], v_final[mask])
        spearman_r, _ = spearmanr(E_N[mask], v_final[mask])
    else:
        pearson_r, spearman_r = 0.0, 0.0

    # Rank preservation: how many of the true top-100 are in predicted top-100
    pred_top_idx = set(np.argsort(v_final)[::-1][:top_k])
    true_top_idx = set(top_idx)
    rank_overlap = len(pred_top_idx & true_top_idx)

    # KL divergence (truth || prediction)
    eps = 1e-12
    e_safe = np.maximum(E_N, eps)
    v_safe = np.maximum(v_final, eps)
    e_norm = e_safe / e_safe.sum()
    v_norm = v_safe / v_safe.sum()
    kl_div = np.sum(e_norm * np.log(e_norm / v_norm))

    # Max relative error in top-100
    max_rel_error_top = np.max(rel_errors[top_idx])

    return {
        'overall_mre': float(overall_mre),
        'top100_mre': float(top100_mre),
        'bottom100_mre': float(bottom_mre),
        'mae': float(mae),
        'rmse': float(rmse),
        'pearson_r': float(pearson_r),
        'spearman_r': float(spearman_r),
        'rank_overlap_top100': int(rank_overlap),
        'kl_divergence': float(kl_div),
        'max_rel_error_top100': float(max_rel_error_top),
    }


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_stat(label, value, unit=""):
    if isinstance(value, float):
        print(f"  {label:.<50} {value:>10.4f} {unit}")
    else:
        print(f"  {label:.<50} {str(value):>10} {unit}")


def generate_plots(all_metrics, output_dir):
    """Generate evaluation visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('ggplot')

    n = len(all_metrics)

    # Extract arrays
    top100s = [m['top100_mre'] for m in all_metrics]
    overalls = [m['overall_mre'] for m in all_metrics]
    pearsons = [m['pearson_r'] for m in all_metrics]
    overlaps = [m['rank_overlap_top100'] for m in all_metrics]
    timeframe_labels = [m['timeframe'] for m in all_metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Two-Phase PageRank Evaluation ({n} Timeframes)', fontsize=16, fontweight='bold')

    # 1. MRE distribution
    axes[0, 0].hist(top100s, bins=20, alpha=0.7, color='#e63946', label='Top-100 MRE', edgecolor='black', linewidth=0.3)
    axes[0, 0].hist(overalls, bins=20, alpha=0.7, color='#457b9d', label='Overall MRE', edgecolor='black', linewidth=0.3)
    axes[0, 0].axvline(np.mean(top100s), color='#e63946', linestyle='--', linewidth=2, label=f'Top-100 mean={np.mean(top100s):.4f}')
    axes[0, 0].axvline(np.mean(overalls), color='#457b9d', linestyle='--', linewidth=2, label=f'Overall mean={np.mean(overalls):.4f}')
    axes[0, 0].set_xlabel('MRE')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('MRE Distribution Across Timeframes')
    axes[0, 0].legend(fontsize=8)

    # 2. Top-100 vs Overall scatter
    axes[0, 1].scatter(overalls, top100s, alpha=0.6, color='steelblue', edgecolor='black', linewidth=0.3)
    axes[0, 1].set_xlabel('Overall MRE')
    axes[0, 1].set_ylabel('Top-100 MRE')
    axes[0, 1].set_title('Top-100 vs Overall MRE')
    # Add diagonal reference
    lim = max(max(overalls), max(top100s)) * 1.1
    axes[0, 1].plot([0, lim], [0, lim], 'k--', alpha=0.3)

    # 3. Pearson correlation distribution
    axes[1, 0].hist(pearsons, bins=20, color='seagreen', edgecolor='black', linewidth=0.3)
    axes[1, 0].axvline(np.mean(pearsons), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean r={np.mean(pearsons):.4f}')
    axes[1, 0].set_xlabel('Pearson Correlation')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Prediction Correlation')
    axes[1, 0].legend()

    # 4. Rank overlap distribution
    axes[1, 1].hist(overlaps, bins=range(min(overlaps), max(overlaps) + 2), color='coral', edgecolor='black', linewidth=0.3)
    axes[1, 1].axvline(np.mean(overlaps), color='darkred', linestyle='--', linewidth=2, label=f'Mean={np.mean(overlaps):.0f}/100')
    axes[1, 1].set_xlabel('Top-100 Rank Overlap')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('How Many True Top-100 Links Are in Predicted Top-100')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_summary.png', dpi=200, bbox_inches='tight')
    plt.close()

    # 5. Temporal MRE heatmap (by hour and day-of-week)
    hourly_top100 = defaultdict(list)
    hourly_overall = defaultdict(list)
    dow_top100 = defaultdict(list)

    for m in all_metrics:
        try:
            dt = datetime.strptime(m['timeframe'], "%Y-%m-%d %H:%M:%S")
            hourly_top100[dt.hour].append(m['top100_mre'])
            hourly_overall[dt.hour].append(m['overall_mre'])
            dow_top100[dt.strftime('%A')].append(m['top100_mre'])
        except:
            pass

    if hourly_top100:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('MRE by Time of Day', fontsize=14, fontweight='bold')

        hours = sorted(hourly_top100.keys())
        avg_t100 = [np.mean(hourly_top100[h]) for h in hours]
        avg_all = [np.mean(hourly_overall[h]) for h in hours]

        axes[0].bar(hours, avg_t100, color='#e63946', alpha=0.8, edgecolor='black', linewidth=0.3)
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Top-100 MRE')
        axes[0].set_title('Top-100 MRE by Hour')
        axes[0].set_xticks(range(0, 24, 3))

        axes[1].bar(hours, avg_all, color='#457b9d', alpha=0.8, edgecolor='black', linewidth=0.3)
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Overall MRE')
        axes[1].set_title('Overall MRE by Hour')
        axes[1].set_xticks(range(0, 24, 3))

        plt.tight_layout()
        plt.savefig(output_dir / 'evaluation_temporal.png', dpi=200, bbox_inches='tight')
        plt.close()

    print(f"  Plots saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Two-Phase PageRank evaluation')
    parser.add_argument('--num-frames', type=int, default=49,
                        help='Number of random timeframes (default: 49)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--top-k', type=int, default=100,
                        help='Number of top links for Top-K MRE (default: 100)')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    graph_data, links, N, lid_to_idx, matrix, times, pop_link_ids = load_data(config.POPULARITY_NPZ)

    # Sample timeframes
    random.seed(args.seed)
    num_frames = min(args.num_frames, len(times))
    sample_times = random.sample(times, num_frames)

    # Build matrix
    print("Building 2N x 2N transition matrix...")
    M_2N = build_two_phase_matrix(graph_data)

    # Print configuration
    print_section("EVALUATION CONFIGURATION")
    print_stat("Network size (N)", N, "links")
    print_stat("Timeframes", num_frames)
    print_stat("Random seed", args.seed)
    print_stat("Top-K", args.top_k)
    print_stat("mu", PARAMS['mu'])
    print_stat("damping", PARAMS['damping'])
    print_stat("beta", PARAMS['beta'])

    # Run evaluation
    print_section("RUNNING EVALUATION")
    print(f"  {'#':>3} | {'Timeframe':>20} | {'Top-100':>8} | {'Overall':>8} | {'Pearson':>8} | {'Rank':>5} | {'Time':>5}")
    print(f"  {'-' * 75}")

    all_metrics = []
    start_all = time.time()

    # Accumulate per-link errors for identifying worst links
    per_link_abs_error_sum = np.zeros(N)
    per_link_rel_error_sum = np.zeros(N)

    for idx, t_str in enumerate(sample_times):
        start = time.time()
        t_idx = times.index(t_str)

        E_N = extract_teleportation(matrix, t_idx, pop_link_ids, lid_to_idx, N)
        E_2N = np.concatenate([E_N, np.zeros(N)])

        v_2N = run_power_iteration(M_2N, E_2N)
        v_final = v_2N[:N] + v_2N[N:]
        v_final /= v_final.sum()

        metrics = compute_metrics(E_N, v_final, top_k=args.top_k)
        metrics['timeframe'] = t_str
        elapsed = time.time() - start

        # Accumulate per-link errors
        per_link_abs_error_sum += np.abs(E_N - v_final)
        per_link_rel_error_sum += np.abs(E_N - v_final) / (E_N + 1e-9)

        all_metrics.append(metrics)

        print(f"  {idx+1:3} | {t_str:>20} | {metrics['top100_mre']:8.4f} | "
              f"{metrics['overall_mre']:8.4f} | {metrics['pearson_r']:8.4f} | "
              f"{metrics['rank_overlap_top100']:>3}/100 | {elapsed:4.1f}s")

    total_time = time.time() - start_all

    # Aggregate results
    print_section("AGGREGATE RESULTS")

    metric_keys = ['overall_mre', 'top100_mre', 'bottom100_mre', 'mae', 'rmse',
                   'pearson_r', 'spearman_r', 'rank_overlap_top100', 'kl_divergence', 'max_rel_error_top100']

    agg = {}
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        agg[key] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
        }

    print(f"\n  {'Metric':.<40} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-' * 80}")
    nice_names = {
        'overall_mre': 'Overall MRE',
        'top100_mre': 'Top-100 MRE',
        'bottom100_mre': 'Bottom-100 MRE',
        'mae': 'Mean Absolute Error',
        'rmse': 'Root Mean Squared Error',
        'pearson_r': 'Pearson Correlation',
        'spearman_r': 'Spearman Rank Correlation',
        'rank_overlap_top100': 'Top-100 Rank Overlap',
        'kl_divergence': 'KL Divergence',
        'max_rel_error_top100': 'Max Relative Error (Top-100)',
    }
    for key in metric_keys:
        name = nice_names.get(key, key)
        a = agg[key]
        print(f"  {name:.<40} {a['mean']:>10.4f} {a['std']:>10.4f} {a['min']:>10.4f} {a['max']:>10.4f}")

    # Best and worst timeframes
    print_section("BEST & WORST TIMEFRAMES")
    sorted_by_top100 = sorted(all_metrics, key=lambda m: m['top100_mre'])

    print("\n  Best 5 (lowest Top-100 MRE):")
    for m in sorted_by_top100[:5]:
        print(f"    {m['timeframe']}  Top-100={m['top100_mre']:.4f}  Overall={m['overall_mre']:.4f}  r={m['pearson_r']:.4f}")

    print("\n  Worst 5 (highest Top-100 MRE):")
    for m in sorted_by_top100[-5:]:
        print(f"    {m['timeframe']}  Top-100={m['top100_mre']:.4f}  Overall={m['overall_mre']:.4f}  r={m['pearson_r']:.4f}")

    # Worst-predicted links (highest average relative error)
    print_section("MOST DIFFICULT LINKS (highest avg relative error)")
    avg_rel_error = per_link_rel_error_sum / num_frames
    worst_link_idx = np.argsort(avg_rel_error)[::-1][:15]

    print(f"\n  {'Rank':>4}  {'Link ID':>10}  {'Avg RelErr':>10}  {'Length':>8}  {'Speed':>8}  {'Lanes':>5}")
    for rank, idx in enumerate(worst_link_idx, 1):
        lid = links[idx]
        info = graph_data['links'][lid]
        print(f"  {rank:>4}  {lid:>10}  {avg_rel_error[idx]:>10.4f}  {info.get('length', 0):>7.1f}m  "
              f"{info.get('speed', 0):>7.2f}  {info.get('lanes', 1):>5.1f}")

    # Temporal patterns
    print_section("TEMPORAL PATTERNS")
    hourly = defaultdict(list)
    dow = defaultdict(list)
    for m in all_metrics:
        try:
            dt = datetime.strptime(m['timeframe'], "%Y-%m-%d %H:%M:%S")
            hourly[dt.hour].append(m['top100_mre'])
            dow[dt.strftime('%A')].append(m['top100_mre'])
        except:
            pass

    if hourly:
        print("\n  Top-100 MRE by hour of day:")
        for h in sorted(hourly.keys()):
            vals = hourly[h]
            bar = '#' * int(np.mean(vals) / max(0.001, max(np.mean(v) for v in hourly.values())) * 30)
            print(f"    {h:02d}:00  mean={np.mean(vals):.4f}  n={len(vals):>3}  {bar}")

    if dow:
        print("\n  Top-100 MRE by day of week:")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for d in day_order:
            if d in dow:
                vals = dow[d]
                print(f"    {d:<10}  mean={np.mean(vals):.4f}  n={len(vals):>3}")

    print(f"\n  Total evaluation time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Generate plots
    print_section("GENERATING PLOTS")
    plot_dir = config.FIGURES_DIR / 'evaluation'
    generate_plots(all_metrics, plot_dir)

    # Save results
    print_section("SAVING RESULTS")

    results_data = {
        'configuration': {
            'num_frames': num_frames,
            'seed': args.seed,
            'top_k': args.top_k,
            'params': {k: v for k, v in PARAMS.items()},
        },
        'aggregate': agg,
        'per_timeframe': all_metrics,
    }

    json_file = config.RESULTS_DIR / 'evaluation_results.json'
    with open(json_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"  JSON results: {json_file}")

    txt_file = config.RESULTS_DIR / 'evaluation_results.txt'
    with open(txt_file, 'w') as f:
        f.write(f"Two-Phase PageRank Evaluation Results\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Timeframes: {num_frames}, Seed: {args.seed}\n")
        f.write(f"Parameters: mu={PARAMS['mu']}, d={PARAMS['damping']}, beta={PARAMS['beta']}\n\n")

        f.write(f"AGGREGATE METRICS:\n")
        for key in metric_keys:
            name = nice_names.get(key, key)
            a = agg[key]
            f.write(f"  {name}: mean={a['mean']:.6f}, std={a['std']:.6f}, "
                    f"min={a['min']:.6f}, max={a['max']:.6f}\n")

        f.write(f"\nPER-TIMEFRAME RESULTS:\n")
        f.write(f"{'Timeframe':>22}  {'Top-100':>8}  {'Overall':>8}  {'Pearson':>8}  {'Rank':>5}\n")
        f.write(f"{'-' * 60}\n")
        for m in sorted(all_metrics, key=lambda x: x['timeframe']):
            f.write(f"{m['timeframe']:>22}  {m['top100_mre']:>8.4f}  {m['overall_mre']:>8.4f}  "
                    f"{m['pearson_r']:>8.4f}  {m['rank_overlap_top100']:>3}/100\n")

    print(f"  Text results: {txt_file}")
    print(f"\nDone.")


if __name__ == '__main__':
    main()
