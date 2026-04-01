#!/usr/bin/env python3
"""
Damping factor sweep: run 49-timeframe evaluation for d in [0.80, 0.82, ..., 0.96].
Data loaded once, matrix rebuilt per damping value, all E_N cached.
"""

import sys
import json
import random
import time as _time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config
from core.pagerank import build_two_phase_matrix, run_power_iteration, PARAMS

OUT_DIR = Path(__file__).resolve().parent

DAMPING_VALUES = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96]
NUM_FRAMES = 49
SEED = 42
TOP_K = 100


def load_data():
    with open(config.GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)
    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    loader = np.load(str(config.POPULARITY_NPZ), allow_pickle=True)
    matrix = csr_matrix((
        loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']
    ), shape=loader['matrix_shape'])
    times = list(loader['times'])
    pop_ids = list(loader['link_ids'])

    return graph_data, links, N, lid_to_idx, matrix, times, pop_ids


def cache_teleportation(matrix, times, sample_times, pop_ids, lid_to_idx, N):
    cache = {}
    for ts in sample_times:
        t_idx = times.index(ts)
        row = matrix.getrow(t_idx)
        E = np.zeros(N)
        for i, val in zip(row.indices, row.data):
            lid = pop_ids[i]
            if lid in lid_to_idx:
                E[lid_to_idx[lid]] = float(val)
        s = E.sum()
        E = E / s if s > 0 else np.ones(N) / N
        cache[ts] = E
    return cache


def compute_mre(E_N, v_final, top_k=100):
    rel = np.abs(E_N - v_final) / (E_N + 1e-9)
    overall = float(np.mean(rel))
    top_idx = np.argsort(E_N)[::-1][:top_k]
    top100 = float(np.mean(rel[top_idx]))
    mask = E_N > 0
    r, _ = pearsonr(E_N[mask], v_final[mask]) if np.sum(mask) > 100 else (0, 0)
    return overall, top100, float(r)


def main():
    t0 = _time.time()
    print("=" * 70)
    print("DAMPING FACTOR SWEEP  (49 timeframes × 9 damping values)")
    print("=" * 70)

    # Load once
    print("\nLoading data...")
    graph_data, links, N, lid_to_idx, matrix, times, pop_ids = load_data()
    print(f"  {N:,} links, {len(times):,} timeframes")

    # Sample timeframes
    random.seed(SEED)
    sample_times = random.sample(times, min(NUM_FRAMES, len(times)))
    print(f"  Sampled {len(sample_times)} timeframes (seed={SEED})")

    # Cache all teleportation vectors
    print("  Caching teleportation vectors...")
    E_cache = cache_teleportation(matrix, times, sample_times, pop_ids, lid_to_idx, N)
    print(f"  Cached {len(E_cache)} vectors in {_time.time()-t0:.1f}s")

    # Save original damping
    orig_damping = PARAMS['damping']

    results = {}

    for d in DAMPING_VALUES:
        PARAMS['damping'] = d
        print(f"\n  d={d:.2f}: building matrix...", end=' ', flush=True)
        M_2N = build_two_phase_matrix(graph_data)

        overalls, top100s, pearsons = [], [], []
        t1 = _time.time()

        for ts in sample_times:
            E_N = E_cache[ts]
            E_2N = np.concatenate([E_N, np.zeros(N)])
            v_2N = run_power_iteration(M_2N, E_2N)
            v = v_2N[:N] + v_2N[N:]
            v /= v.sum()

            ov, t100, r = compute_mre(E_N, v, TOP_K)
            overalls.append(ov)
            top100s.append(t100)
            pearsons.append(r)

        elapsed = _time.time() - t1
        results[d] = {
            'overall_mean': np.mean(overalls),
            'overall_std': np.std(overalls),
            'top100_mean': np.mean(top100s),
            'top100_std': np.std(top100s),
            'pearson_mean': np.mean(pearsons),
            'pearson_std': np.std(pearsons),
        }
        print(f"Top-100 MRE={np.mean(top100s):.4f} ± {np.std(top100s):.4f}  "
              f"Overall={np.mean(overalls):.4f}  r={np.mean(pearsons):.4f}  "
              f"({elapsed:.1f}s)")

    # Restore
    PARAMS['damping'] = orig_damping

    # ── Plot ──
    ds = sorted(results.keys())
    t100_means = [results[d]['top100_mean'] for d in ds]
    t100_stds = [results[d]['top100_std'] for d in ds]
    ov_means = [results[d]['overall_mean'] for d in ds]
    ov_stds = [results[d]['overall_std'] for d in ds]
    r_means = [results[d]['pearson_mean'] for d in ds]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: MRE vs damping
    ax = axes[0]
    ax.errorbar(ds, t100_means, yerr=t100_stds, fmt='-o', color='#e63946',
                capsize=4, lw=2, ms=8, label='Top-100 MRE')
    ax.errorbar(ds, ov_means, yerr=ov_stds, fmt='-s', color='#457b9d',
                capsize=4, lw=2, ms=8, label='Overall MRE')
    ax.set_xlabel('Damping Factor (d)', fontsize=12)
    ax.set_ylabel('Mean Relative Error', fontsize=12)
    ax.set_title('(A)  MRE vs Damping Factor', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ds)

    # Mark best
    best_idx = np.argmin(t100_means)
    ax.annotate(f'Best: d={ds[best_idx]:.2f}\nMRE={t100_means[best_idx]:.4f}',
                xy=(ds[best_idx], t100_means[best_idx]),
                xytext=(ds[best_idx]+0.02, t100_means[best_idx]+0.01),
                arrowprops=dict(arrowstyle='->', color='#e63946'),
                fontsize=9, fontweight='bold', color='#e63946')

    # Panel B: Pearson correlation vs damping
    ax = axes[1]
    ax.plot(ds, r_means, '-D', color='#2a9d8f', lw=2, ms=8)
    ax.set_xlabel('Damping Factor (d)', fontsize=12)
    ax.set_ylabel('Mean Pearson Correlation', fontsize=12)
    ax.set_title('(B)  Correlation vs Damping Factor', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ds)

    # Panel C: Top-100 MRE bar chart
    ax = axes[2]
    colors = ['#e63946' if i == best_idx else '#457b9d' for i in range(len(ds))]
    bars = ax.bar([f'{d:.2f}' for d in ds], t100_means, color=colors, alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    ax.errorbar(range(len(ds)), t100_means, yerr=t100_stds, fmt='none',
                ecolor='black', capsize=3)
    for i, (bar, v) in enumerate(zip(bars, t100_means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xlabel('Damping Factor (d)', fontsize=12)
    ax.set_ylabel('Top-100 MRE', fontsize=12)
    ax.set_title('(C)  Top-100 MRE by Damping', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Damping Factor Sensitivity (49 timeframes, seed={SEED}, '
                 f'$\\mu$=20, $\\beta$=0.9)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = OUT_DIR / 'fig_damping_sweep.png'
    plt.savefig(str(p), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: {p}")

    # Save JSON
    out = {str(d): results[d] for d in ds}
    jp = OUT_DIR / 'damping_sweep_results.json'
    with open(jp, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Results saved: {jp}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'d':>6s}  {'Top-100 MRE':>14s}  {'Overall MRE':>14s}  {'Pearson r':>12s}")
    print(f"  {'─'*50}")
    for d in ds:
        r = results[d]
        marker = ' ◄── best' if d == ds[best_idx] else ''
        print(f"  {d:6.2f}  {r['top100_mean']:10.4f} ± {r['top100_std']:.4f}"
              f"  {r['overall_mean']:10.4f} ± {r['overall_std']:.4f}"
              f"  {r['pearson_mean']:8.4f}{marker}")

    print(f"\n  Total time: {_time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
