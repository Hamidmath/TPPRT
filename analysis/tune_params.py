"""
Unified parameter tuning for Two-Phase PageRank.

Grid search over mu, damping, and beta on a configurable number of
random timeframes. Reports Top-100 MRE and Overall MRE for each
configuration and ranks results.

Usage:
    python scripts/tuning/tune_params.py                    # 49 timeframes (default)
    python scripts/tuning/tune_params.py --num-frames 10    # faster, 10 timeframes
    python scripts/tuning/tune_params.py --seed 123         # different random seed
"""
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import build_two_phase_matrix, run_power_iteration, PARAMS
import config

logging.basicConfig(level=logging.WARNING, format='%(message)s')


def load_teleportation_cache(graph_data, num_frames, seed):
    """Load and cache E_N vectors for the sampled timeframes."""
    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    loader = np.load(str(config.POPULARITY_NPZ), allow_pickle=True)
    matrix = csr_matrix(
        (loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']),
        shape=loader['matrix_shape']
    )
    times = list(loader['times'])
    pop_link_ids = list(loader['link_ids'])

    random.seed(seed)
    num_frames = min(num_frames, len(times))
    sample_times = random.sample(times, num_frames)

    E_N_cache = []
    for t_str in sample_times:
        t_idx = times.index(t_str)
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
        E_N_cache.append(E_N)

    return E_N_cache, N


def evaluate_config(graph_data, E_N_cache, N, mu, damping, beta):
    """Evaluate a single (mu, damping, beta) configuration."""
    PARAMS['mu'] = mu
    PARAMS['beta'] = beta
    PARAMS['damping'] = damping
    PARAMS['alpha_s'] = 0.0
    PARAMS['alpha_l'] = 0.0
    PARAMS['tau'] = 1.0
    PARAMS['top_k_boost'] = 1.0

    M_2N = build_two_phase_matrix(graph_data)

    top100_sum = 0.0
    overall_sum = 0.0
    num_frames = len(E_N_cache)

    for E_N in E_N_cache:
        E_2N = np.concatenate([E_N, np.zeros(N)])
        v_2N = run_power_iteration(M_2N, E_2N)
        v = v_2N[:N] + v_2N[N:]
        v /= v.sum()

        top_idx = np.argsort(E_N)[::-1][:100]
        top100_sum += np.mean(np.abs(E_N[top_idx] - v[top_idx]) / (E_N[top_idx] + 1e-9))
        overall_sum += np.mean(np.abs(E_N - v) / (E_N + 1e-9))

    return top100_sum / num_frames, overall_sum / num_frames


def main():
    parser = argparse.ArgumentParser(description='Tune mu, damping, beta for Two-Phase PageRank')
    parser.add_argument('--num-frames', type=int, default=49,
                        help='Number of random timeframes to evaluate (default: 49)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for timeframe sampling (default: 42)')
    parser.add_argument('--mu', type=str, default='0,1,2,3,5,8,10,12,15,20',
                        help='Comma-separated mu values (default: 0,1,2,3,5,8,10,12,15,20)')
    parser.add_argument('--damping', type=str, default='0.80,0.85,0.90',
                        help='Comma-separated damping values (default: 0.80,0.85,0.90)')
    parser.add_argument('--beta', type=str, default='0.0,0.2,0.5,0.9',
                        help='Comma-separated beta values (default: 0.0,0.2,0.5,0.9)')
    args = parser.parse_args()

    mu_vals = [float(x) for x in args.mu.split(',')]
    damping_vals = [float(x) for x in args.damping.split(',')]
    beta_vals = [float(x) for x in args.beta.split(',')]

    total_configs = len(mu_vals) * len(damping_vals) * len(beta_vals)

    print(f"Two-Phase PageRank Parameter Tuning")
    print(f"{'=' * 60}")
    print(f"  Timeframes : {args.num_frames} (seed={args.seed})")
    print(f"  mu values  : {mu_vals}")
    print(f"  damping    : {damping_vals}")
    print(f"  beta       : {beta_vals}")
    print(f"  Total      : {total_configs} configurations")
    print(f"{'=' * 60}\n")

    # Load data
    print("Loading graph and caching teleportation vectors...")
    with open(config.GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)

    E_N_cache, N = load_teleportation_cache(graph_data, args.num_frames, args.seed)
    print(f"Cached {len(E_N_cache)} teleportation vectors (N={N})\n")

    # Grid search
    print(f"{'#':>3} | {'mu':>4} | {'d':>5} | {'beta':>5} | {'Top-100':>8} | {'Overall':>8} | {'Time':>6}")
    print("-" * 58)

    results = []
    config_num = 0
    start_all = time.time()

    for mu in mu_vals:
        for beta in beta_vals:
            for damp in damping_vals:
                config_num += 1
                start = time.time()

                t100, tall = evaluate_config(graph_data, E_N_cache, N, mu, damp, beta)
                elapsed = time.time() - start

                results.append({
                    'mu': mu, 'damping': damp, 'beta': beta,
                    'top100_mre': t100, 'overall_mre': tall
                })

                print(f"{config_num:3} | {mu:4.0f} | {damp:5.2f} | {beta:5.2f} | "
                      f"{t100:8.4f} | {tall:8.4f} | {elapsed:5.1f}s")

    total_time = time.time() - start_all

    # Sort and display results
    results.sort(key=lambda r: r['top100_mre'])

    print(f"\n{'=' * 60}")
    print(f"TOP 10 CONFIGURATIONS (sorted by Top-100 MRE)")
    print(f"{'=' * 60}")
    print(f"{'Rank':>4} | {'mu':>4} | {'d':>5} | {'beta':>5} | {'Top-100':>8} | {'Overall':>8}")
    print("-" * 50)
    for i, r in enumerate(results[:10], 1):
        print(f"{i:4} | {r['mu']:4.0f} | {r['damping']:5.2f} | {r['beta']:5.2f} | "
              f"{r['top100_mre']:8.4f} | {r['overall_mre']:8.4f}")

    best = results[0]
    print(f"\nBest: mu={best['mu']:.0f}, damping={best['damping']:.2f}, beta={best['beta']:.2f}")
    print(f"      Top-100 MRE = {best['top100_mre']:.4f}, Overall MRE = {best['overall_mre']:.4f}")
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Save results
    out_file = config.RESULTS_DIR / 'tuning_results.txt'
    with open(out_file, 'w') as f:
        f.write(f"Parameter Tuning Results\n")
        f.write(f"Timeframes: {len(E_N_cache)}, Seed: {args.seed}\n")
        f.write(f"{'=' * 60}\n\n")

        f.write(f"TOP 10:\n")
        for i, r in enumerate(results[:10], 1):
            f.write(f"  {i:2}. mu={r['mu']:.0f}, d={r['damping']:.2f}, beta={r['beta']:.2f}"
                    f" -> top100={r['top100_mre']:.4f}, overall={r['overall_mre']:.4f}\n")

        f.write(f"\nALL CONFIGURATIONS (sorted by mu, beta, damping):\n")
        for r in sorted(results, key=lambda r: (r['mu'], r['beta'], r['damping'])):
            f.write(f"  mu={r['mu']:4.0f}, d={r['damping']:.2f}, beta={r['beta']:.2f}"
                    f" -> top100={r['top100_mre']:.4f}, overall={r['overall_mre']:.4f}\n")

    print(f"Results saved to {out_file}")

    # Also save as JSON for programmatic use
    json_file = config.RESULTS_DIR / 'tuning_results.json'
    with open(json_file, 'w') as f:
        json.dump({
            'settings': {
                'num_frames': len(E_N_cache),
                'seed': args.seed,
                'mu_values': mu_vals,
                'damping_values': damping_vals,
                'beta_values': beta_vals,
            },
            'best': best,
            'all_results': results,
        }, f, indent=2)
    print(f"JSON results saved to {json_file}")


if __name__ == '__main__':
    main()
