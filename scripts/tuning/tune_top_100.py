import json
import logging
import os
import random
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core import build_two_phase_matrix, run_power_iteration, PARAMS
import config

logging.basicConfig(level=logging.WARNING, format='%(message)s')

def main():
    with open(config.GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)

    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    print("Loading SMOOTHED NPZ...")
    loader_smooth = np.load(config.POPULARITY_NPZ, allow_pickle=True)
    matrix_smooth = csr_matrix((loader_smooth['matrix_data'], loader_smooth['matrix_indices'], loader_smooth['matrix_indptr']), shape=loader_smooth['matrix_shape'])
    times_smooth = list(loader_smooth['times'])
    pop_link_ids_smooth = list(loader_smooth['link_ids'])

    # Randomly select 49 time frames
    random.seed(42) # For reproducibility
    sample_times = random.sample(times_smooth, 49)

    # Define grid to search based on user suggestion (larger negative values)
    alpha_s_vals = [-2.0, -3.0, -4.0, -5.0]
    alpha_l_vals = [-2.0, -3.0, -4.0, -5.0]
    beta = 0.2

    print(f"Fine-tuning alpha_s and alpha_l on {len(sample_times)} random timeframes...")
    print("Focusing on minimizing Top-100 MRE...")
    print(f"{'alpha_s':>8} | {'alpha_l':>8} | {'Top-100 MRE':>12} | {'Overall MRE':>12}")
    print("-" * 50)

    best_mre = float('inf')
    best_params = None
    results = []

    for a_s in alpha_s_vals:
        for a_l in alpha_l_vals:
            # Update PARAMS dynamically
            PARAMS['alpha_s'] = a_s
            PARAMS['alpha_l'] = a_l
            PARAMS['beta'] = beta

            # Rebuild matrix for these params
            M_2N = build_two_phase_matrix(graph_data)

            top100_mre_sum = 0.0
            overall_mre_sum = 0.0

            for t_str in sample_times:
                t_idx = times_smooth.index(t_str)
                row = matrix_smooth.getrow(t_idx)

                E_N = np.zeros(N)
                for i, val in zip(row.indices, row.data):
                    lid = pop_link_ids_smooth[i]
                    if lid in lid_to_idx:
                        E_N[lid_to_idx[lid]] = float(val)

                s = np.sum(E_N)
                if s > 0:
                    E_N /= s
                else:
                    E_N = np.ones(N) / N

                E_2N = np.concatenate([E_N, np.zeros(N)])

                v_2N = run_power_iteration(M_2N, E_2N)
                v_final = v_2N[:N] + v_2N[N:]
                v_final /= np.sum(v_final)

                # Get top 100 links by ground truth
                top_100_indices = np.argsort(E_N)[::-1][:100]

                # Top 100 MRE
                t_top100 = E_N[top_100_indices]
                p_top100 = v_final[top_100_indices]
                mre_100 = np.mean(np.abs(t_top100 - p_top100) / (t_top100 + 1e-9))
                top100_mre_sum += mre_100

                # Overall MRE
                mre_all = np.mean(np.abs(E_N - v_final) / (E_N + 1e-9))
                overall_mre_sum += mre_all

            avg_top100_mre = top100_mre_sum / len(sample_times)
            avg_overall_mre = overall_mre_sum / len(sample_times)

            print(f"{a_s:8.1f} | {a_l:8.1f} | {avg_top100_mre:12.4f} | {avg_overall_mre:12.4f}")
            results.append({
                'alpha_s': a_s,
                'alpha_l': a_l,
                'top100_mre': avg_top100_mre,
                'overall_mre': avg_overall_mre
            })

            if avg_top100_mre < best_mre:
                best_mre = avg_top100_mre
                best_params = (a_s, a_l)

    print("-" * 50)
    print(f"Best Parameters for Top-100 MRE: alpha_s={best_params[0]}, alpha_l={best_params[1]} (MRE: {best_mre:.4f})")

    # Save results
    out_file = config.RESULTS_DIR / 'top100_tuning_results.txt'
    with open(out_file, 'w') as f:
        f.write(f"Best Parameters for Top-100 MRE: alpha_s={best_params[0]}, alpha_l={best_params[1]} (MRE: {best_mre:.4f})\n\n")
        f.write(f"{'alpha_s':>8} | {'alpha_l':>8} | {'Top-100 MRE':>12} | {'Overall MRE':>12}\n")
        f.write("-" * 50 + "\n")
        for r in results:
            f.write(f"{r['alpha_s']:8.1f} | {r['alpha_l']:8.1f} | {r['top100_mre']:12.4f} | {r['overall_mre']:12.4f}\n")
    print(f"Detailed output saved to {out_file}")

if __name__ == '__main__':
    main()
