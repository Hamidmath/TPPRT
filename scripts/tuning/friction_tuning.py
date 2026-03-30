import json
import logging
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
import random

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.friction import build_custom_matrix, get_eval_metrics
from core.pagerank import run_power_iteration
import config

logging.basicConfig(level=logging.WARNING)

def main():
    with open(config.GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)

    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    loader = np.load(config.POPULARITY_NPZ, allow_pickle=True)
    matrix = csr_matrix((loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']), shape=loader['matrix_shape'])
    times = list(loader['times'])
    pop_link_ids = list(loader['link_ids'])

    # We will use 10 random timeframes to keep tuning computationally reasonable
    random.seed(42)
    sample_times = random.sample(times, 10)

    alpha_s_vals = [0.0]
    alpha_l_vals = [-30.0]

    print(f"Fine-tuning Congestion Friction model on {len(sample_times)} random timeframes...")
    print(f"{'alpha_s':>8} | {'alpha_l':>8} | {'Top-100 MRE':>12} | {'Overall MRE':>12}")
    print("-" * 50)

    results = []

    for a_s in alpha_s_vals:
        for a_l in alpha_l_vals:
            conf = {
                'alpha_s': float(a_s),
                'alpha_l': float(a_l),
                'beta': 0.2,
                'self_loops': True,
                'friction_factor': 0.05
            }

            M_2N = build_custom_matrix(graph_data, conf)

            top_mre_sum = 0
            all_mre_sum = 0

            for t_str in sample_times:
                t_idx = times.index(t_str)
                row = matrix.getrow(t_idx)

                E_N = np.zeros(N)
                for i, val in zip(row.indices, row.data):
                    lid = pop_link_ids[i]
                    if lid in lid_to_idx:
                        E_N[lid_to_idx[lid]] = float(val)
                if E_N.sum() > 0:
                    E_N /= E_N.sum()
                E_2N = np.concatenate([E_N, np.zeros(N)])

                v_2N = run_power_iteration(M_2N, E_2N)
                v_final = v_2N[:N] + v_2N[N:]
                v_final /= np.sum(v_final)

                mre_top, mre_all = get_eval_metrics(E_N, v_final, top_k=100)
                top_mre_sum += mre_top
                all_mre_sum += mre_all

            avg_top = top_mre_sum / len(sample_times)
            avg_all = all_mre_sum / len(sample_times)

            print(f"{a_s:8.1f} | {a_l:8.1f} | {avg_top:12.4f} | {avg_all:12.4f}")
            results.append((avg_top, a_s, a_l))

    results.sort(key=lambda x: x[0])
    print("-" * 50)
    print(f"Best Top-100 Configuration: alpha_s={results[0][1]}, alpha_l={results[0][2]} (MRE: {results[0][0]:.4f})")

if __name__ == '__main__':
    main()
