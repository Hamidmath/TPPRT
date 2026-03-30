import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core import build_two_phase_matrix, run_power_iteration, PARAMS
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

    random.seed(42)
    sample_times = random.sample(times, 10)

    PARAMS['alpha_s'] = 0.0
    PARAMS['alpha_l'] = -30.0
    PARAMS['beta'] = 0.2

    M_2N = build_two_phase_matrix(graph_data)

    top100_mre_sum = 0.0
    overall_mre_sum = 0.0

    for t_str in sample_times:
        t_idx = times.index(t_str)
        row = matrix.getrow(t_idx)
        E_N = np.zeros(N)
        for i, val in zip(row.indices, row.data):
            lid = pop_link_ids[i]
            if lid in lid_to_idx:
                E_N[lid_to_idx[lid]] = float(val)
        s = np.sum(E_N)
        if s > 0: E_N /= s
        else: E_N = np.ones(N) / N

        E_2N = np.concatenate([E_N, np.zeros(N)])
        v_2N = run_power_iteration(M_2N, E_2N)
        v_final = v_2N[:N] + v_2N[N:]
        v_final /= np.sum(v_final)

        top_100_indices = np.argsort(E_N)[::-1][:100]
        t_top100 = E_N[top_100_indices]
        p_top100 = v_final[top_100_indices]

        mre_100 = np.mean(np.abs(t_top100 - p_top100) / (t_top100 + 1e-9))
        top100_mre_sum += mre_100
        mre_all = np.mean(np.abs(E_N - v_final) / (E_N + 1e-9))
        overall_mre_sum += mre_all

    print("Baseline Model (No Friction)")
    print("alpha_s=0.0, alpha_l=-30.0")
    print(f"Top-100 MRE: {top100_mre_sum/10:.4f}")
    print(f"Overall MRE: {overall_mre_sum/10:.4f}")

if __name__ == '__main__':
    main()
