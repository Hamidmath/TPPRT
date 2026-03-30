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

    # We use the defaults from run_two_phase_pagerank: alpha_s=0.0, alpha_l=0.0
    M_2N = build_two_phase_matrix(graph_data)

    loader = np.load(config.POPULARITY_NPZ, allow_pickle=True)
    matrix = csr_matrix((loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']), shape=loader['matrix_shape'])
    times = list(loader['times'])
    pop_link_ids = list(loader['link_ids'])

    random.seed(42)
    sample_times = random.sample(times, 49)

    overall_mre_sum = 0.0

    print(f"Running Two-Phase PageRank (alpha_s={PARAMS['alpha_s']}, alpha_l={PARAMS['alpha_l']}) on {len(sample_times)} timeframes...")

    for idx, t_str in enumerate(sample_times):
        t_idx = times.index(t_str)
        row = matrix.getrow(t_idx)

        E_N = np.zeros(N)
        for i, val in zip(row.indices, row.data):
            lid = pop_link_ids[i]
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

        mre_all = np.mean(np.abs(E_N - v_final) / (E_N + 1e-9))
        overall_mre_sum += mre_all

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(sample_times)}...")

    avg_overall = overall_mre_sum / len(sample_times)

    print("-" * 60)
    print(f"Average Overall Network MRE: {avg_overall:.4f}")

if __name__ == '__main__':
    main()
