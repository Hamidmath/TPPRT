import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

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

    PARAMS['alpha_s'] = 0.0
    PARAMS['alpha_l'] = -15.0
    M_2N = build_two_phase_matrix(graph_data)

    loader = np.load(config.POPULARITY_NPZ, allow_pickle=True)
    matrix = csr_matrix((loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']), shape=loader['matrix_shape'])
    times = list(loader['times'])
    pop_link_ids = list(loader['link_ids'])

    target_time = '2018-09-08 08:00:00'
    if target_time not in times:
        print(f"Time {target_time} not found.")
        return

    t_idx = times.index(target_time)
    row = matrix.getrow(t_idx)

    E_N = np.zeros(N)
    for i, val in zip(row.indices, row.data):
        lid = pop_link_ids[i]
        if lid in lid_to_idx:
            E_N[lid_to_idx[lid]] = float(val)

    if E_N.sum() > 0:
        E_N /= E_N.sum()

    E_2N = np.concatenate([E_N, np.zeros(N)])

    # Run PageRank
    v_2N = run_power_iteration(M_2N, E_2N)
    v_final = v_2N[:N] + v_2N[N:]
    v_final /= np.sum(v_final)

    # Get top 100 links by ground truth
    top_100_indices = np.argsort(E_N)[::-1][:100]

    print(f"--- Top 100 Links Comparison for {target_time} ---")
    print(f"{'Rank':<5} | {'Link ID':<10} | {'Truth Prob':<12} | {'Pred Prob':<12} | {'Rel Error':<10}")
    print("-" * 60)

    mre_sum = 0.0
    for rank, idx in enumerate(top_100_indices):
        t_val = E_N[idx]
        p_val = v_final[idx]
        rel_error = abs(t_val - p_val) / (t_val + 1e-9)
        mre_sum += rel_error

        # Print first 20 to avoid overwhelming the console
        if rank < 20:
            print(f"{rank+1:<5} | {links[idx]:<10} | {t_val:.8f}   | {p_val:.8f}   | {rel_error:.4f}")

    top_100_mre = mre_sum / 100.0
    overall_mre = np.mean(np.abs(E_N - v_final) / (E_N + 1e-9))

    print("-" * 60)
    print(f"Top 100 Links MRE:   {top_100_mre:.4f}")
    print(f"Overall Network MRE: {overall_mre:.4f}")

if __name__ == '__main__':
    main()
