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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    with open(config.GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)

    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    # Build the matrix once (topology and physics do not change across timeframes)
    M_2N = build_two_phase_matrix(graph_data)

    # Load NPZ once to get times and data
    logger.info("Loading popularity matrix...")
    loader = np.load(config.POPULARITY_NPZ, allow_pickle=True)
    matrix = csr_matrix((loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']), shape=loader['matrix_shape'])
    times = list(loader['times'])
    pop_link_ids = list(loader['link_ids'])

    # Pick 49 random times
    sample_times = random.sample(times, 49)

    results = []
    total_mre = 0.0

    logger.info(f"Evaluating {len(sample_times)} random timeframes...")

    for idx, t_str in enumerate(sample_times):
        t_idx = times.index(t_str)
        row = matrix.getrow(t_idx)

        # Build teleportation vector (ground truth)
        E_N = np.zeros(N)
        for i, val in zip(row.indices, row.data):
            lid = pop_link_ids[i]
            if lid in lid_to_idx:
                E_N[lid_to_idx[lid]] = float(val)

        s = np.sum(E_N)
        if s > 0:
            E_N = E_N / s
        else:
            E_N = np.ones(N) / N

        E_2N = np.concatenate([E_N, np.zeros(N)])

        # We don't want the solver to log 100 lines per timeframe, so we temporarily elevate log level
        logger.setLevel(logging.WARNING)
        v_2N = run_power_iteration(M_2N, E_2N)
        logger.setLevel(logging.INFO)

        v_final = v_2N[:N] + v_2N[N:]
        v_final = v_final / np.sum(v_final)

        mre_sum = 0
        for i in range(N):
            mre_sum += abs(E_N[i] - v_final[i]) / (E_N[i] + 1e-9)
        mre = mre_sum / N

        logger.info(f"[{idx+1}/49] Time: {t_str} | MRE: {mre:.4f}")
        results.append((t_str, mre))
        total_mre += mre

    avg_mre = total_mre / len(sample_times)
    logger.info(f"Finished. Average MRE: {avg_mre:.4f}")

    # Save to text file
    out_path = config.RESULTS_DIR / 'random_49_mre_results.txt'
    with open(out_path, 'w') as f:
        f.write(f"Average MRE across {len(sample_times)} random timeframes: {avg_mre:.6f}\n\n")
        f.write("Timeframe\tMRE\n")
        for t_str, mre in results:
            f.write(f"{t_str}\t{mre:.6f}\n")

    logger.info(f"Results saved to {out_path}")

if __name__ == '__main__':
    main()
