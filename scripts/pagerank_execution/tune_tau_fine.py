import json
import logging
import random
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

from run_two_phase_pagerank import (
    build_two_phase_matrix, run_power_iteration,
    sharpen_teleportation, PARAMS
)

logging.basicConfig(level=logging.WARNING, format='%(message)s')

GRAPH_FILE = '../../data/city_graph_full.json'
POPULARITY_NPZ = '../../data/popularity_results_smoothed.npz'


def main():
    base_dir = Path(__file__).parent

    with open(base_dir / GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)

    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    loader = np.load(base_dir / POPULARITY_NPZ, allow_pickle=True)
    matrix = csr_matrix((
        loader['matrix_data'], loader['matrix_indices'],
        loader['matrix_indptr']
    ), shape=loader['matrix_shape'])
    times = list(loader['times'])
    pop_link_ids = list(loader['link_ids'])

    random.seed(42)
    sample_times = random.sample(times, 49)

    PARAMS['alpha_s'] = 0.0
    PARAMS['alpha_l'] = 0.0
    PARAMS['beta'] = 0.2
    M_2N = build_two_phase_matrix(graph_data)

    # Pre-cache E_N vectors
    E_N_cache = []
    for t_str in sample_times:
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
        E_N_cache.append(E_N)

    # Fine grid: lower damping + mild tau
    tau_vals = [1.0, 1.02, 1.05, 1.08, 1.1]
    damping_vals = [0.75, 0.78, 0.80, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87]

    print(f"Fine-tuning tau x damping on {len(sample_times)} timeframes")
    print(f"Grid: {len(tau_vals)} tau x {len(damping_vals)} damping = "
          f"{len(tau_vals) * len(damping_vals)} configurations")
    print(f"{'tau':>6} | {'damping':>8} | {'Top-100 MRE':>12} | "
          f"{'Overall MRE':>12} | {'Feasible':>8}")
    print("-" * 65)

    results = []
    best_feasible = None

    for tau in tau_vals:
        for damp in damping_vals:
            PARAMS['damping'] = damp
            top100_sum = 0.0
            overall_sum = 0.0

            for E_N in E_N_cache:
                E_sharp = sharpen_teleportation(E_N, tau)
                E_2N = np.concatenate([E_sharp, np.zeros(N)])

                v_2N = run_power_iteration(M_2N, E_2N)
                v_final = v_2N[:N] + v_2N[N:]
                v_final /= np.sum(v_final)

                top_idx = np.argsort(E_N)[::-1][:100]
                mre_100 = np.mean(
                    np.abs(E_N[top_idx] - v_final[top_idx])
                    / (E_N[top_idx] + 1e-9)
                )
                top100_sum += mre_100

                mre_all = np.mean(
                    np.abs(E_N - v_final) / (E_N + 1e-9)
                )
                overall_sum += mre_all

            avg_top100 = top100_sum / len(sample_times)
            avg_overall = overall_sum / len(sample_times)
            feasible = avg_overall <= 0.198

            print(f"{tau:6.2f} | {damp:8.2f} | {avg_top100:12.4f} | "
                  f"{avg_overall:12.4f} | {'YES' if feasible else 'no':>8}")

            results.append({
                'tau': tau, 'damping': damp,
                'top100_mre': avg_top100, 'overall_mre': avg_overall,
                'feasible': feasible
            })

            if feasible:
                if (best_feasible is None
                        or avg_top100 < best_feasible['top100_mre']):
                    best_feasible = results[-1]

    print("=" * 65)
    if best_feasible:
        print(f"BEST FEASIBLE: tau={best_feasible['tau']}, "
              f"damping={best_feasible['damping']}")
        print(f"  Top-100 MRE: {best_feasible['top100_mre']:.4f}")
        print(f"  Overall MRE: {best_feasible['overall_mre']:.4f}")

    # Save
    out_file = base_dir / '../../results/tau_fine_tuning_results.txt'
    with open(out_file, 'w') as f:
        if best_feasible:
            f.write(f"Best Feasible: tau={best_feasible['tau']}, "
                    f"damping={best_feasible['damping']}, "
                    f"top100={best_feasible['top100_mre']:.4f}, "
                    f"overall={best_feasible['overall_mre']:.4f}\n\n")
        f.write(f"{'tau':>6} | {'damping':>8} | {'Top-100 MRE':>12} | "
                f"{'Overall MRE':>12} | {'Feasible':>8}\n")
        f.write("-" * 65 + "\n")
        for r in results:
            f.write(f"{r['tau']:6.2f} | {r['damping']:8.2f} | "
                    f"{r['top100_mre']:12.4f} | {r['overall_mre']:12.4f} | "
                    f"{'YES' if r['feasible'] else 'no':>8}\n")
    print(f"Results saved to {out_file}")


if __name__ == '__main__':
    main()
