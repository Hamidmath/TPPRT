"""Grid search over mu (self-loop scaling), damping, and beta."""
import json, logging, random
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

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

    loader = np.load(config.POPULARITY_NPZ, allow_pickle=True)
    matrix = csr_matrix((loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']), shape=loader['matrix_shape'])
    times = list(loader['times'])
    pop_link_ids = list(loader['link_ids'])

    random.seed(42)
    sample_times = random.sample(times, 49)

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
        if s > 0: E_N /= s
        else: E_N = np.ones(N) / N
        E_N_cache.append(E_N)

    PARAMS['alpha_s'] = 0.0
    PARAMS['alpha_l'] = 0.0
    PARAMS['tau'] = 1.0
    PARAMS['top_k_boost'] = 1.0

    mu_vals = [0, 1, 2, 3, 5, 8, 10, 12, 15, 20]
    damping_vals = [0.80, 0.85, 0.90]
    beta_vals = [0.0, 0.2, 0.5, 0.9]

    print(f"Tuning mu x damping x beta on {len(sample_times)} timeframes")
    total = len(mu_vals) * len(damping_vals) * len(beta_vals)
    print(f"Grid: {len(mu_vals)} mu x {len(damping_vals)} d x {len(beta_vals)} beta = {total} configs")
    print(f"{'mu':>4} | {'d':>5} | {'beta':>5} | {'Top-100':>8} | {'Overall':>8}")
    print("-" * 45)

    results = []
    for mu in mu_vals:
        for beta in beta_vals:
            PARAMS['mu'] = mu
            PARAMS['beta'] = beta
            M_2N = build_two_phase_matrix(graph_data)
            for damp in damping_vals:
                PARAMS['damping'] = damp
                t_sum, a_sum = 0.0, 0.0
                for E_N in E_N_cache:
                    E_2N = np.concatenate([E_N, np.zeros(N)])
                    v_2N = run_power_iteration(M_2N, E_2N)
                    v = v_2N[:N] + v_2N[N:]
                    v /= v.sum()
                    top_idx = np.argsort(E_N)[::-1][:100]
                    t_sum += np.mean(np.abs(E_N[top_idx] - v[top_idx]) / (E_N[top_idx] + 1e-9))
                    a_sum += np.mean(np.abs(E_N - v) / (E_N + 1e-9))
                t100 = t_sum / 49
                tall = a_sum / 49
                results.append({'mu': mu, 'd': damp, 'beta': beta, 't100': t100, 'all': tall})
                print(f"{mu:4} | {damp:5.2f} | {beta:5.2f} | {t100:8.4f} | {tall:8.4f}")

    # Sort by top-100 (all configs should beat 0.198 overall)
    results.sort(key=lambda r: r['t100'])
    print("\n" + "=" * 60)
    print("TOP 10 CONFIGURATIONS:")
    print(f"{'mu':>4} | {'d':>5} | {'beta':>5} | {'Top-100':>8} | {'Overall':>8}")
    print("-" * 45)
    for r in results[:10]:
        print(f"{r['mu']:4} | {r['d']:5.2f} | {r['beta']:5.2f} | {r['t100']:8.4f} | {r['all']:8.4f}")

    out_file = config.RESULTS_DIR / 'mu_tuning_results.txt'
    with open(out_file, 'w') as f:
        f.write("TOP 10:\n")
        for r in results[:10]:
            f.write(f"  mu={r['mu']}, d={r['d']}, beta={r['beta']}: top100={r['t100']:.4f}, overall={r['all']:.4f}\n")
        f.write(f"\nALL:\n")
        for r in sorted(results, key=lambda r: (r['mu'], r['beta'], r['d'])):
            f.write(f"  mu={r['mu']:2}, d={r['d']:.2f}, beta={r['beta']:.2f}: top100={r['t100']:.4f}, overall={r['all']:.4f}\n")
    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    main()
