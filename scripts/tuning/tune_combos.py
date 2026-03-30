"""
Combination strategies at damping=0.85, exploiting high-beta overall budget.
"""
import json
import logging
import random
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core import (
    build_two_phase_matrix, run_power_iteration,
    sharpen_teleportation, PARAMS
)
import config

logging.basicConfig(level=logging.WARNING, format='%(message)s')

DAMPING = 0.85
OVERALL_BUDGET = 0.198


def eval_one(M_2N, E_2N, E_N_truth, N):
    v_2N = run_power_iteration(M_2N, E_2N)
    v_final = v_2N[:N] + v_2N[N:]
    v_final /= np.sum(v_final)
    top_idx = np.argsort(E_N_truth)[::-1][:100]
    mre_100 = np.mean(np.abs(E_N_truth[top_idx] - v_final[top_idx]) / (E_N_truth[top_idx] + 1e-9))
    mre_all = np.mean(np.abs(E_N_truth - v_final) / (E_N_truth + 1e-9))
    return mre_100, mre_all


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
        s = np.sum(E_N)
        if s > 0: E_N /= s
        else: E_N = np.ones(N) / N
        E_N_cache.append(E_N)

    PARAMS['damping'] = DAMPING
    PARAMS['alpha_s'] = 0.0
    PARAMS['alpha_l'] = 0.0

    all_results = []
    def record(strat, params_str, t100, tall):
        f = tall <= OVERALL_BUDGET
        all_results.append({'s': strat, 'p': params_str, 't100': t100, 'all': tall, 'f': f})
        print(f"  {params_str:<50} | {t100:8.4f} | {tall:8.4f} | {'YES' if f else 'no':>3}")

    # ==========================================================
    # G: high beta + two-pass
    # ==========================================================
    print("\n=== G: High beta + two-pass ===")
    print(f"  {'Params':<50} | {'Top100':>8} | {'Overall':>8} | OK?")
    print("  " + "-" * 76)
    for beta in [0.30, 0.40, 0.50, 0.60, 0.70]:
        PARAMS['beta'] = beta
        M = build_two_phase_matrix(graph_data)
        for gamma in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
            t_sum, a_sum = 0.0, 0.0
            for E_N in E_N_cache:
                E_2N_1 = np.concatenate([E_N, np.zeros(N)])
                v1_2N = run_power_iteration(M, E_2N_1)
                v1 = v1_2N[:N] + v1_2N[N:]; v1 /= v1.sum()
                ratio = E_N / (v1 + 1e-12)
                E_c = E_N * np.power(ratio, gamma); E_c /= E_c.sum()
                E_2N_2 = np.concatenate([E_c, np.zeros(N)])
                m100, mall = eval_one(M, E_2N_2, E_N, N)
                t_sum += m100; a_sum += mall
            record("G", f"beta={beta:.2f}, 2pass gamma={gamma:.1f}", t_sum/49, a_sum/49)

    # ==========================================================
    # H: high beta + top-K boost
    # ==========================================================
    print("\n=== H: High beta + top-K boost ===")
    print(f"  {'Params':<50} | {'Top100':>8} | {'Overall':>8} | OK?")
    print("  " + "-" * 76)
    for beta in [0.30, 0.40, 0.50, 0.60, 0.70]:
        PARAMS['beta'] = beta
        M = build_two_phase_matrix(graph_data)
        for boost in [1.2, 1.5, 2.0, 3.0, 5.0]:
            t_sum, a_sum = 0.0, 0.0
            for E_N in E_N_cache:
                E_b = E_N.copy()
                top_idx = np.argsort(E_N)[::-1][:100]
                E_b[top_idx] *= boost; E_b /= E_b.sum()
                E_2N = np.concatenate([E_b, np.zeros(N)])
                m100, mall = eval_one(M, E_2N, E_N, N)
                t_sum += m100; a_sum += mall
            record("H", f"beta={beta:.2f}, K=100 boost={boost:.1f}", t_sum/49, a_sum/49)

    # ==========================================================
    # I: high beta + top-K boost + two-pass
    # ==========================================================
    print("\n=== I: High beta + two-pass + top-K boost ===")
    print(f"  {'Params':<50} | {'Top100':>8} | {'Overall':>8} | OK?")
    print("  " + "-" * 76)
    for beta in [0.40, 0.50, 0.60]:
        PARAMS['beta'] = beta
        M = build_two_phase_matrix(graph_data)
        for gamma in [0.2, 0.3, 0.5]:
            for boost in [1.5, 2.0, 3.0]:
                t_sum, a_sum = 0.0, 0.0
                for E_N in E_N_cache:
                    # Pass 1
                    E_2N_1 = np.concatenate([E_N, np.zeros(N)])
                    v1_2N = run_power_iteration(M, E_2N_1)
                    v1 = v1_2N[:N] + v1_2N[N:]; v1 /= v1.sum()
                    # Residual correction
                    ratio = E_N / (v1 + 1e-12)
                    E_c = E_N * np.power(ratio, gamma); E_c /= E_c.sum()
                    # Top-K boost on corrected vector
                    top_idx = np.argsort(E_N)[::-1][:100]
                    E_c[top_idx] *= boost; E_c /= E_c.sum()
                    # Pass 2
                    E_2N_2 = np.concatenate([E_c, np.zeros(N)])
                    m100, mall = eval_one(M, E_2N_2, E_N, N)
                    t_sum += m100; a_sum += mall
                record("I", f"b={beta:.2f}, g={gamma:.1f}, boost={boost:.1f}", t_sum/49, a_sum/49)

    # ==========================================================
    # J: Multi-pass (3 passes) at high beta
    # ==========================================================
    print("\n=== J: 3-pass residual at high beta ===")
    print(f"  {'Params':<50} | {'Top100':>8} | {'Overall':>8} | OK?")
    print("  " + "-" * 76)
    for beta in [0.40, 0.50, 0.60]:
        PARAMS['beta'] = beta
        M = build_two_phase_matrix(graph_data)
        for gamma in [0.1, 0.15, 0.2, 0.25, 0.3]:
            t_sum, a_sum = 0.0, 0.0
            for E_N in E_N_cache:
                E_tele = E_N.copy()
                for _pass in range(3):
                    E_2N = np.concatenate([E_tele, np.zeros(N)])
                    v_2N = run_power_iteration(M, E_2N)
                    v = v_2N[:N] + v_2N[N:]; v /= v.sum()
                    ratio = E_N / (v + 1e-12)
                    E_tele = E_N * np.power(ratio, gamma); E_tele /= E_tele.sum()
                E_2N_f = np.concatenate([E_tele, np.zeros(N)])
                m100, mall = eval_one(M, E_2N_f, E_N, N)
                t_sum += m100; a_sum += mall
            record("J", f"beta={beta:.2f}, 3pass gamma={gamma:.2f}", t_sum/49, a_sum/49)

    # ==========================================================
    # Summary
    # ==========================================================
    feasible = [r for r in all_results if r['f']]
    feasible.sort(key=lambda r: r['t100'])
    print("\n" + "=" * 90)
    print("TOP 15 FEASIBLE (Overall <= 0.198)")
    print(f"  {'Strategy':<6} | {'Params':<50} | {'Top100':>8} | {'Overall':>8}")
    print("  " + "-" * 80)
    for r in feasible[:15]:
        print(f"  {r['s']:<6} | {r['p']:<50} | {r['t100']:8.4f} | {r['all']:8.4f}")

    out_file = config.RESULTS_DIR / 'combo_tuning_results.txt'
    with open(out_file, 'w') as f:
        f.write("TOP FEASIBLE (overall <= 0.198):\n")
        for r in feasible[:15]:
            f.write(f"  {r['s']:<6} {r['p']:<50} top100={r['t100']:.4f} overall={r['all']:.4f}\n")
        f.write(f"\nALL RESULTS:\n")
        for r in all_results:
            f.write(f"  {r['s']:<6} {r['p']:<50} top100={r['t100']:.4f} overall={r['all']:.4f} {'YES' if r['f'] else 'no'}\n")
    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    main()
