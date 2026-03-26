"""
Multi-strategy search to maximize Top-100 improvement at damping=0.85.
Strategies:
  A) Fine tau search
  B) Beta tuning (varying phase-transition rate)
  C) Joint tau + beta
  D) Two-pass residual correction
  E) Top-K targeted boost
"""
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
DAMPING = 0.85
OVERALL_BUDGET = 0.198  # must stay at or below this


def eval_one(M_2N, E_2N_teleport, E_N_truth, N):
    """Run power iteration and return (top100_mre, overall_mre)."""
    v_2N = run_power_iteration(M_2N, E_2N_teleport)
    v_final = v_2N[:N] + v_2N[N:]
    v_final /= np.sum(v_final)

    top_idx = np.argsort(E_N_truth)[::-1][:100]
    mre_100 = np.mean(
        np.abs(E_N_truth[top_idx] - v_final[top_idx])
        / (E_N_truth[top_idx] + 1e-9)
    )
    mre_all = np.mean(
        np.abs(E_N_truth - v_final) / (E_N_truth + 1e-9)
    )
    return mre_100, mre_all, v_final


def run_strategy(name, M_2N, E_N_cache, N, make_teleport_fn):
    """Evaluate a strategy across all cached timeframes."""
    top_sum, all_sum = 0.0, 0.0
    extra = None
    for E_N in E_N_cache:
        E_tele = make_teleport_fn(E_N, M_2N, N)
        E_2N = np.concatenate([E_tele, np.zeros(N)])
        m100, mall, vf = eval_one(M_2N, E_2N, E_N, N)
        top_sum += m100
        all_sum += mall
    n = len(E_N_cache)
    return top_sum / n, all_sum / n


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

    PARAMS['damping'] = DAMPING
    PARAMS['alpha_s'] = 0.0
    PARAMS['alpha_l'] = 0.0

    all_results = []

    def record(strategy, params_str, top100, overall):
        feasible = overall <= OVERALL_BUDGET
        all_results.append({
            'strategy': strategy, 'params': params_str,
            'top100': top100, 'overall': overall, 'feasible': feasible
        })
        mark = 'YES' if feasible else 'no'
        print(f"  {params_str:<40} | {top100:8.4f} | {overall:8.4f} | {mark:>3}")

    # =========================================================================
    # Strategy A: Fine tau (beta=0.2 fixed)
    # =========================================================================
    print("\n=== Strategy A: Fine tau at d=0.85, beta=0.2 ===")
    print(f"  {'Params':<40} | {'Top100':>8} | {'Overall':>8} | OK?")
    print("  " + "-" * 70)

    PARAMS['beta'] = 0.2
    M_2N = build_two_phase_matrix(graph_data)

    for tau in [1.0, 1.01, 1.02, 1.03, 1.04, 1.05]:
        def make_tele(E_N, M, n, _tau=tau):
            return sharpen_teleportation(E_N, _tau)
        t100, tall = run_strategy(f"tau={tau}", M_2N, E_N_cache, N, make_tele)
        record("A:tau", f"tau={tau:.2f}", t100, tall)

    # =========================================================================
    # Strategy B: Beta tuning (tau=1.0)
    # =========================================================================
    print("\n=== Strategy B: Beta tuning at d=0.85, tau=1.0 ===")
    print(f"  {'Params':<40} | {'Top100':>8} | {'Overall':>8} | OK?")
    print("  " + "-" * 70)

    for beta in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        PARAMS['beta'] = beta
        M_2N_b = build_two_phase_matrix(graph_data)
        def make_tele(E_N, M, n):
            return E_N.copy()
        t100, tall = run_strategy(f"beta={beta}", M_2N_b, E_N_cache, N, make_tele)
        record("B:beta", f"beta={beta:.2f}", t100, tall)

    # =========================================================================
    # Strategy C: Joint tau + beta
    # =========================================================================
    print("\n=== Strategy C: Joint tau + beta at d=0.85 ===")
    print(f"  {'Params':<40} | {'Top100':>8} | {'Overall':>8} | OK?")
    print("  " + "-" * 70)

    for beta in [0.05, 0.10, 0.15]:
        PARAMS['beta'] = beta
        M_2N_c = build_two_phase_matrix(graph_data)
        for tau in [1.02, 1.05, 1.08, 1.10, 1.15, 1.20]:
            def make_tele(E_N, M, n, _tau=tau):
                return sharpen_teleportation(E_N, _tau)
            t100, tall = run_strategy(f"beta={beta},tau={tau}", M_2N_c, E_N_cache, N, make_tele)
            record("C:joint", f"beta={beta:.2f}, tau={tau:.2f}", t100, tall)

    # =========================================================================
    # Strategy D: Two-pass residual correction
    # =========================================================================
    print("\n=== Strategy D: Two-pass residual correction at d=0.85 ===")
    print(f"  {'Params':<40} | {'Top100':>8} | {'Overall':>8} | OK?")
    print("  " + "-" * 70)

    PARAMS['beta'] = 0.2
    M_2N = build_two_phase_matrix(graph_data)

    for gamma in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        top_sum, all_sum = 0.0, 0.0
        for E_N in E_N_cache:
            # Pass 1: standard run
            E_2N_1 = np.concatenate([E_N, np.zeros(N)])
            v_2N_1 = run_power_iteration(M_2N, E_2N_1)
            v1 = v_2N_1[:N] + v_2N_1[N:]
            v1 /= np.sum(v1)

            # Compute correction: boost underpredicted links
            ratio = E_N / (v1 + 1e-12)
            E_corrected = E_N * np.power(ratio, gamma)
            E_corrected /= np.sum(E_corrected)

            # Pass 2: run with corrected teleportation
            E_2N_2 = np.concatenate([E_corrected, np.zeros(N)])
            m100, mall, _ = eval_one(M_2N, E_2N_2, E_N, N)
            top_sum += m100
            all_sum += mall

        t100 = top_sum / len(E_N_cache)
        tall = all_sum / len(E_N_cache)
        record("D:2pass", f"gamma={gamma:.1f}", t100, tall)

    # =========================================================================
    # Strategy E: Top-K targeted boost
    # =========================================================================
    print("\n=== Strategy E: Top-K targeted boost at d=0.85 ===")
    print(f"  {'Params':<40} | {'Top100':>8} | {'Overall':>8} | OK?")
    print("  " + "-" * 70)

    for K in [100, 200, 500]:
        for boost in [1.5, 2.0, 3.0, 5.0, 10.0]:
            def make_tele(E_N, M, n, _K=K, _b=boost):
                E_b = E_N.copy()
                top_idx = np.argsort(E_N)[::-1][:_K]
                E_b[top_idx] *= _b
                E_b /= np.sum(E_b)
                return E_b
            t100, tall = run_strategy(f"K={K},boost={boost}", M_2N, E_N_cache, N, make_tele)
            record("E:topK", f"K={K}, boost={boost:.1f}", t100, tall)

    # =========================================================================
    # Strategy F: Two-pass + beta tuning combo
    # =========================================================================
    print("\n=== Strategy F: Two-pass + low beta ===")
    print(f"  {'Params':<40} | {'Top100':>8} | {'Overall':>8} | OK?")
    print("  " + "-" * 70)

    for beta in [0.05, 0.10]:
        PARAMS['beta'] = beta
        M_2N_f = build_two_phase_matrix(graph_data)
        for gamma in [0.2, 0.3, 0.5, 0.7, 1.0]:
            top_sum, all_sum = 0.0, 0.0
            for E_N in E_N_cache:
                E_2N_1 = np.concatenate([E_N, np.zeros(N)])
                v_2N_1 = run_power_iteration(M_2N_f, E_2N_1)
                v1 = v_2N_1[:N] + v_2N_1[N:]
                v1 /= np.sum(v1)

                ratio = E_N / (v1 + 1e-12)
                E_corrected = E_N * np.power(ratio, gamma)
                E_corrected /= np.sum(E_corrected)

                E_2N_2 = np.concatenate([E_corrected, np.zeros(N)])
                m100, mall, _ = eval_one(M_2N_f, E_2N_2, E_N, N)
                top_sum += m100
                all_sum += mall

            t100 = top_sum / len(E_N_cache)
            tall = all_sum / len(E_N_cache)
            record("F:2pass+beta", f"beta={beta:.2f}, gamma={gamma:.1f}", t100, tall)

    # =========================================================================
    # Summary
    # =========================================================================
    feasible = [r for r in all_results if r['feasible']]
    feasible.sort(key=lambda r: r['top100'])

    print("\n" + "=" * 80)
    print("TOP 10 FEASIBLE CONFIGURATIONS (Overall MRE <= 0.198)")
    print(f"  {'Strategy':<12} | {'Params':<40} | {'Top100':>8} | {'Overall':>8}")
    print("  " + "-" * 74)
    for r in feasible[:10]:
        print(f"  {r['strategy']:<12} | {r['params']:<40} | {r['top100']:8.4f} | {r['overall']:8.4f}")

    # Save
    out_file = base_dir / '../../results/strategy_tuning_results.txt'
    with open(out_file, 'w') as f:
        f.write("TOP FEASIBLE (overall <= 0.198):\n")
        for r in feasible[:10]:
            f.write(f"  {r['strategy']:<12} {r['params']:<40} "
                    f"top100={r['top100']:.4f} overall={r['overall']:.4f}\n")
        f.write(f"\nALL RESULTS:\n")
        for r in all_results:
            mark = "YES" if r['feasible'] else "no"
            f.write(f"  {r['strategy']:<12} {r['params']:<40} "
                    f"top100={r['top100']:.4f} overall={r['overall']:.4f} {mark}\n")
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
