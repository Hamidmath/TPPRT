"""Final fine-tuning around the best configs at damping=0.85."""
import json, logging, random
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from run_two_phase_pagerank import build_two_phase_matrix, run_power_iteration, PARAMS

logging.basicConfig(level=logging.WARNING, format='%(message)s')

def main():
    base_dir = Path(__file__).parent
    with open(base_dir / '../../data/city_graph_full.json', 'r') as f:
        graph_data = json.load(f)
    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    loader = np.load(base_dir / '../../data/popularity_results_smoothed.npz', allow_pickle=True)
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

    PARAMS['damping'] = 0.85
    PARAMS['alpha_s'] = 0.0
    PARAMS['alpha_l'] = 0.0

    print(f"{'beta':>6} | {'boost':>6} | {'Top-100':>8} | {'Overall':>8} | OK?")
    print("-" * 50)

    results = []
    for beta in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        PARAMS['beta'] = beta
        M = build_two_phase_matrix(graph_data)
        for boost in [1.0, 1.3, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0]:
            t_sum, a_sum = 0.0, 0.0
            for E_N in E_N_cache:
                E_b = E_N.copy()
                top_idx = np.argsort(E_N)[::-1][:100]
                E_b[top_idx] *= boost
                E_b /= E_b.sum()
                E_2N = np.concatenate([E_b, np.zeros(N)])
                v_2N = run_power_iteration(M, E_2N)
                v = v_2N[:N] + v_2N[N:]
                v /= v.sum()
                m100 = np.mean(np.abs(E_N[top_idx] - v[top_idx]) / (E_N[top_idx] + 1e-9))
                mall = np.mean(np.abs(E_N - v) / (E_N + 1e-9))
                t_sum += m100; a_sum += mall
            t100 = t_sum / 49; tall = a_sum / 49
            ok = tall <= 0.198
            results.append((beta, boost, t100, tall, ok))
            print(f"{beta:6.2f} | {boost:6.1f} | {t100:8.4f} | {tall:8.4f} | {'YES' if ok else 'no':>3}")

    # Also test two-pass at high beta with fine gamma
    print("\n--- Two-pass at high beta ---")
    print(f"{'beta':>6} | {'gamma':>6} | {'Top-100':>8} | {'Overall':>8} | OK?")
    print("-" * 50)
    for beta in [0.60, 0.70, 0.80, 0.90]:
        PARAMS['beta'] = beta
        M = build_two_phase_matrix(graph_data)
        for gamma in [0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
            t_sum, a_sum = 0.0, 0.0
            for E_N in E_N_cache:
                E_2N_1 = np.concatenate([E_N, np.zeros(N)])
                v1_2N = run_power_iteration(M, E_2N_1)
                v1 = v1_2N[:N] + v1_2N[N:]; v1 /= v1.sum()
                ratio = E_N / (v1 + 1e-12)
                E_c = E_N * np.power(ratio, gamma); E_c /= E_c.sum()
                E_2N_2 = np.concatenate([E_c, np.zeros(N)])
                v2_2N = run_power_iteration(M, E_2N_2)
                v = v2_2N[:N] + v2_2N[N:]; v /= v.sum()
                top_idx = np.argsort(E_N)[::-1][:100]
                m100 = np.mean(np.abs(E_N[top_idx] - v[top_idx]) / (E_N[top_idx] + 1e-9))
                mall = np.mean(np.abs(E_N - v) / (E_N + 1e-9))
                t_sum += m100; a_sum += mall
            t100 = t_sum / 49; tall = a_sum / 49
            ok = tall <= 0.198
            results.append((beta, gamma, t100, tall, ok))
            print(f"{beta:6.2f} | {gamma:6.2f} | {t100:8.4f} | {tall:8.4f} | {'YES' if ok else 'no':>3}")

    # Also: two-pass + boost combo at high beta
    print("\n--- Two-pass + boost at high beta ---")
    print(f"{'params':<40} | {'Top-100':>8} | {'Overall':>8} | OK?")
    print("-" * 65)
    for beta in [0.60, 0.70, 0.80]:
        PARAMS['beta'] = beta
        M = build_two_phase_matrix(graph_data)
        for gamma in [0.2, 0.3]:
            for boost in [1.3, 1.5, 2.0]:
                t_sum, a_sum = 0.0, 0.0
                for E_N in E_N_cache:
                    E_2N_1 = np.concatenate([E_N, np.zeros(N)])
                    v1_2N = run_power_iteration(M, E_2N_1)
                    v1 = v1_2N[:N] + v1_2N[N:]; v1 /= v1.sum()
                    ratio = E_N / (v1 + 1e-12)
                    E_c = E_N * np.power(ratio, gamma); E_c /= E_c.sum()
                    top_idx = np.argsort(E_N)[::-1][:100]
                    E_c[top_idx] *= boost; E_c /= E_c.sum()
                    E_2N_2 = np.concatenate([E_c, np.zeros(N)])
                    v2_2N = run_power_iteration(M, E_2N_2)
                    v = v2_2N[:N] + v2_2N[N:]; v /= v.sum()
                    m100 = np.mean(np.abs(E_N[top_idx] - v[top_idx]) / (E_N[top_idx] + 1e-9))
                    mall = np.mean(np.abs(E_N - v) / (E_N + 1e-9))
                    t_sum += m100; a_sum += mall
                t100 = t_sum / 49; tall = a_sum / 49
                ok = tall <= 0.198
                pstr = f"b={beta:.2f}, g={gamma:.1f}, boost={boost:.1f}"
                results.append((pstr, 0, t100, tall, ok))
                print(f"{pstr:<40} | {t100:8.4f} | {tall:8.4f} | {'YES' if ok else 'no':>3}")

    # Final summary
    feas = [(r[2], r) for r in results if r[4]]
    feas.sort()
    print("\n" + "=" * 70)
    print("TOP 10 FEASIBLE:")
    for _, r in feas[:10]:
        print(f"  {str(r[0]):>6} | {str(r[1]):>6} | top100={r[2]:.4f} | overall={r[3]:.4f}")

    out_file = base_dir / '../../results/final_tuning_results.txt'
    with open(out_file, 'w') as f:
        f.write("TOP FEASIBLE:\n")
        for _, r in feas[:10]:
            f.write(f"  {r[0]} | {r[1]} | top100={r[2]:.4f} | overall={r[3]:.4f}\n")
    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    main()
