import json
import logging
import random
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack

# We reuse build_custom_matrix from friction_tuning to build the friction model,
# and build_two_phase_matrix from run_two_phase_pagerank to build the baseline model.
from friction_tuning import build_custom_matrix, get_eval_metrics
from run_two_phase_pagerank import build_two_phase_matrix, run_power_iteration, PARAMS

logging.basicConfig(level=logging.WARNING, format='%(message)s')

GRAPH_FILE = '../../data/city_graph_full.json'
POPULARITY_NPZ = '../../data/popularity_results_smoothed.npz'

def main():
    base_dir = Path(__file__).parent
    
    print("Loading Graph...")
    with open(base_dir / GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)
        
    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    
    print("Loading NPZ...")
    loader = np.load(base_dir / POPULARITY_NPZ, allow_pickle=True)
    matrix = csr_matrix((loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']), shape=loader['matrix_shape'])
    times = list(loader['times'])
    pop_link_ids = list(loader['link_ids'])
    
    # 49 random timeframes
    random.seed(42)
    sample_times = random.sample(times, 49)
    
    # 1. Baseline Model (alpha_s=0.0, alpha_l=0.0, No Friction)
    PARAMS['alpha_s'] = 0.0
    PARAMS['alpha_l'] = 0.0
    PARAMS['beta'] = 0.2
    M_2N_base = build_two_phase_matrix(graph_data)
    
    # 2. Friction Model (alpha_s=0.0, alpha_l=-30.0, With Friction factor 0.05)
    friction_conf = {
        'alpha_s': 0.0, 
        'alpha_l': -30.0, 
        'beta': 0.2,
        'self_loops': True, 
        'friction_factor': 0.05
    }
    M_2N_fric = build_custom_matrix(graph_data, friction_conf)
    
    base_top_sum, base_all_sum = 0.0, 0.0
    fric_top_sum, fric_all_sum = 0.0, 0.0
    
    print(f"\nEvaluating Baseline vs Friction on {len(sample_times)} random timeframes...")
    
    for idx, t_str in enumerate(sample_times):
        t_idx = times.index(t_str)
        row = matrix.getrow(t_idx)
        
        E_N = np.zeros(N)
        for i, val in zip(row.indices, row.data):
            lid = pop_link_ids[i]
            if lid in lid_to_idx:
                E_N[lid_to_idx[lid]] = float(val)
                
        if E_N.sum() > 0:
            E_N /= E_N.sum()
        else:
            E_N = np.ones(N) / N
            
        E_2N = np.concatenate([E_N, np.zeros(N)])
        
        # Base Eval
        v_2N_base = run_power_iteration(M_2N_base, E_2N)
        v_final_base = v_2N_base[:N] + v_2N_base[N:]
        v_final_base /= np.sum(v_final_base)
        m_top_b, m_all_b = get_eval_metrics(E_N, v_final_base, top_k=100)
        base_top_sum += m_top_b
        base_all_sum += m_all_b
        
        # Friction Eval
        v_2N_fric = run_power_iteration(M_2N_fric, E_2N)
        v_final_fric = v_2N_fric[:N] + v_2N_fric[N:]
        v_final_fric /= np.sum(v_final_fric)
        m_top_f, m_all_f = get_eval_metrics(E_N, v_final_fric, top_k=100)
        fric_top_sum += m_top_f
        fric_all_sum += m_all_f
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(sample_times)}...")

    avg_base_top = base_top_sum / len(sample_times)
    avg_base_all = base_all_sum / len(sample_times)
    avg_fric_top = fric_top_sum / len(sample_times)
    avg_fric_all = fric_all_sum / len(sample_times)
    
    print("\n" + "="*65)
    print(f"{'Model':<30} | {'Top-100 MRE':<12} | {'Overall MRE':<12}")
    print("-" * 65)
    print(f"{'Baseline (αs=0, αl=0, NO fric)':<30} | {avg_base_top:<12.4f} | {avg_base_all:<12.4f}")
    print(f"{'Friction (αs=0, αl=-30, +fric)':<30} | {avg_fric_top:<12.4f} | {avg_fric_all:<12.4f}")
    print("=" * 65 + "\n")

if __name__ == '__main__':
    main()
