import json
import logging
import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from pathlib import Path
import time
import random

logging.basicConfig(level=logging.WARNING)

GRAPH_FILE = '../../data/city_graph_full.json'
POPULARITY_NPZ = '../../data/popularity_results_smoothed.npz'

def get_eval_metrics(truth, pred, top_k=100):
    top_k_idx = np.argsort(truth)[::-1][:top_k]
    t_top = truth[top_k_idx]
    p_top = pred[top_k_idx]
    mre_top = np.mean(np.abs(t_top - p_top) / (t_top + 1e-9))
    mre_all = np.mean(np.abs(truth - pred) / (truth + 1e-9))
    return mre_top, mre_all

def build_custom_matrix(graph_data, config):
    links = list(graph_data['links'].keys())
    N = len(links)
    id_to_idx = {lid: i for i, lid in enumerate(links)}
    adj = graph_data.get('adjacency', {})
    
    speeds = np.array([graph_data['links'][lid].get('speed', 11.17) for lid in links])
    lanes = np.array([graph_data['links'][lid].get('lanes', 1.0) for lid in links])
    lengths = np.array([graph_data['links'][lid].get('length', 100.0) for lid in links])
    
    mean_s = np.mean(speeds) if np.mean(speeds) > 0 else 1.0
    mean_l = np.mean(lanes) if np.mean(lanes) > 0 else 1.0
    
    def build_phase(is_up):
        a_s = config['alpha_s'] if is_up else -config['alpha_s']
        a_l = config['alpha_l'] if is_up else -config['alpha_l']
        
        row, col, data = [], [], []
        
        for i, lid in enumerate(links):
            out_lids = adj.get(lid, [])
            succ_idx = [id_to_idx[x] for x in out_lids if x in id_to_idx]
            
            weights = []
            for j in succ_idx:
                s_j = speeds[j] / mean_s
                l_j = lanes[j] / mean_l
                w = (s_j ** a_s) * (l_j ** a_l)
                weights.append(w)
            
            # Option 1: Self-Loops (Friction)
            self_w = 0.0
            if config.get('self_loops', True):
                travel_time = lengths[i] / max(speeds[i], 0.1)
                self_w = travel_time * config.get('friction_factor', 0.05)
            
            total_w = sum(weights) + self_w
            if total_w > 0:
                for j, w in zip(succ_idx, weights):
                    row.append(i)
                    col.append(j)
                    data.append(w / total_w)
                if self_w > 0:
                    row.append(i)
                    col.append(i)
                    data.append(self_w / total_w)
                    
        return csr_matrix((data, (row, col)), shape=(N, N))
        
    P_up = build_phase(True)
    P_down = build_phase(False)
    
    beta = config['beta']
    P_up = P_up.multiply(1.0 - beta)
    
    row_T = np.arange(N)
    col_T = np.arange(N)
    data_T = np.ones(N) * beta
    T_down = csr_matrix((data_T, (row_T, col_T)), shape=(N, N))
    
    zero_block = csr_matrix((N, N))
    top_block = hstack([P_up, T_down])
    bottom_block = hstack([zero_block, P_down])
    
    return vstack([top_block, bottom_block])

def run_power_iteration(M_2N, E_2N, damping=0.89, max_iters=100, tol=1e-6):
    v_2N = E_2N.copy()
    M_T = M_2N.transpose()
    for i in range(max_iters):
        v_next = damping * M_T.dot(v_2N) + (1 - damping) * E_2N
        S = np.sum(v_next)
        if S < 1.0:
            v_next += E_2N * (1.0 - S)
        diff = np.sum(np.abs(v_next - v_2N))
        v_2N = v_next
        if diff < tol:
            break
    return v_2N

def main():
    base_dir = Path(__file__).parent
    
    with open(base_dir / GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)
        
    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    
    loader = np.load(base_dir / POPULARITY_NPZ, allow_pickle=True)
    matrix = csr_matrix((loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']), shape=loader['matrix_shape'])
    times = list(loader['times'])
    pop_link_ids = list(loader['link_ids'])
    
    # We will use 10 random timeframes to keep tuning computationally reasonable
    random.seed(42)
    sample_times = random.sample(times, 10)
    
    alpha_s_vals = [0.0]
    alpha_l_vals = [-30.0]
    
    print(f"Fine-tuning Congestion Friction model on {len(sample_times)} random timeframes...")
    print(f"{'alpha_s':>8} | {'alpha_l':>8} | {'Top-100 MRE':>12} | {'Overall MRE':>12}")
    print("-" * 50)
    
    results = []
    
    for a_s in alpha_s_vals:
        for a_l in alpha_l_vals:
            conf = {
                'alpha_s': float(a_s), 
                'alpha_l': float(a_l), 
                'beta': 0.2,
                'self_loops': True, 
                'friction_factor': 0.05
            }
            
            M_2N = build_custom_matrix(graph_data, conf)
            
            top_mre_sum = 0
            all_mre_sum = 0
            
            for t_str in sample_times:
                t_idx = times.index(t_str)
                row = matrix.getrow(t_idx)
                
                E_N = np.zeros(N)
                for i, val in zip(row.indices, row.data):
                    lid = pop_link_ids[i]
                    if lid in lid_to_idx:
                        E_N[lid_to_idx[lid]] = float(val)
                if E_N.sum() > 0:
                    E_N /= E_N.sum()
                E_2N = np.concatenate([E_N, np.zeros(N)])
                
                v_2N = run_power_iteration(M_2N, E_2N)
                v_final = v_2N[:N] + v_2N[N:]
                v_final /= np.sum(v_final)
                
                mre_top, mre_all = get_eval_metrics(E_N, v_final, top_k=100)
                top_mre_sum += mre_top
                all_mre_sum += mre_all
                
            avg_top = top_mre_sum / len(sample_times)
            avg_all = all_mre_sum / len(sample_times)
            
            print(f"{a_s:8.1f} | {a_l:8.1f} | {avg_top:12.4f} | {avg_all:12.4f}")
            results.append((avg_top, a_s, a_l))

    results.sort(key=lambda x: x[0])
    print("-" * 50)
    print(f"Best Top-100 Configuration: alpha_s={results[0][1]}, alpha_l={results[0][2]} (MRE: {results[0][0]:.4f})")

if __name__ == '__main__':
    main()