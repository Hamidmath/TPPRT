import numpy as np
import scipy.sparse as sparse
import os
import sys
import json
import random
from collections import defaultdict
from datetime import datetime
from scipy.stats import pearsonr
import logging

# Add scripts directory to path to import the standalone models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/pagerank_execution')))
from run_two_phase_pagerank import build_two_phase_matrix, PARAMS

logging.basicConfig(level=logging.WARNING)

def build_directed_adjacency(city_graph, link_ids):
    n = len(link_ids)
    id_to_idx = {lid: i for i, lid in enumerate(link_ids)}
    row_ind, col_ind, data = [], [], []
    adj = city_graph.get('adjacency', {})
    for lid, out_links in adj.items():
        if lid not in id_to_idx: continue
        i = id_to_idx[lid]
        valid_outs = [nid for nid in out_links if nid in id_to_idx]
        if not valid_outs: continue
        for out_lid in valid_outs:
            row_ind.append(i)
            col_ind.append(id_to_idx[out_lid])
            data.append(1.0 / len(valid_outs))
    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def load_popularity_matrix(filepath):
    loader = np.load(filepath, allow_pickle=True)
    matrix = sparse.csr_matrix(
        (loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']), 
        shape=loader['matrix_shape']
    )
    return {
        'matrix': matrix,
        'times': list(loader['times']),
        'link_ids': list(loader['link_ids'])
    }

def main():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, '../data')
    
    print("Loading data...")
    raw_data = load_popularity_matrix(os.path.join(data_dir, 'popularity_results.npz'))
    city_graph = load_json(os.path.join(data_dir, 'city_graph_full.json'))
    
    links_raw = raw_data['link_ids']
    N_raw = len(links_raw)
    
    links_graph = list(city_graph['links'].keys())
    N_graph = len(links_graph)
    
    # Map from raw index to graph index
    graph_idx_map = {lid: i for i, lid in enumerate(links_graph)}
    raw_to_graph_idx = []
    valid_raw_idx = []
    
    for i, lid in enumerate(links_raw):
        if lid in graph_idx_map:
            raw_to_graph_idx.append(graph_idx_map[lid])
            valid_raw_idx.append(i)
            
    raw_to_graph_idx = np.array(raw_to_graph_idx)
    valid_raw_idx = np.array(valid_raw_idx)
    
    print("Building adjacency...")
    P_dir = build_directed_adjacency(city_graph, links_raw)
    
    print("Building Two-Phase Matrix...")
    PARAMS['alpha_s'] = 0.0
    PARAMS['alpha_l'] = 0.0
    PARAMS['beta'] = 0.2
    M_2N = build_two_phase_matrix(city_graph)
    M_2N_T = M_2N.transpose().tocsr()
    
    # Identify splits
    times = raw_data['times']
    timeframe_counts = defaultdict(list)
    for t_idx, t_str in enumerate(times):
        dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
        timeframe_counts[(dt.weekday(), dt.strftime("%H:%M:%S"))].append(t_idx)
    valid_splits = {k: v for k, v in timeframe_counts.items() if len(v) >= 2}
    
    random.seed(42)
    # Pick up to 49 random splits for faster execution in this demo (original used 490)
    num_splits = min(49, len(valid_splits))
    selected_keys = random.sample(list(valid_splits.keys()), num_splits)
    print(f"Selected {num_splits} timeframe splits.")
    
    gammas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.26, 0.30, 0.35, 0.40, 0.50, 0.60]
    num_gammas = len(gammas)
    
    avg_mse_raw = np.zeros(num_gammas)
    avg_corr_raw = np.zeros(num_gammas)
    avg_mse_pr = np.zeros(num_gammas)
    avg_corr_pr = np.zeros(num_gammas)
    avg_var_ret = np.zeros(num_gammas)
    
    alpha = 0.1
    
    # Process in batches to save memory and speed up
    batch_size = 10
    valid_sample_count = 0
    damping = 0.89
    
    for b_start in range(0, num_splits, batch_size):
        b_keys = selected_keys[b_start:b_start+batch_size]
        print(f"Processing batch {b_start // batch_size + 1}/{(num_splits+batch_size-1)//batch_size}...")
        
        target_B_list = []
        c_A_list = []
        
        for k in b_keys:
            t_indices = valid_splits[k]
            np.random.seed(hash(k) % (2**32)) 
            idx_copy = list(t_indices)
            random.shuffle(idx_copy)
            half = len(idx_copy) // 2
            set_A = idx_copy[:half]
            set_B = idx_copy[half:]
            
            raw_A = np.zeros(N_raw)
            for idx in set_A:
                row = raw_data['matrix'].getrow(idx)
                for i, val in zip(row.indices, row.data): raw_A[i] += float(val)
            
            raw_B = np.zeros(N_raw)
            for idx in set_B:
                row = raw_data['matrix'].getrow(idx)
                for i, val in zip(row.indices, row.data): raw_B[i] += float(val)
                
            sum_B = raw_B.sum()
            if sum_B <= 0: continue
            
            target_B_list.append(raw_B / sum_B)
            c_A_list.append(raw_A + alpha)
            
        if not c_A_list: continue
        
        current_batch_size = len(c_A_list)
        valid_sample_count += current_batch_size
        
        # 1. PageRank for Target B
        E_2N_B = np.zeros((2 * N_graph, current_batch_size))
        for b in range(current_batch_size):
            # Project N_raw to N_graph
            E_2N_B[raw_to_graph_idx, b] = target_B_list[b][valid_raw_idx]
            
            if E_2N_B[:N_graph, b].sum() > 0:
                E_2N_B[:N_graph, b] /= E_2N_B[:N_graph, b].sum()
            else:
                E_2N_B[:N_graph, b] = 1.0 / N_graph
                
        v_2N_B = E_2N_B.copy()
        for _ in range(60): 
            v_next = damping * M_2N_T.dot(v_2N_B) + (1 - damping) * E_2N_B
            S = np.sum(v_next, axis=0)
            missing = 1.0 - S
            missing[missing < 0] = 0
            v_next += E_2N_B * missing
            v_2N_B = v_next
        
        target_pr_B_matrix_full = v_2N_B[:N_graph, :] + v_2N_B[N_graph:, :]
        
        # Extract back to N_raw
        target_pr_B_matrix = np.zeros((N_raw, current_batch_size))
        for b in range(current_batch_size):
            target_pr_B_matrix[valid_raw_idx, b] = target_pr_B_matrix_full[raw_to_graph_idx, b]
            s = target_pr_B_matrix[:, b].sum()
            if s > 0: target_pr_B_matrix[:, b] /= s
        
        # 2. Process Set A across all Gammas
        total_cols = current_batch_size * num_gammas
        E_2N_A = np.zeros((2 * N_graph, total_cols))
        dist_A_matrix = np.zeros((N_raw, total_cols))
        base_var_list = []
        
        for b in range(current_batch_size):
            c_A = c_A_list[b]
            P_c_A = P_dir.dot(c_A)
            base_var = np.var(c_A / c_A.sum())
            base_var_list.append(base_var)
            
            for g_idx, g in enumerate(gammas):
                c_diff = (1 - g) * c_A + g * P_c_A
                dist = c_diff / c_diff.sum()
                col_idx = b * num_gammas + g_idx
                dist_A_matrix[:, col_idx] = dist
                
                # Project N_raw to N_graph
                E_2N_A[raw_to_graph_idx, col_idx] = dist[valid_raw_idx]
                if E_2N_A[:N_graph, col_idx].sum() > 0:
                    E_2N_A[:N_graph, col_idx] /= E_2N_A[:N_graph, col_idx].sum()
                else:
                    E_2N_A[:N_graph, col_idx] = 1.0 / N_graph
                
        v_2N_A = E_2N_A.copy()
        for _ in range(60):
            v_next = damping * M_2N_T.dot(v_2N_A) + (1 - damping) * E_2N_A
            S = np.sum(v_next, axis=0)
            missing = 1.0 - S
            missing[missing < 0] = 0
            v_next += E_2N_A * missing
            v_2N_A = v_next
            
        dist_pr_A_matrix_full = v_2N_A[:N_graph, :] + v_2N_A[N_graph:, :]
        
        # Extract back to N_raw
        dist_pr_A_matrix = np.zeros((N_raw, total_cols))
        for col_idx in range(total_cols):
            dist_pr_A_matrix[valid_raw_idx, col_idx] = dist_pr_A_matrix_full[raw_to_graph_idx, col_idx]
            s = dist_pr_A_matrix[:, col_idx].sum()
            if s > 0: dist_pr_A_matrix[:, col_idx] /= s
        
        # Aggregate Metrics
        for b in range(current_batch_size):
            target_B = target_B_list[b]
            target_pr_B = target_pr_B_matrix[:, b]
            base_var = base_var_list[b]
            
            for g_idx, g in enumerate(gammas):
                col_idx = b * num_gammas + g_idx
                dist_A = dist_A_matrix[:, col_idx]
                dist_pr_A = dist_pr_A_matrix[:, col_idx]
                
                mse_raw = np.mean((dist_A - target_B)**2)
                r_raw, _ = pearsonr(dist_A, target_B)
                
                mse_pr = np.mean((dist_pr_A - target_pr_B)**2)
                r_pr, _ = pearsonr(dist_pr_A, target_pr_B)
                
                avg_mse_raw[g_idx] += mse_raw
                if not np.isnan(r_raw): avg_corr_raw[g_idx] += r_raw
                
                avg_mse_pr[g_idx] += mse_pr
                if not np.isnan(r_pr): avg_corr_pr[g_idx] += r_pr
                
                avg_var_ret[g_idx] += (np.var(dist_A) / base_var)

    if valid_sample_count > 0:
        avg_mse_raw /= valid_sample_count
        avg_corr_raw /= valid_sample_count
        avg_mse_pr /= valid_sample_count
        avg_corr_pr /= valid_sample_count
        avg_var_ret = (avg_var_ret / valid_sample_count) * 100
        
    out_dir = base_dir
    
    with open(os.path.join(out_dir, 'smoothing_grid_search_results.json'), 'w') as f:
        json.dump({
            'gammas': gammas,
            'mse_raw': avg_mse_raw.tolist(),
            'corr_raw': avg_corr_raw.tolist(),
            'mse_pr': avg_mse_pr.tolist(),
            'corr_pr': avg_corr_pr.tolist(),
            'variance_ret': avg_var_ret.tolist(),
            'splits_used': valid_sample_count
        }, f, indent=4)
        
    print(f"Optimization finished over {valid_sample_count} splits. Results saved to {out_dir}/smoothing_grid_search_results.json")

if __name__ == '__main__':
    main()
