import json
import logging
import random
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import time

from run_two_phase_pagerank import build_two_phase_matrix, run_power_iteration

logging.basicConfig(level=logging.WARNING, format='%(message)s')

def extract_vector(matrix, t_idx, pop_link_ids, lid_to_idx, N):
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
    return E_N

def main():
    base_dir = Path(__file__).parent
    
    with open(base_dir / '../../data/city_graph_full.json', 'r') as f:
        graph_data = json.load(f)
        
    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    
    print("Building 2N Matrix...")
    M_2N = build_two_phase_matrix(graph_data)
    
    print("Loading RAW NPZ...")
    loader_raw = np.load(base_dir / '../../data/popularity_results.npz', allow_pickle=True)
    matrix_raw = csr_matrix((loader_raw['matrix_data'], loader_raw['matrix_indices'], loader_raw['matrix_indptr']), shape=loader_raw['matrix_shape'])
    times_raw = list(loader_raw['times'])
    pop_link_ids_raw = list(loader_raw['link_ids'])
    
    print("Loading SMOOTHED NPZ...")
    loader_smooth = np.load(base_dir / '../../data/popularity_results_smoothed.npz', allow_pickle=True)
    matrix_smooth = csr_matrix((loader_smooth['matrix_data'], loader_smooth['matrix_indices'], loader_smooth['matrix_indptr']), shape=loader_smooth['matrix_shape'])
    times_smooth = list(loader_smooth['times'])
    pop_link_ids_smooth = list(loader_smooth['link_ids'])
    
    common_times = list(set(times_raw) & set(times_smooth))
    sample_times = random.sample(common_times, 490)
    
    print(f"Evaluating {len(sample_times)} timeframes for both RAW and SMOOTHED targets...")
    
    raw_mres = []
    smooth_mres = []
    
    start_time = time.time()
    
    for idx, t_str in enumerate(sample_times):
        # 1. RAW PIPELINE
        t_idx_r = times_raw.index(t_str)
        E_N_raw = extract_vector(matrix_raw, t_idx_r, pop_link_ids_raw, lid_to_idx, N)
        E_2N_raw = np.concatenate([E_N_raw, np.zeros(N)])
        v_2N_raw = run_power_iteration(M_2N, E_2N_raw)
        
        v_final_raw = v_2N_raw[:N] + v_2N_raw[N:]
        v_final_raw /= np.sum(v_final_raw)
        
        mre_raw = np.mean(np.abs(E_N_raw - v_final_raw) / (E_N_raw + 1e-9))
        raw_mres.append(mre_raw)
        
        # 2. SMOOTHED PIPELINE
        t_idx_s = times_smooth.index(t_str)
        E_N_smooth = extract_vector(matrix_smooth, t_idx_s, pop_link_ids_smooth, lid_to_idx, N)
        E_2N_smooth = np.concatenate([E_N_smooth, np.zeros(N)])
        v_2N_smooth = run_power_iteration(M_2N, E_2N_smooth)
        
        v_final_smooth = v_2N_smooth[:N] + v_2N_smooth[N:]
        v_final_smooth /= np.sum(v_final_smooth)
        
        mre_smooth = np.mean(np.abs(E_N_smooth - v_final_smooth) / (E_N_smooth + 1e-9))
        smooth_mres.append(mre_smooth)
        
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[{idx+1}/490] processed... (Elapsed: {elapsed:.1f}s)")
            
    avg_raw = np.mean(raw_mres)
    avg_smooth = np.mean(smooth_mres)
    
    print("\n=======================================================")
    print("                FINAL COMPARISON RESULTS               ")
    print("=======================================================")
    print(f"Sample Size: {len(sample_times)} timeframes (1-month dataset)")
    print(f"Average MRE against RAW sparse target:      {avg_raw:,.4f}")
    print(f"Average MRE against SMOOTHED dense target:  {avg_smooth:.4f}")
    print("=======================================================\n")
    
    out_file = base_dir / '../../results/comparison_490_mre_results.txt'
    with open(out_file, 'w') as f:
        f.write(f"Sample Size: {len(sample_times)} timeframes\n")
        f.write(f"Average MRE RAW: {avg_raw:,.4f}\n")
        f.write(f"Average MRE SMOOTHED: {avg_smooth:.4f}\n\n")
        f.write("Timeframe\tRAW_MRE\tSMOOTH_MRE\n")
        for i, t_str in enumerate(sample_times):
            f.write(f"{t_str}\t{raw_mres[i]:.4f}\t{smooth_mres[i]:.4f}\n")
    print(f"Detailed output saved to {out_file}")

if __name__ == '__main__':
    main()