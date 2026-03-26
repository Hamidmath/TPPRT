import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
GRAPH_FILE = '../../data/city_graph_full.json'
POPULARITY_NPZ = '../../data/popularity_results_smoothed.npz'
OUTPUT_FILE = '../../data/two_phase_pagerank_vector.json'

# Default Hyperparameters
PARAMS = {
    'alpha_s': 0.0,
    'alpha_l': 0.0,
    'beta': 0.9,
    'damping': 0.80,
    'mu': 20.0,
    'tau': 1.0,
    'top_k': 100,
    'top_k_boost': 1.0,
    'max_iters': 100,
    'tol': 1e-6
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------

def compute_speed_lane_weights(graph_data: Dict, is_up_phase: bool) -> np.ndarray:
    """Computes a vector of physics-based weights, perfectly matching the original logic."""
    links = list(graph_data['links'].keys())
    N = len(links)
    weights = np.ones(N)
    
    alpha_s = PARAMS['alpha_s'] if is_up_phase else -PARAMS['alpha_s']
    alpha_l = PARAMS['alpha_l'] if is_up_phase else -PARAMS['alpha_l']
    
    speeds = []
    lanes = []
    
    for lid in links:
        edge_data = graph_data['links'].get(lid, {})
        speeds.append(edge_data.get('speed', 11.17))
        lanes.append(edge_data.get('lanes', 1.0))
        
    speeds = np.array(speeds)
    lanes = np.array(lanes)
    
    if np.mean(speeds) > 0: speeds = speeds / np.mean(speeds)
    if np.mean(lanes) > 0: lanes = lanes / np.mean(lanes)
    
    for i in range(N):
        weight = 1.0
        if alpha_s != 0: weight *= (speeds[i] ** alpha_s)
        if alpha_l != 0: weight *= (lanes[i] ** alpha_l)
        weights[i] = weight
        
    return weights

def build_phase_matrix(graph_data: Dict, weights: np.ndarray) -> csr_matrix:
    """Builds a standard N x N transition matrix P for a single phase.

    Includes travel-time self-loops scaled by PARAMS['mu']. Each link retains
    a fraction of its mass proportional to length/speed, modeling the dwell time
    vehicles spend traversing the link.
    """
    links = list(graph_data['links'].keys())
    N = len(links)
    id_to_idx = {lid: i for i, lid in enumerate(links)}
    mu = PARAMS.get('mu', 0.0)

    row = []
    col = []
    data = []

    adj = graph_data.get('adjacency', {})

    for i, lid in enumerate(links):
        out_links = adj.get(lid, [])

        successors_indices = [id_to_idx[out_lid] for out_lid in out_links if out_lid in id_to_idx]

        out_weights = [weights[j] for j in successors_indices]

        # Travel-time self-loop: mass stays proportional to time spent on link
        edge_data = graph_data['links'][lid]
        self_w = mu * edge_data.get('length', 100.0) / max(edge_data.get('speed', 11.17), 0.1) if mu > 0 else 0.0

        total_weight = sum(out_weights) + self_w

        if total_weight > 0:
            if self_w > 0:
                row.append(i)
                col.append(i)
                data.append(self_w / total_weight)
            for j, w in zip(successors_indices, out_weights):
                row.append(i)
                col.append(j)
                data.append(w / total_weight)
                
    return csr_matrix((data, (row, col)), shape=(N, N))


def build_two_phase_matrix(graph_data: Dict) -> csr_matrix:
    """Builds the 2N x 2N Two-Phase Markov block matrix."""
    N = len(graph_data['links'])
    logger.info(f"Building 2N x 2N Two-Phase Matrix (N={N})")
    
    up_weights = compute_speed_lane_weights(graph_data, is_up_phase=True)
    down_weights = compute_speed_lane_weights(graph_data, is_up_phase=False)
    
    P_up_raw = build_phase_matrix(graph_data, up_weights)
    P_down_raw = build_phase_matrix(graph_data, down_weights)
    
    beta = PARAMS['beta']
    P_up = P_up_raw.multiply(1.0 - beta)
    
    row_T = np.arange(N)
    col_T = np.arange(N)
    data_T = np.ones(N) * beta
    T_down = csr_matrix((data_T, (row_T, col_T)), shape=(N, N))
    
    zero_block = csr_matrix((N, N))
    
    top_block = hstack([P_up, T_down])
    bottom_block = hstack([zero_block, P_down_raw])
    
    M_2N = vstack([top_block, bottom_block])
    logger.info(f"Generated 2N Matrix of shape {M_2N.shape}")
    
    return M_2N

def build_teleportation_vector(graph_data: Dict, target_time: str) -> np.ndarray:
    """Constructs the exact E_N Teleportation vector by extracting the smoothed ground truth."""
    logger.info(f"Extracting Teleportation Vector for {target_time} from target matrix...")
    links = list(graph_data['links'].keys())
    N = len(links)
    
    loader = np.load(POPULARITY_NPZ, allow_pickle=True)
    matrix = csr_matrix((loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']), shape=loader['matrix_shape'])
    times = list(loader['times'])
    pop_link_ids = list(loader['link_ids'])
    
    if target_time not in times:
        target_time = times[0]
        logger.warning(f"Target time not found. Defaulting to {target_time}")
        
    t_idx = times.index(target_time)
    row = matrix.getrow(t_idx)
    
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
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
        
    return E_N

def sharpen_teleportation(E_N: np.ndarray, tau: float) -> np.ndarray:
    """Raise teleportation vector to power tau and renormalize to concentrate mass on top links."""
    if tau == 1.0:
        return E_N.copy()
    E_sharp = np.power(E_N, tau)
    s = np.sum(E_sharp)
    if s > 0:
        E_sharp /= s
    else:
        E_sharp = E_N.copy()
    return E_sharp

def boost_top_k(E_N: np.ndarray, K: int, boost: float) -> np.ndarray:
    """Multiply teleportation weights of top-K links by boost factor, then renormalize."""
    if boost == 1.0 or K <= 0:
        return E_N.copy()
    E_b = E_N.copy()
    top_idx = np.argsort(E_N)[::-1][:K]
    E_b[top_idx] *= boost
    E_b /= np.sum(E_b)
    return E_b

def run_power_iteration(M_2N: csr_matrix, E_2N: np.ndarray) -> np.ndarray:
    """Executes the PageRank power iteration."""
    logger.info("Executing 2N Power Iteration...")
    damping = PARAMS['damping']
    tol = PARAMS['tol']
    max_iters = PARAMS['max_iters']
    
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
            logger.info(f"Converged successfully at iteration {i} (diff={diff:.8f})")
            break
            
    return v_2N

def main():
    logger.info("Starting Polished Two-Phase PageRank Execution...")
    
    base_dir = Path(__file__).parent
    
    with open(base_dir / GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)
        
    links = list(graph_data['links'].keys())
    N = len(links)
    
    M_2N = build_two_phase_matrix(graph_data)
    
    target_time = "2018-09-08 08:00:00"
    E_N = build_teleportation_vector(graph_data, target_time)

    # Concentrate teleportation on top-K links
    E_tele = sharpen_teleportation(E_N, PARAMS['tau'])
    E_tele = boost_top_k(E_tele, PARAMS['top_k'], PARAMS['top_k_boost'])
    E_2N = np.concatenate([E_tele, np.zeros(N)])
    
    v_2N = run_power_iteration(M_2N, E_2N)
    
    logger.info("Collapsing 2N output into macroscopic N dimensions...")
    v_up = v_2N[:N]
    v_down = v_2N[N:]
    v_final = v_up + v_down
    
    v_final = v_final / np.sum(v_final)
    
    # Evaluate MRE
    mre_sum = 0
    for i in range(N):
        mre_sum += abs(E_N[i] - v_final[i]) / (E_N[i] + 1e-9)
    logger.info(f"SUCCESS: Architectural Mean Relative Error (MRE) against target: {mre_sum / N:.4f}")

    # Top-100 MRE
    top_100_indices = np.argsort(E_N)[::-1][:100]
    t_top100 = E_N[top_100_indices]
    p_top100 = v_final[top_100_indices]
    mre_top100 = np.mean(np.abs(t_top100 - p_top100) / (t_top100 + 1e-9))
    logger.info(f"Top-100 MRE: {mre_top100:.4f}")

    # Save Results
    result_dict = {lid: float(v_final[i]) for i, lid in enumerate(links)}
    out_path = base_dir / OUTPUT_FILE
    
    with open(out_path, 'w') as f:
        json.dump(result_dict, f)
        
    logger.info(f"Successfully generated Two-Phase PageRank vector at {out_path}.")

if __name__ == "__main__":
    main()