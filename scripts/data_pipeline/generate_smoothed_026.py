import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import sparse

INPUT_FILE = '../../data/popularity_results.npz'
OUTPUT_FILE = '../../data/popularity_results_smoothed_026.npz'
GRAPH_FILE = '../../data/city_graph_full.json'

ALPHA = 0.1       
GAMMA = 0.26      # Optimal smoothing factor found via cross-validation

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_graph(graph_file: str) -> Dict:
    with open(graph_file, 'r') as f: return json.load(f)

def load_popularity_matrix(file_path: str) -> Dict:
    loader = np.load(file_path, allow_pickle=True)
    matrix = sparse.csr_matrix(
        (loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']), 
        shape=loader['matrix_shape']
    )
    return {
        'matrix': matrix,
        'times': list(loader['times']),
        'link_ids': list(loader['link_ids'])
    }

def build_adjacency_matrix(city_graph: Dict, link_ids: List[str]) -> sparse.csr_matrix:
    n = len(link_ids)
    id_to_idx = {lid: i for i, lid in enumerate(link_ids)}
    row_ind, col_ind, data = [], [], []
    adj = city_graph.get('adjacency', {})
    
    for lid, out_links in adj.items():
        if lid not in id_to_idx: continue
        i = id_to_idx[lid]
        valid_outs = [nid for nid in out_links if nid in id_to_idx]
        if not valid_outs: continue
        degree = len(valid_outs)
        weight = 1.0 / degree 
        for out_lid in valid_outs:
            j = id_to_idx[out_lid]
            row_ind.append(i); col_ind.append(j); data.append(weight)
            
    P = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    return P

def process_and_save_matrix(pop_data: Dict, city_graph: Dict, output_path: str):
    input_times = pop_data['times']
    input_matrix = pop_data['matrix']
    
    # We must use ALL links from the graph as our universe
    graph_links = list(city_graph['links'].keys())
    n = len(graph_links)
    
    # Map raw indices to graph indices
    raw_link_ids = pop_data['link_ids']
    graph_idx_map = {lid: i for i, lid in enumerate(graph_links)}
    
    raw_to_graph_idx = []
    valid_raw_idx = []
    for i, lid in enumerate(raw_link_ids):
        if lid in graph_idx_map:
            raw_to_graph_idx.append(graph_idx_map[lid])
            valid_raw_idx.append(i)
            
    P = build_adjacency_matrix(city_graph, graph_links)
    logger.info(f"Applying smoothing (Alpha={ALPHA}, Gamma={GAMMA}) across {len(input_times)} timeframes for {n} links...")
    
    output_matrix = np.zeros((len(input_times), n), dtype=np.float32)
    
    for t_i, time_str in enumerate(input_times):
        if t_i % 100 == 0: logger.info(f"Processing timeframe ({t_i}/{len(input_times)})...")
        
        raw_row = input_matrix.getrow(t_i).toarray().flatten()
        
        # Project into 99,716 vector
        c = np.zeros(n, dtype=np.float32)
        c[raw_to_graph_idx] = raw_row[valid_raw_idx]
        
        # Laplace Smoothing
        c = c + ALPHA
        
        # Graph Diffusion
        c_diffused = (1 - GAMMA) * c + GAMMA * (P.dot(c))
        
        total_c = c_diffused.sum()
        if total_c > 0: probs = c_diffused / total_c
        else: probs = c_diffused 
        output_matrix[t_i, :] = probs
        
    final_sparse = sparse.csr_matrix(output_matrix)
    np.savez_compressed(
        output_path, 
        matrix_data=final_sparse.data, 
        matrix_indices=final_sparse.indices, 
        matrix_indptr=final_sparse.indptr,
        matrix_shape=final_sparse.shape,
        times=input_times, 
        link_ids=graph_links
    )

def main():
    base_dir = Path(__file__).parent
    input_path = base_dir / INPUT_FILE
    graph_path = base_dir / GRAPH_FILE
    output_path = base_dir / OUTPUT_FILE

    city_graph = load_graph(str(graph_path))
    pop_data = load_popularity_matrix(str(input_path))
    process_and_save_matrix(pop_data, city_graph, str(output_path))

if __name__ == "__main__":
    main()
