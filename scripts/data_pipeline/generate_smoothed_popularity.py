import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_FILE = '../../data/popularity_results.npz'
OUTPUT_FILE = '../../data/popularity_results_smoothed.npz'
GRAPH_FILE = '../../data/city_graph_full.json'

# Smoothing Parameters
ALPHA = 0.1       # Smoothing pseudo-count (Laplace smoothing)
GAMMA = 0.1       # Diffusion factor (how much weight flows to neighbors)
N_SCALE_DEFAULT = 1e6 

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------

def load_graph(graph_file: str) -> Dict:
    """Loads the pre-computed graph structure containing adjacency information."""
    logger.info(f"Loading graph from {graph_file}...")
    with open(graph_file, 'r') as f:
        return json.load(f)

def load_popularity_matrix(file_path: str) -> Dict:
    """Loads raw popularity data from .npz file."""
    logger.info(f"Loading raw traffic matrix from {file_path}...")
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
    """
    Builds a Row-Stochastic Adjacency Matrix (P) for diffusion smoothing.
    P[i, j] = 1/deg(i) if link i connects to link j.
    """
    logger.info("Building row-stochastic adjacency matrix P...")
    n = len(link_ids)
    id_to_idx = {lid: i for i, lid in enumerate(link_ids)}
    
    row_ind = []
    col_ind = []
    data = []
    
    adj = city_graph.get('adjacency', {})
    
    for lid, out_links in adj.items():
        if lid not in id_to_idx:
            continue
        
        i = id_to_idx[lid]
        
        # Only consider out_links that are part of our tracked link_ids universe
        valid_outs = [nid for nid in out_links if nid in id_to_idx]
        if not valid_outs:
            continue
            
        degree = len(valid_outs)
        weight = 1.0 / degree 
        
        for out_lid in valid_outs:
            j = id_to_idx[out_lid]
            row_ind.append(i)
            col_ind.append(j)
            data.append(weight)
            
    P = sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    logger.info(f"Adjacency matrix built. Non-zero entries: {P.nnz}")
    return P

def process_and_save_matrix(pop_data: Dict, city_graph: Dict, output_path: str):
    """
    Applies graph-based diffusion smoothing to the raw popularity matrix 
    and saves the resulting smoothed matrix.
    """
    input_times = pop_data['times']
    input_matrix = pop_data['matrix']
    src_link_ids = pop_data['link_ids']
    
    # We must align the graph links with the matrix links.
    # The 'universe' of links is defined by the input data matrix.
    n = len(src_link_ids)
    
    # Structural P Matrix
    P = build_adjacency_matrix(city_graph, src_link_ids)
    
    logger.info(f"Applying smoothing (Alpha={ALPHA}, Gamma={GAMMA}) across {len(input_times)} timeframes...")
    
    # Output Matrix (Dense first, then sparsified to save memory during iteration)
    output_matrix = np.zeros((len(input_times), n), dtype=np.float32)
    
    for t_i, time_str in enumerate(input_times):
        if t_i % 100 == 0:
            logger.info(f"Processing timeframe ({t_i}/{len(input_times)}): {time_str}...")
        
        # 1. Get Raw Vector x for this timeframe
        # input_matrix is CSR, getrow returns a 1xN CSR matrix. .toarray() makes it dense 1xN.
        x = input_matrix.getrow(t_i).toarray().flatten()
                        
        # 2. Pseudo-counts (Laplace Smoothing)
        c = x + ALPHA
        
        # 3. Graph Diffusion Smoothing
        # c_diff = (1-gamma)*c + gamma*(P @ c)
        # We use sparse matrix multiplication for speed
        c_diffused = (1 - GAMMA) * c + GAMMA * (P.dot(c))
        
        # 4. Normalize into Probabilities
        total_c = c_diffused.sum()
        if total_c > 0:
            probs = c_diffused / total_c
        else:
            probs = c_diffused 
            
        # Store in output
        output_matrix[t_i, :] = probs
        
    logger.info("Smoothing complete. Sparsifying resulting matrix...")
    final_sparse = sparse.csr_matrix(output_matrix)
    
    logger.info(f"Saving smoothed matrix to {output_path}...")
    np.savez_compressed(
        output_path, 
        matrix_data=final_sparse.data, 
        matrix_indices=final_sparse.indices, 
        matrix_indptr=final_sparse.indptr,
        matrix_shape=final_sparse.shape,
        times=input_times, 
        link_ids=src_link_ids
    )
    logger.info("Pipeline executed successfully.")

def main():
    logger.info("Starting Polished Smoothing Pipeline...")
    
    base_dir = Path(__file__).parent
    input_path = base_dir / INPUT_FILE
    graph_path = base_dir / GRAPH_FILE
    output_path = base_dir / OUTPUT_FILE

    # Ensure required files exist
    if not input_path.exists():
        logger.error(f"Input file {input_path} not found. Run analyze_popularity.py first.")
        return
        
    if not graph_path.exists():
        logger.error(f"Graph file {graph_path} not found. Please provide the city graph JSON.")
        return

    # 1. Load data
    city_graph = load_graph(str(graph_path))
    pop_data = load_popularity_matrix(str(input_path))
    
    # 2. Smooth and Save
    process_and_save_matrix(pop_data, city_graph, str(output_path))

if __name__ == "__main__":
    main()