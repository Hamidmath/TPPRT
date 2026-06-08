"""
Generate smoothed popularity data using graph-diffusion smoothing.

Usage:
    python generate_smoothed.py                    # gamma=0.20 (paper value, default)
    python generate_smoothed.py --gamma 0.10       # gamma=0.10
    python generate_smoothed.py --gamma 0.20 --output custom_name.npz
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

ALPHA = 0.01  # Laplace smoothing pseudo-count (alpha experiment)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_graph(graph_file: str) -> Dict:
    with open(graph_file, 'r') as f:
        return json.load(f)

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
    """Builds a Row-Stochastic Adjacency Matrix (P) for diffusion smoothing."""
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

def process_and_save_matrix(pop_data: Dict, city_graph: Dict, output_path: str, gamma: float):
    """Applies graph-based diffusion smoothing to the raw popularity matrix."""
    input_times = pop_data['times']
    input_matrix = pop_data['matrix']

    # Use ALL links from the graph as the universe (99,716 links)
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
    logger.info(f"Applying smoothing (Alpha={ALPHA}, Gamma={gamma}) across {len(input_times)} timeframes for {n} links...")

    output_matrix = np.zeros((len(input_times), n), dtype=np.float32)

    for t_i, time_str in enumerate(input_times):
        if t_i % 100 == 0: logger.info(f"Processing timeframe ({t_i}/{len(input_times)})...")

        raw_row = input_matrix.getrow(t_i).toarray().flatten()

        # Project into full graph vector
        c = np.zeros(n, dtype=np.float32)
        c[raw_to_graph_idx] = raw_row[valid_raw_idx]

        # Laplace Smoothing
        c = c + ALPHA

        # Graph Diffusion
        c_diffused = (1 - gamma) * c + gamma * (P.dot(c))

        total_c = c_diffused.sum()
        if total_c > 0:
            probs = c_diffused / total_c
        else:
            probs = c_diffused
        output_matrix[t_i, :] = probs

    # After Laplace smoothing every cell is positive, so the matrix is
    # effectively dense. Storing it as CSR roughly doubles memory (data
    # + indices arrays), so we save it dense instead. Readers detect
    # the format via the presence of the 'matrix' key.
    np.savez_compressed(
        output_path,
        matrix=output_matrix,
        matrix_shape=np.array(output_matrix.shape),
        times=input_times,
        link_ids=graph_links,
    )
    logger.info(f"Saved smoothed matrix to {output_path} (dense, {output_matrix.shape})")

def main():
    parser = argparse.ArgumentParser(description='Generate smoothed popularity data')
    parser.add_argument('--gamma', type=float, default=0.20,
                        help='Diffusion factor (default: 0.20, the value used in the paper)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: auto-generated from gamma)')
    args = parser.parse_args()

    if args.output:
        output_path = config.DATA_DIR / args.output
    else:
        gamma_str = str(args.gamma).replace('.', '')
        output_path = config.DATA_DIR / f'popularity_results_smoothed_{gamma_str}.npz'
        # Special case: the main smoothed file (used by default)
        if args.gamma == 0.20:
            output_path = config.POPULARITY_NPZ

    city_graph = load_graph(str(config.GRAPH_FILE))
    pop_data = load_popularity_matrix(str(config.POPULARITY_RAW_NPZ))
    process_and_save_matrix(pop_data, city_graph, str(output_path), args.gamma)

if __name__ == "__main__":
    main()
