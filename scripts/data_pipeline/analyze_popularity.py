import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
from scipy import sparse

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ROUTES_FILE = '../../data/matched_routes.json'
OUTPUT_FILE = '../../data/popularity_results.npz'
BIN_SIZE_SECONDS = 900  # 15 minutes

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------

def process_routes(routes_file: str) -> Dict[str, Dict[str, int]]:
    """
    Processes the raw matched routes and bins traversals spatiotemporally.
    Returns nested dictionary: counts[hour_key][lid] = count
    """
    logger.info(f"Processing routes from {routes_file}...")
    counts = defaultdict(lambda: defaultdict(int))
    
    ref_date = datetime(2018, 1, 1)
    base_ts_ref = ref_date.timestamp()

    count_routes = 0
    try:
        with open(routes_file, 'r') as f:
            data = json.load(f)
            
            for item in data:
                count_routes += 1
                if count_routes % 10000 == 0:
                    logger.info(f"Processed {count_routes} routes...")
                    
                route = item.get('route', {})
                routelinks = route.get('routelinks', [])
                
                for link_info in routelinks:
                    lid = str(link_info[0])
                    t_enter = float(link_info[1])
                    
                    t_abs = base_ts_ref + t_enter
                    bin_start_ts = (t_abs // BIN_SIZE_SECONDS) * BIN_SIZE_SECONDS
                    dt_bin = datetime.fromtimestamp(bin_start_ts)
                    hour_key = dt_bin.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Count Link Usage per time bin
                    counts[hour_key][lid] += 1
                        
    except Exception as e:
        logger.error(f"Error processing routes file {routes_file}: {e}")
        raise

    logger.info(f"Finished processing {count_routes} routes total.")
    return counts

def save_popularity_matrix(counts: Dict[str, Dict[str, int]], output_path: str):
    """
    Saves popularity data as a compressed CSR matrix in .npz format.
    """
    logger.info("Converting popularity data to sparse matrix format...")
    
    times = sorted(list(counts.keys()))
    time_to_idx = {t: i for i, t in enumerate(times)}
    
    # Collect all unique link IDs across all times
    all_links = set()
    for t_str, t_data in counts.items():
        all_links.update(t_data.keys())
                
    sorted_links = sorted(list(all_links))
    link_to_idx = {lid: i for i, lid in enumerate(sorted_links)}
    
    logger.info(f"Matrix Dimensions: {len(times)} time bins x {len(sorted_links)} unique links")
    
    row_ind = []
    col_ind = []
    data_vals = []
    
    # Build COO coordinate lists
    for t_str, t_data in counts.items():
        t_idx = time_to_idx[t_str]
        for lid, count in t_data.items():
            l_idx = link_to_idx[lid]
            row_ind.append(t_idx)
            col_ind.append(l_idx)
            data_vals.append(count)
                    
    # Create CSR Matrix
    matrix = sparse.csr_matrix(
        (data_vals, (row_ind, col_ind)), 
        shape=(len(times), len(sorted_links)), 
        dtype=np.float32
    )
                               
    logger.info(f"Saving compressed matrix to {output_path}...")
    np.savez_compressed(
        output_path, 
        matrix_data=matrix.data, 
        matrix_indices=matrix.indices, 
        matrix_indptr=matrix.indptr,
        matrix_shape=matrix.shape,
        times=times, 
        link_ids=sorted_links
    )
    logger.info("Save complete.")

def main():
    logger.info("Starting Polished Popularity Analysis Pipeline...")
    
    # Resolve paths relative to this script
    base_dir = Path(__file__).parent
    routes_path = base_dir / ROUTES_FILE
    output_path = base_dir / OUTPUT_FILE
    
    # 1. Process routes into aggregations (now directly grouping by time -> link_id)
    raw_counts = process_routes(str(routes_path))
    
    # 2. Export as compressed matrix
    save_popularity_matrix(raw_counts, str(output_path))
    
    logger.info("Pipeline executed successfully.")

if __name__ == "__main__":
    main()