"""
Bin matched GPS observations into a (time-bin x link) popularity matrix.

Count-based rule (walkthrough §5):
    For each matched observation (route_id, t, lid):
      1. b = floor((t - t0) / 300)              # 5-minute bin index
      2. mark the triple (b, lid, route_id) as observed
    For every (bin, link) cell, count the number of *distinct* route_ids
    observed in that cell. Multiple observations from the same trip on the
    same link inside the same 5-minute window are counted once.

Two input schemas are supported:

    "fixes":      [[lid, t_gps], ...]                 # per-fix matched output
                                                       (current `match_routes.py`)
    "routelinks": [[lid, t_enter, t_exit], ...]       # legacy per-edge output
                                                       (old `match_routes.py`)

For the legacy `routelinks` schema, `t = t_enter` is used as the
observation timestamp and each edge visit contributes one (b, lid, route_id)
triple. Per-fix and per-edge dedup yield the same cell count whenever every
trip's per-link fixes fall in a single 5-minute bin (overwhelmingly true at
this resolution).

Resolution: 5-minute bins (BIN_SIZE_SECONDS = 300).
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

BIN_SIZE_SECONDS = 300  # 5 minutes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def collect_trip_sets(routes_file: str) -> Dict[Tuple[str, str], Set[int]]:
    """Returns trips[(bin_key, lid)] = set of route_ids on that link in that bin.
    Cell count is then |set|. Accepts both 'fixes' and 'routelinks' schemas.
    """
    logger.info(f"Processing routes from {routes_file}...")
    trips: Dict[Tuple[str, str], Set[int]] = {}

    base_ts_ref = datetime(2018, 1, 1).timestamp()

    count_routes = 0
    schema_seen = None
    with open(routes_file, 'r') as f:
        data = json.load(f)
        for item in data:
            count_routes += 1
            if count_routes % 20000 == 0:
                logger.info(f"Processed {count_routes} routes...")

            route = item.get('route', {})
            rid = int(route.get('route_id'))

            if 'fixes' in route:
                schema_seen = 'fixes'
                obs_iter = ((str(f[0]), float(f[1])) for f in route['fixes'])
            elif 'routelinks' in route:
                schema_seen = 'routelinks'
                # Legacy per-edge schema: use t_enter as the observation timestamp.
                obs_iter = ((str(rl[0]), float(rl[1])) for rl in route['routelinks'])
            else:
                continue

            for lid, t in obs_iter:
                t_abs = base_ts_ref + t
                b = int(t_abs // BIN_SIZE_SECONDS)
                bin_start = b * BIN_SIZE_SECONDS
                key = datetime.fromtimestamp(bin_start).strftime('%Y-%m-%d %H:%M:%S')
                trips.setdefault((key, lid), set()).add(rid)

    logger.info(f"Finished processing {count_routes} routes total (schema='{schema_seen}').")
    return trips


def save_popularity_matrix(trips: Dict[Tuple[str, str], Set[int]], output_path: str):
    """Save as CSR .npz alongside row-timestamp and column-link-id labels."""
    logger.info("Converting to sparse matrix...")

    times = sorted({key[0] for key in trips})
    time_to_idx = {t: i for i, t in enumerate(times)}
    sorted_links = sorted({key[1] for key in trips})
    link_to_idx = {lid: i for i, lid in enumerate(sorted_links)}

    logger.info(f"Matrix dimensions: {len(times)} bins x {len(sorted_links)} links")

    row_ind, col_ind, data_vals = [], [], []
    for (t_str, lid), trip_set in trips.items():
        row_ind.append(time_to_idx[t_str])
        col_ind.append(link_to_idx[lid])
        data_vals.append(len(trip_set))

    matrix = sparse.csr_matrix(
        (data_vals, (row_ind, col_ind)),
        shape=(len(times), len(sorted_links)),
        dtype=np.int32,
    )

    logger.info(f"Saving compressed matrix to {output_path}...")
    np.savez_compressed(
        output_path,
        matrix_data=matrix.data,
        matrix_indices=matrix.indices,
        matrix_indptr=matrix.indptr,
        matrix_shape=matrix.shape,
        times=times,
        link_ids=sorted_links,
    )
    logger.info(f"Save complete. nnz={matrix.nnz:,}, dtype={matrix.dtype}")


def main():
    logger.info("Popularity Analysis Pipeline (5-min bins, distinct-trip count)")
    trips = collect_trip_sets(str(config.MATCHED_ROUTES))
    save_popularity_matrix(trips, str(config.POPULARITY_RAW_NPZ))
    logger.info("Pipeline executed successfully.")


if __name__ == "__main__":
    main()
