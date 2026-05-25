"""Build a per-bin trip-ORIGINS matrix from the matched-routes JSON.

For each route we keep ONLY the first fix; that fix's link is the
origin link and its timestamp determines the 5-minute bin. The
output is the same shape as the popularity NPZ (bins x links),
but each cell counts distinct trip *starts* at (bin, link), not
all activity on the link.

Output: data/origins_results.npz with keys
  matrix_data, matrix_indices, matrix_indptr, matrix_shape
  times, link_ids
matching the legacy CSR layout used by load_popularity_npz.
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import config

BIN_SIZE = 300
BASE_TS = datetime(2018, 1, 1).timestamp()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    routes_file = str(config.MATCHED_ROUTES)
    log.info(f"reading {routes_file}")
    with open(routes_file) as f:
        data = json.load(f)
    log.info(f"  {len(data)} routes")

    # (bin_key, lid) -> set of route_ids that start there
    starts = {}
    for item in data:
        route = item.get("route", {})
        rid = int(route.get("route_id"))
        if "fixes" in route and route["fixes"]:
            lid, t = route["fixes"][0]
            lid = str(lid)
            t_abs = BASE_TS + float(t)
            b = int(t_abs // BIN_SIZE)
            key = datetime.fromtimestamp(b * BIN_SIZE).strftime("%Y-%m-%d %H:%M:%S")
            starts.setdefault((key, lid), set()).add(rid)
        elif "routelinks" in route and route["routelinks"]:
            lid, t = route["routelinks"][0][0], route["routelinks"][0][1]
            lid = str(lid)
            t_abs = BASE_TS + float(t)
            b = int(t_abs // BIN_SIZE)
            key = datetime.fromtimestamp(b * BIN_SIZE).strftime("%Y-%m-%d %H:%M:%S")
            starts.setdefault((key, lid), set()).add(rid)

    log.info(f"  unique (bin, origin-link) cells: {len(starts)}")

    times = sorted({k[0] for k in starts})
    sorted_links = sorted({k[1] for k in starts})
    time_to_idx = {t: i for i, t in enumerate(times)}
    link_to_idx = {l: i for i, l in enumerate(sorted_links)}

    row, col, vals = [], [], []
    for (t_str, lid), rid_set in starts.items():
        row.append(time_to_idx[t_str])
        col.append(link_to_idx[lid])
        vals.append(len(rid_set))

    M = csr_matrix((vals, (row, col)),
                    shape=(len(times), len(sorted_links)),
                    dtype=np.int32)
    log.info(f"matrix shape={M.shape}  nnz={M.nnz:,}")

    out = config.DATA_DIR / "origins_results.npz"
    np.savez_compressed(out,
                          matrix_data=M.data,
                          matrix_indices=M.indices,
                          matrix_indptr=M.indptr,
                          matrix_shape=M.shape,
                          times=times,
                          link_ids=sorted_links)
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
