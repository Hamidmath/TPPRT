"""
Map-match cleaned GPS trajectories (from compress_routes.py) onto the SLC
MATSim road network, and emit one JSON record per trip listing the matched
edge for every emitting GPS fix.

Per-fix output:
    For each emitting Viterbi state we read the matched edge id and pair it
    with the timestamp of the GPS fix that the state explains. No per-edge
    enter/exit times, no arclength interpolation, no monotone fit. The
    popularity-matrix step (`analyze_popularity.py`) bins each fix on its own
    timestamp and counts distinct trips per (bin, edge).

Output schema:
    { "route": { "route_id": int,
                 "fixes": [[lid, t_gps], ...] } }
"""
from __future__ import annotations

import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("leuvenmapmatching").setLevel(logging.WARNING)

NETWORK_FILE = str(config.NETWORK_XML)
ROUTES_FILE = str(config.DATA_DIR / "compressed_routes.parquet")
OUTPUT_FILE = str(config.MATCHED_ROUTES)
CENTER_LAT = 40.75
CENTER_LON = -111.90
R_EARTH = 6371000
MIN_GPS_FIXES = 5


def project_to_meters(lat, lon):
    x = (lon - CENTER_LON) * np.cos(np.radians(CENTER_LAT)) * R_EARTH * (np.pi / 180)
    y = (lat - CENTER_LAT) * R_EARTH * (np.pi / 180)
    return x, y


def load_network(xml_file: str):
    """Load MATSim network; return (InMemMap, edge_lid) where
    edge_lid[(u,v)] = string link id from the MATSim XML.
    """
    logger.info(f"Loading network from {xml_file}...")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    map_con = InMemMap("slc", use_latlon=False, use_rtree=True, index_edges=True)

    nodes: Dict[int, Tuple[float, float]] = {}
    for node in root.find("nodes"):
        nid = int(node.get("id"))
        lon = float(node.get("x"))
        lat = float(node.get("y"))
        x, y = project_to_meters(lat, lon)
        nodes[nid] = (x, y)
        map_con.add_node(nid, (x, y))

    edge_lid: Dict[Tuple[int, int], str] = {}
    n_kept = 0
    for link in root.find("links"):
        modes = link.get("modes", "")
        if "car" not in modes:
            continue
        u = int(link.get("from"))
        v = int(link.get("to"))
        if u not in nodes or v not in nodes:
            continue
        lid = link.get("id")
        map_con.add_edge(u, v)
        edge_lid[(u, v)] = lid
        n_kept += 1

    logger.info(f"Network loaded: {len(nodes):,} nodes, {n_kept:,} car edges.")
    return map_con, edge_lid


def extract_fix_matches(
    matcher: DistanceMatcher,
    edge_lid: Dict[Tuple[int, int], str],
    gps_times: np.ndarray,
) -> Optional[List[Tuple[str, float]]]:
    """
    Walk `matcher.lattice_best` and return one (lid, t_gps) pair per emitting
    state, where t_gps is the timestamp of the GPS fix the state explains.

    Returns None if the matcher produced no usable emitting states.
    """
    lb = matcher.lattice_best
    if not lb:
        return None

    out: List[Tuple[str, float]] = []
    for m in lb:
        if m.obs_ne != 0:
            continue  # non-emitting state: not tied to any GPS fix
        e = m.edge_m
        if e.l2 is None:
            continue  # singleton-node fallback (rare, endpoints only)
        edge = (e.l1, e.l2)
        lid = edge_lid.get(edge)
        if lid is None:
            continue
        t = float(gps_times[m.obs])
        out.append((lid, t))

    if not out:
        return None
    return out


def process_routes():
    map_con, edge_lid = load_network(NETWORK_FILE)

    matcher = DistanceMatcher(
        map_con,
        max_dist=400,
        max_dist_init=400,
        min_prob_norm=1e-4,
        non_emitting_states=True,
    )

    logger.info("Opening parquet stream...")
    pf = pq.ParquetFile(ROUTES_FILE)

    total_processed = 0
    total_matched = 0
    n_too_short = 0
    n_no_match = 0
    n_no_fixes = 0

    buffer_df = pd.DataFrame()

    logger.info(f"Writing streaming JSON to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f_out:
        f_out.write("[\n")
        first_item = True

        for i, batch in enumerate(pf.iter_batches(batch_size=500_000)):
            df = batch.to_pandas()
            if not buffer_df.empty:
                df = pd.concat([buffer_df, df])
            if df.empty:
                continue

            unique_ids = df["route_id"].unique()
            routes_to_process = unique_ids[:-1]
            last_route_id = unique_ids[-1]
            buffer_df = df[df["route_id"] == last_route_id].copy()
            if len(routes_to_process) == 0:
                continue

            logger.info(f"Batch {i}: processing {len(routes_to_process):,} routes...")

            for rid in routes_to_process:
                total_processed += 1
                route_df = df[df["route_id"] == rid]
                if len(route_df) < MIN_GPS_FIXES:
                    n_too_short += 1
                    continue

                try:
                    px, py = project_to_meters(route_df["lat"].values, route_df["lon"].values)
                    gps_times = route_df["time_sec"].values.astype(np.float64)
                    path = list(zip(px, py))

                    res = matcher.match(path)
                    states = res[0] if isinstance(res, tuple) else res
                    if not states:
                        n_no_match += 1
                        continue

                    fixes = extract_fix_matches(matcher, edge_lid, gps_times)
                    if fixes is None:
                        n_no_fixes += 1
                        continue

                    item = {
                        "route": {
                            "route_id": int(rid),
                            "fixes": fixes,  # list of [lid, t_gps]
                        }
                    }
                    if not first_item:
                        f_out.write(",\n")
                    else:
                        first_item = False
                    json.dump(item, f_out)
                    total_matched += 1

                except Exception as exc:
                    logger.debug(f"route {rid} error: {exc}")

                if total_processed % 20_000 == 0:
                    logger.info(
                        f"  processed {total_processed:,}  matched {total_matched:,}  "
                        f"(too_short {n_too_short:,}  no_match {n_no_match:,}  "
                        f"no_fixes {n_no_fixes:,})"
                    )

        f_out.write("\n]\n")

    logger.info(
        f"Done. Total processed: {total_processed:,}   matched: {total_matched:,}\n"
        f"  too short ({MIN_GPS_FIXES}+ fixes required): {n_too_short:,}\n"
        f"  matcher returned nothing:                    {n_no_match:,}\n"
        f"  no usable emitting states:                   {n_no_fixes:,}"
    )


if __name__ == "__main__":
    process_routes()
