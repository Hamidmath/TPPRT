import pandas as pd
import pyarrow.parquet as pq
import xml.etree.ElementTree as ET
import sys
from pathlib import Path

from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher
import numpy as np
import json
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

# Logging Setup
logger = logging.getLogger("leuvenmapmatching")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

# Configuration
NETWORK_FILE = str(config.NETWORK_XML)
ROUTES_FILE = str(config.DATA_DIR / 'compressed_routes.parquet')
OUTPUT_FILE = str(config.MATCHED_ROUTES)
CENTER_LAT = 40.75
CENTER_LON = -111.90
R_EARTH = 6371000

def project_to_meters(lat, lon):
    x = (lon - CENTER_LON) * np.cos(np.radians(CENTER_LAT)) * R_EARTH * (np.pi / 180)
    y = (lat - CENTER_LAT) * R_EARTH * (np.pi / 180)
    return x, y

def load_network(xml_file):
    print("Loading network...")
    try:
        tree = ET.parse(xml_file)
    except Exception as e:
        print(f"Failed to parse XML: {e}")
        return None, None, None

    root = tree.getroot()

    map_con = InMemMap("slc", use_latlon=False, use_rtree=True, index_edges=True)

    nodes = {}
    for node in root.find('nodes'):
        nid = int(node.get('id'))
        lon = float(node.get('x'))
        lat = float(node.get('y'))
        x, y = project_to_meters(lat, lon)
        nodes[nid] = (x, y)
        map_con.add_node(nid, (x, y))

    for link in root.find('links'):
        lid = link.get('id')
        from_n = int(link.get('from'))
        to_n = int(link.get('to'))
        length = float(link.get('length'))
        modes = link.get('modes', '')

        if 'car' not in modes:
            continue

        if from_n in nodes and to_n in nodes:
            map_con.add_edge(from_n, to_n)

    edge_to_lid = {}
    for link in root.find('links'):
        edge_to_lid[(int(link.get('from')), int(link.get('to')))] = link.get('id')

    all_x = [pos[0] for pos in nodes.values()]
    all_y = [pos[1] for pos in nodes.values()]
    if all_x:
        print(f"Network Bounds (Meters): X[{min(all_x):.1f}, {max(all_x):.1f}], Y[{min(all_y):.1f}, {max(all_y):.1f}]")

    print(f"Graph Stats: {len(nodes)} nodes, {len(edge_to_lid)} edges")
    return map_con, nodes, edge_to_lid

def interpolate_time(path_nodes, path_coords, gps_trace, nodes_dict):
    if len(path_coords) < 2: return []

    node_dists = [0.0]
    for i in range(1, len(path_coords)):
        dx = path_coords[i][0] - path_coords[i-1][0]
        dy = path_coords[i][1] - path_coords[i-1][1]
        node_dists.append(node_dists[-1] + np.sqrt(dx*dx + dy*dy))

    gps_dists = []
    gps_times = []

    for gx, gy, gt in gps_trace:
        min_d = float('inf')
        best_dist = 0.0
        for idx, (nx, ny) in enumerate(path_coords):
             dist = (gx-nx)**2 + (gy-ny)**2
             if dist < min_d:
                 min_d = dist
                 best_dist = node_dists[idx]
        gps_dists.append(best_dist)
        gps_times.append(gt)

    if len(gps_dists) < 2: return []

    sorted_pairs = sorted(zip(gps_dists, gps_times))
    calib_dist = [p[0] for p in sorted_pairs]
    calib_time = [p[1] for p in sorted_pairs]

    return np.interp(node_dists, calib_dist, calib_time)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super(NpEncoder, self).default(obj)

def process_routes():
    map_con, nodes_dict, edge_to_lid = load_network(NETWORK_FILE)
    if not map_con: return

    matcher = DistanceMatcher(map_con,
                            max_dist=400,
                            max_dist_init=400,
                            min_prob_norm=0.0001,
                            non_emitting_states=True)

    print("Opening parquet stream...")
    pf = pq.ParquetFile(ROUTES_FILE)

    total_processed = 0
    total_matched = 0

    buffer_df = pd.DataFrame()

    print(f"Writing streaming JSON to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f_out:
        f_out.write('[\n')
        first_item = True

        for i, batch in enumerate(pf.iter_batches(batch_size=500000)):
            df = batch.to_pandas()

            if not buffer_df.empty:
                df = pd.concat([buffer_df, df])

            if df.empty: continue

            unique_ids = df['route_id'].unique()
            routes_to_process = unique_ids[:-1]
            last_route_id = unique_ids[-1]

            buffer_df = df[df['route_id'] == last_route_id].copy()

            if len(routes_to_process) == 0:
                continue

            print(f"Batch {i}: Processing {len(routes_to_process)} routes...")

            for rid in routes_to_process:
                try:
                    route_df = df[df['route_id'] == rid]

                    if len(route_df) < 5:
                        total_processed += 1
                        continue

                    path_x, path_y = project_to_meters(route_df['lat'].values, route_df['lon'].values)
                    path = list(zip(path_x, path_y))
                    times = route_df['time_sec'].values

                    match_res = matcher.match(path)

                    if isinstance(match_res, tuple):
                        match_res = match_res[0]

                    matched_edges = match_res
                    if not matched_edges:
                         total_processed += 1
                         continue

                    path_nodes = [matched_edges[0][0]]
                    for u, v in matched_edges:
                        path_nodes.append(v)

                    try:
                        path_coords = [nodes_dict[nid] for nid in path_nodes]
                    except KeyError:
                        continue

                    gps_trace = list(zip(path_x, path_y, times))
                    node_times = interpolate_time(path_nodes, path_coords, gps_trace, nodes_dict)

                    route_links = []
                    for k, (u, v) in enumerate(matched_edges):
                        lid = edge_to_lid.get((u, v))
                        if lid:
                             s_time = node_times[k]
                             e_time = node_times[k+1]
                             route_links.append((lid, s_time, e_time))

                    if route_links:
                        item = {
                            "route": {
                                "route_id": int(rid),
                                "routelinks": route_links
                            }
                        }
                        if not first_item:
                            f_out.write(',\n')
                        else:
                            first_item = False

                        json.dump(item, f_out, cls=NpEncoder)
                        total_matched += 1

                except Exception as e:
                    pass

                total_processed += 1
                if total_processed % 1000 == 0:
                    print(f"Processed {total_processed} routes... Matched: {total_matched}")

        f_out.write('\n]')

    print(f"Finished. Total processed: {total_processed}. Matched: {total_matched}")

if __name__ == '__main__':
    process_routes()
