import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching.matcher.distance import DistanceMatcher

CENTER_LAT = 40.75
CENTER_LON = -111.90
R_EARTH = 6_371_000
MIN_GPS_FIXES = 5
MAX_DIST = 400


def project_to_meters(lat, lon):
    x = (lon - CENTER_LON) * np.cos(np.radians(CENTER_LAT)) * R_EARTH * (np.pi / 180)
    y = (lat - CENTER_LAT) * R_EARTH * (np.pi / 180)
    return x, y


def load_network(xml_file):
    root = ET.parse(xml_file).getroot()
    map_con = InMemMap("slc", use_latlon=False, use_rtree=True, index_edges=True)
    nodes = {}
    for node in root.find("nodes"):
        nid = int(node.get("id"))
        x, y = project_to_meters(float(node.get("y")), float(node.get("x")))
        nodes[nid] = (x, y)
        map_con.add_node(nid, (x, y))
    edge_lid = {}
    for link in root.find("links"):
        if "car" not in link.get("modes", ""):
            continue
        u, v = int(link.get("from")), int(link.get("to"))
        if u not in nodes or v not in nodes:
            continue
        map_con.add_edge(u, v)
        edge_lid[(u, v)] = link.get("id")
    print(f"Network: {len(nodes):,} nodes, {len(edge_lid):,} car edges.")
    return map_con, edge_lid


def extract_fixes(matcher, edge_lid, gps_times):
    out = []
    for m in matcher.lattice_best:
        if m.obs_ne != 0:
            continue
        e = m.edge_m
        if e.l2 is None:
            continue
        lid = edge_lid.get((e.l1, e.l2))
        if lid is None:
            continue
        out.append((lid, float(gps_times[m.obs])))
    return out or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True, help="MATSim network XML")
    ap.add_argument("--routes", default="compressed_routes.parquet")
    ap.add_argument("--output", default="matched_routes.json")
    args = ap.parse_args()

    map_con, edge_lid = load_network(args.network)
    matcher = DistanceMatcher(
        map_con, max_dist=MAX_DIST, max_dist_init=MAX_DIST,
        min_prob_norm=1e-4, non_emitting_states=True,
    )

    pf = pq.ParquetFile(args.routes)
    total = matched = 0
    buffer_df = pd.DataFrame()
    with open(args.output, "w") as f_out:
        f_out.write("[\n")
        first = True
        for batch in pf.iter_batches(batch_size=500_000):
            df = batch.to_pandas()
            if not buffer_df.empty:
                df = pd.concat([buffer_df, df])
            if df.empty:
                continue
            ids = df["route_id"].unique()
            to_process, last = ids[:-1], ids[-1]
            buffer_df = df[df["route_id"] == last].copy()
            for rid in to_process:
                total += 1
                route = df[df["route_id"] == rid]
                if len(route) < MIN_GPS_FIXES:
                    continue
                try:
                    px, py = project_to_meters(route["lat"].values, route["lon"].values)
                    times = route["time_sec"].values.astype(np.float64)
                    res = matcher.match(list(zip(px, py)))
                    states = res[0] if isinstance(res, tuple) else res
                    if not states:
                        continue
                    fixes = extract_fixes(matcher, edge_lid, times)
                    if fixes is None:
                        continue
                    item = {"route": {"route_id": int(rid), "fixes": fixes}}
                    f_out.write("" if first else ",\n")
                    json.dump(item, f_out)
                    first = False
                    matched += 1
                except Exception:
                    pass
                if total % 20_000 == 0:
                    print(f"  processed {total:,}, matched {matched:,}")
        f_out.write("\n]\n")

    print(f"Done. processed {total:,}, matched {matched:,} -> {args.output}")


if __name__ == "__main__":
    main()
