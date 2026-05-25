"""Generate an interactive map showing long MATSim links and their visit
status under a given matched_routes JSON.

All car links are drawn as a thin gray base layer. Long links (length
>= --threshold meters) are drawn on top, color-coded by visit status:

    red    = mapped to OSM, definitely 0 visits in this run
    orange = no OSM->MATSim mapping (visit status undetermined)
    green  = visited (only highlighted when --show-visited is set)

Usage:
    python analysis/generate_unvisited_map.py \\
        --matched data/matched_routes_osm.json \\
        --xml data/slc_network.xml \\
        --pkl data/network_osm.pkl \\
        --out data/network_long_unvisited_osm.html \\
        --threshold 500
"""
import argparse
import json
import os
import pickle
import time
import xml.etree.ElementTree as ET
from collections import Counter

import folium


def load_visits(path):
    print(f"Loading {path} ({os.path.getsize(path)/1e6:.0f} MB)...", flush=True)
    t0 = time.time()
    with open(path) as f:
        data = json.load(f)
    counts = Counter()
    for item in data:
        if "route" not in item:
            continue
        for lid, _t in item["route"]["fixes"]:
            counts[str(lid)] += 1
    print(
        f"  {sum(counts.values()):,} fix-counts, "
        f"{len(counts):,} unique links ({time.time()-t0:.0f}s)",
        flush=True,
    )
    return counts


def load_matsim(xml_path):
    print(f"Loading MATSim XML: {xml_path}")
    root = ET.parse(xml_path).getroot()
    nodes = {}
    for n in root.find("nodes"):
        nodes[n.get("id")] = (float(n.get("y")), float(n.get("x")))  # lat, lon
    links = []
    for lk in root.find("links"):
        if "car" not in lk.get("modes", ""):
            continue
        u = lk.get("from")
        v = lk.get("to")
        if u not in nodes or v not in nodes:
            continue
        links.append({
            "id": lk.get("id"),
            "from": u,
            "to": v,
            "lat1": nodes[u][0],
            "lon1": nodes[u][1],
            "lat2": nodes[v][0],
            "lon2": nodes[v][1],
            "length": float(lk.get("length")),
        })
    print(f"  car links plotted: {len(links):,}")
    return links, nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument(
        "--pkl",
        default=None,
        help="OSM->MATSim mapping pickle (network_osm.pkl). If given, "
        "long links not in the mapping are drawn orange instead of red.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=500.0)
    ap.add_argument("--show-visited", action="store_true")
    args = ap.parse_args()

    visits = load_visits(args.matched)
    links, _ = load_matsim(args.xml)

    mapped_ids = None
    if args.pkl is not None and os.path.exists(args.pkl):
        print(f"Loading OSM->MATSim mapping: {args.pkl}")
        with open(args.pkl, "rb") as f:
            data = pickle.load(f)
        mapped_ids = set(data["edge_to_matsim"].values())
        print(f"  matsim links in mapping: {len(mapped_ids):,}")

    L = args.threshold
    n_long = n_red = n_orange = n_green = 0
    long_red = []
    long_orange = []
    long_green = []
    for lk in links:
        if lk["length"] < L:
            continue
        n_long += 1
        v = visits.get(lk["id"], 0)
        if v > 0:
            long_green.append(lk)
            n_green += 1
        elif mapped_ids is not None and lk["id"] not in mapped_ids:
            long_orange.append(lk)
            n_orange += 1
        else:
            long_red.append(lk)
            n_red += 1
    print(
        f"\nlong links (>= {L} m): n={n_long:,}\n"
        f"  red (mapped, 0 visits): {n_red:,}\n"
        f"  orange (no mapping):    {n_orange:,}\n"
        f"  green (visited):        {n_green:,}"
    )

    # Center on SLC
    lat0 = sum(lk["lat1"] for lk in links) / len(links)
    lon0 = sum(lk["lon1"] for lk in links) / len(links)
    m = folium.Map(
        location=[lat0, lon0], zoom_start=11, tiles="cartodbpositron"
    )

    print("Building base layer (all car links, gray)...")
    base_features = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lk["lon1"], lk["lat1"]], [lk["lon2"], lk["lat2"]]],
                },
                "properties": {"id": lk["id"]},
            }
            for lk in links
        ],
    }
    folium.GeoJson(
        base_features,
        name="all car links",
        style_function=lambda f: {
            "color": "#888888",
            "weight": 0.4,
            "opacity": 0.45,
        },
    ).add_to(m)

    def add_layer(features, name, color, weight, popup_prefix):
        gj_features = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [lk["lon1"], lk["lat1"]],
                            [lk["lon2"], lk["lat2"]],
                        ],
                    },
                    "properties": {
                        "popup": (
                            f"{popup_prefix}<br>"
                            f"link {lk['id']}<br>"
                            f"length {lk['length']:.0f} m<br>"
                            f"visits {visits.get(lk['id'], 0):,}"
                        ),
                    },
                }
                for lk in features
            ],
        }
        folium.GeoJson(
            gj_features,
            name=name,
            style_function=lambda f, c=color, w=weight: {
                "color": c,
                "weight": w,
                "opacity": 1.0,
            },
            popup=folium.GeoJsonPopup(fields=["popup"], aliases=[""]),
        ).add_to(m)

    if long_red:
        add_layer(
            long_red,
            f"long unvisited (mapped, n={n_red})",
            "#d62728",
            3.0,
            "<b>long unvisited (mapped)</b>",
        )
    if long_orange:
        add_layer(
            long_orange,
            f"long no-mapping (n={n_orange})",
            "#ff9900",
            3.0,
            "<b>long no OSM->MATSim mapping</b>",
        )
    if args.show_visited and long_green:
        add_layer(
            long_green,
            f"long visited (n={n_green})",
            "#2ca02c",
            2.0,
            "<b>long visited</b>",
        )

    folium.LayerControl(collapsed=False).add_to(m)
    print(f"Saving {args.out}...")
    m.save(args.out)
    sz = os.path.getsize(args.out) / 1e6
    print(f"  wrote {sz:.1f} MB")


if __name__ == "__main__":
    main()
