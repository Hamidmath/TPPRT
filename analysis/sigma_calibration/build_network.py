"""Build OSM-shape-preserving network pickle for Leuven map-matching.

Follows the Leuven library's official OSM ingestion pattern (their docs at
LeuvenMapMatching/docs/usage/openstreetmap.rst):

    for way in osm.ways:                                  # drivable highway
        for n_a, n_b in zip(way.nodes, way.nodes[1:]):    # consecutive shape pair
            map_con.add_edge(n_a, n_b)
            map_con.add_edge(n_b, n_a)
    for node in osm.nodes:
        map_con.add_node(node.id, (node.lat, node.lon))

We additionally build (osm_node_a, osm_node_b) -> matsim_link_id mapping by
walking the OSM degree-2 chain from each MATSim node to the next and looking
the result up in the MATSim link table. This lets us aggregate matched
sub-edges back to MATSim links for the popularity matrix.

Reads:
    ~/tppr/inputs/utah.osm.pbf  (Geofabrik Utah extract)
    ~/tppr/inputs/slc_network.xml  (MATSim car network)
Writes:
    ~/tppr/inputs/network_osm.pkl
        proj_nodes: {osm_node_id: (x, y)}  -- equirectangular at (40.75, -111.90)
        edges: list of (n_a, n_b)
        edge_to_matsim: {(n_a, n_b): matsim_link_id_str}
        meta: counts, bbox, drivable tags
    ~/tppr/inputs/build_network.log
"""
import math
import os
import pickle
import time
import xml.etree.ElementTree as ET

import osmium

CENTER_LAT = 40.75
CENTER_LON = -111.90
R = 6371000.0
LAT_MIN, LAT_MAX = 40.30, 40.95
LON_MIN, LON_MAX = -112.30, -111.50

DRIVABLE = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified",
    "residential",
    "living_street",
    "service",
    "road",
}

INPUTS = os.path.expanduser("~/tppr/inputs")
PBF = os.path.join(INPUTS, "utah.osm.pbf")
XML = os.path.join(INPUTS, "slc_network.xml")
OUT = os.path.join(INPUTS, "network_osm.pkl")
LOG = os.path.join(INPUTS, "build_network.log")

T0 = time.time()


def log(msg):
    line = f"[{time.time()-T0:7.1f}s] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def project(lat, lon):
    cosphi = math.cos(math.radians(CENTER_LAT))
    x = (lon - CENTER_LON) * cosphi * R * (math.pi / 180.0)
    y = (lat - CENTER_LAT) * R * (math.pi / 180.0)
    return x, y


class WayCollector(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.ways = []

    def way(self, w):
        hw = w.tags.get("highway")
        if hw is None or hw not in DRIVABLE:
            return
        nodes = [n.ref for n in w.nodes]
        if len(nodes) < 2:
            return
        oneway = w.tags.get("oneway", "no")
        self.ways.append((w.id, nodes, oneway))


class NodeCollector(osmium.SimpleHandler):
    def __init__(self, needed):
        super().__init__()
        self.needed = needed
        self.nodes = {}

    def node(self, n):
        if n.id not in self.needed:
            return
        lat, lon = n.location.lat, n.location.lon
        if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
            self.nodes[n.id] = (lat, lon)


def main():
    open(LOG, "w").close()
    log(f"Start. PBF={PBF} XML={XML}")
    log(f"bbox lat={LAT_MIN}..{LAT_MAX} lon={LON_MIN}..{LON_MAX}")
    log(f"drivable={sorted(DRIVABLE)}")

    log("Parsing MATSim XML...")
    mt = ET.parse(XML).getroot()
    matsim_node_set = {int(n.get("id")) for n in mt.find("nodes")}
    matsim_links = {}
    for lk in mt.find("links"):
        if "car" not in lk.get("modes", ""):
            continue
        f = int(lk.get("from"))
        t = int(lk.get("to"))
        matsim_links[(f, t)] = lk.get("id")
    log(f"  matsim nodes={len(matsim_node_set):,}, car links={len(matsim_links):,}")

    log("PBF pass 1: collecting drivable ways")
    wc = WayCollector()
    wc.apply_file(PBF)
    log(f"  drivable ways: {len(wc.ways):,}")

    needed = set()
    for _, nodes, _ in wc.ways:
        needed.update(nodes)
    log(f"  unique nodes referenced: {len(needed):,}")

    log("PBF pass 2: node positions in bbox")
    nc = NodeCollector(needed)
    nc.apply_file(PBF)
    log(f"  nodes in bbox: {len(nc.nodes):,}")

    matsim_in_osm = sum(1 for n in matsim_node_set if n in nc.nodes)
    log(f"  matsim nodes also in OSM bbox: {matsim_in_osm:,}/{len(matsim_node_set):,}")

    log("Projecting nodes")
    proj = {nid: project(lat, lon) for nid, (lat, lon) in nc.nodes.items()}

    log("Building directed adjacency + edge set")
    adj = {}
    edges_set = set()
    for wid, nodes, oneway in wc.ways:
        if oneway == "-1":
            for i in range(len(nodes) - 1):
                a, b = nodes[i + 1], nodes[i]
                if a in nc.nodes and b in nc.nodes and a != b:
                    adj.setdefault(a, set()).add(b)
                    edges_set.add((a, b))
        elif oneway in ("yes", "true", "1"):
            for i in range(len(nodes) - 1):
                a, b = nodes[i], nodes[i + 1]
                if a in nc.nodes and b in nc.nodes and a != b:
                    adj.setdefault(a, set()).add(b)
                    edges_set.add((a, b))
        else:
            for i in range(len(nodes) - 1):
                a, b = nodes[i], nodes[i + 1]
                if a in nc.nodes and b in nc.nodes and a != b:
                    adj.setdefault(a, set()).add(b)
                    adj.setdefault(b, set()).add(a)
                    edges_set.add((a, b))
                    edges_set.add((b, a))
    log(f"  directed edges: {len(edges_set):,}, nodes with adj: {len(adj):,}")

    log("Walking degree-2 chains: matsim node -> next matsim node")
    edge_to_matsim = {}
    n_walked = n_link_found = n_link_missing = n_dead = 0
    MAX_STEPS = 1000
    for A in matsim_node_set:
        outs = adj.get(A)
        if outs is None:
            continue
        for first in list(outs):
            path = [(A, first)]
            prev, cur = A, first
            valid = True
            steps = 0
            while cur not in matsim_node_set:
                steps += 1
                if steps > MAX_STEPS:
                    valid = False
                    n_dead += 1
                    break
                cur_outs = adj.get(cur, set())
                cands = [n for n in cur_outs if n != prev]
                if len(cands) != 1:
                    valid = False
                    break
                nxt = cands[0]
                path.append((cur, nxt))
                prev, cur = cur, nxt
            if not valid or cur == A:
                continue
            n_walked += 1
            link_id = matsim_links.get((A, cur))
            if link_id is None:
                n_link_missing += 1
                continue
            n_link_found += 1
            for sub in path:
                edge_to_matsim[sub] = link_id

    log(
        f"  walks_ok={n_walked:,}, link_found={n_link_found:,}, "
        f"link_missing={n_link_missing:,}, dead={n_dead:,}"
    )
    log(
        f"  edges with matsim mapping: {len(edge_to_matsim):,} / "
        f"{len(edges_set):,} ({100.0*len(edge_to_matsim)/max(len(edges_set),1):.1f}%)"
    )
    matsim_links_used = len({lid for lid in edge_to_matsim.values()})
    log(
        f"  matsim car links covered: {matsim_links_used:,}/{len(matsim_links):,} "
        f"({100.0*matsim_links_used/max(len(matsim_links),1):.1f}%)"
    )

    data = {
        "proj_nodes": proj,
        "edges": sorted(edges_set),
        "edge_to_matsim": edge_to_matsim,
        "meta": {
            "matsim_links_total": len(matsim_links),
            "matsim_links_covered": matsim_links_used,
            "bbox": (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX),
            "drivable": sorted(DRIVABLE),
            "center": (CENTER_LAT, CENTER_LON),
        },
    }
    log(f"Pickling to {OUT} (protocol=4)")
    with open(OUT, "wb") as f:
        pickle.dump(data, f, protocol=4)
    sz = os.path.getsize(OUT) / 1e6
    log(f"Saved {OUT} ({sz:.1f} MB). total elapsed={time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
