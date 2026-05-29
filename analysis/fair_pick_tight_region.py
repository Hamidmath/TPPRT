"""Pick a tight sub-cluster of the 39 Fair-anomaly links inside a
~1 km road-network radius.

We have no coordinates in city_graph_full.json, only directed adjacency
and per-link lengths. So we approximate "within 1 km of a center" by
graph-weighted shortest-path distance on the *undirected* link graph,
with each edge's weight = (length(src) + length(dst)) / 2 (the road
distance you traverse moving from one link to the next).

Algorithm:
  1. Build the undirected, length-weighted link graph (nodes = links).
  2. For each candidate center c in the 39 region links, run Dijkstra
     with cutoff 1000 m. Collect the set of the 39 region links that
     fall within 1000 m of c.
  3. Pick the c that gives the largest such set. That set is the
     "tight region".

Output: documents/walkthrough2/results/anomalies/fair/tight_region.json
"""
import heapq
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRAPH_JSON = "data/city_graph_full.json"
REGION_FILE = ("documents/walkthrough2/results/anomalies/fair/"
                "region_components.json")
OUT = Path("documents/walkthrough2/results/anomalies/fair/tight_region.json")

RADIUS_M = 1000.0


def build_undirected(graph):
    """Undirected adjacency: nbr[lid] = {(other_lid, edge_weight), ...}"""
    nbr = {lid: {} for lid in graph["links"]}
    for src, succ_list in graph["adjacency"].items():
        if src not in nbr:
            continue
        src_len = graph["links"][src].get("length", 0.0)
        for dst in succ_list:
            if dst not in nbr:
                continue
            dst_len = graph["links"][dst].get("length", 0.0)
            w = 0.5 * (src_len + dst_len)
            if dst not in nbr[src] or w < nbr[src][dst]:
                nbr[src][dst] = w
                nbr[dst][src] = w
    return nbr


def dijkstra_within(nbr, source, cutoff):
    """Return dict lid -> distance for all lids within cutoff of source."""
    dist = {source: 0.0}
    h = [(0.0, source)]
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]:
            continue
        for v, w in nbr[u].items():
            nd = d + w
            if nd > cutoff:
                continue
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return dist


def main():
    print("[start] pick tight sub-region of the 39 Fair-anomaly links")
    graph = json.load(open(GRAPH_JSON))
    comps = json.load(open(REGION_FILE))
    region_lids = [str(x) for x in comps[0]["link_ids"]]
    region_set = set(region_lids)
    print(f"  candidate region: {len(region_lids)} links")

    nbr = build_undirected(graph)
    print(f"  undirected graph: {len(nbr):,} nodes")

    best = None
    for c in region_lids:
        dist = dijkstra_within(nbr, c, RADIUS_M)
        reached = [r for r in region_lids if r in dist]
        score = len(reached)
        if best is None or score > best["n_reached"]:
            best = dict(center=c, n_reached=score,
                         reached=reached, distances=dist)

    reached = best["reached"]
    print(f"\n  best center: link {best['center']}  "
          f"reaches {best['n_reached']} of {len(region_lids)} within "
          f"{RADIUS_M:.0f} m")
    dist = best["distances"]
    inset = [(r, dist[r]) for r in reached]
    inset.sort(key=lambda x: x[1])
    print(f"  links in tight sub-region (distance from center, m):")
    for r, d in inset:
        attrs = graph["links"].get(r, {})
        print(f"    {r:>6s}  d={d:6.0f}  lanes={attrs.get('lanes')}  "
              f"speed={attrs.get('speed'):.1f}  length={attrs.get('length'):.0f}")

    out = dict(
        center=best["center"],
        radius_m=RADIUS_M,
        n_reached=best["n_reached"],
        n_total=len(region_lids),
        tight_link_ids=reached,
        distances={r: float(d) for r, d in inset},
    )
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n[done] saved {OUT}")


if __name__ == "__main__":
    main()
