"""Produce a stadium-network figure for each of the four cluster
definitions evaluated in event_sept15_cluster_compare.py:

  - top25_1km      : top 25 most-trafficked links within 1 km
  - top50_1km      : top 50
  - top100_1km     : top 100
  - neighbors_3hop : 3-hop BFS expansion from links within 200 m

Each figure has the same layout as fig_stadium_corridors_1km.png:
network in gray, cluster links in red, stadium starred, side panel
with default MRE / tuned MRE / 4D-best parameters from the comparison
experiment.
"""
import json
import re
import sys
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz

OUT = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project/results/event_sept15")

# Same params used in the cluster comparison
STADIUM_LAT = 40.7596
STADIUM_LON = -111.8485
CLUSTER_RADIUS_KM = 1.0
NEIGHBOR_SEED_M = 200.0
NEIGHBOR_HOPS = 3
GAMMA = 0.20
ALPHA = 0.01
CORRIDOR_RANK_DAYS  = ["2018-09-01", "2018-09-22"]
CORRIDOR_RANK_START = "17:00:00"
CORRIDOR_RANK_END   = "23:55:00"

# Viewport
VIEW_RADIUS_KM = 1.5


# ---------- mask builders (copies of the experiment's logic) ----------
def parse_matsim_xml(xml_path):
    nodes, links_ep = {}, {}
    n_re = re.compile(r'<node\s+id="(\d+)"\s+x="(-?[\d.]+)"\s+y="(-?[\d.]+)"')
    l_re = re.compile(r'<link\s+id="(\d+)"\s+from="(\d+)"\s+to="(\d+)"')
    with open(xml_path) as f:
        for line in f:
            m = n_re.search(line)
            if m:
                nodes[m.group(1)] = (float(m.group(2)), float(m.group(3))); continue
            m = l_re.search(line)
            if m:
                links_ep[m.group(1)] = (m.group(2), m.group(3))
    return nodes, links_ep


def build_diffusion_P(graph_data, links_order):
    id_to_idx = {lid: i for i, lid in enumerate(links_order)}
    adj = graph_data.get('adjacency', {})
    N = len(links_order)
    row, col, data = [], [], []
    for i, lid in enumerate(links_order):
        succ = [id_to_idx[ol] for ol in adj.get(lid, []) if ol in id_to_idx]
        if not succ:
            continue
        w = 1.0 / len(succ)
        for j in succ:
            row.append(i); col.append(j); data.append(w)
    return csr_matrix((data, (row, col)), shape=(N, N), dtype=np.float64)


def link_midpoints(links_order, nodes, links_ep):
    N = len(links_order)
    mid = np.full((N, 2), np.nan, dtype=np.float64)
    for i, lid in enumerate(links_order):
        ep = links_ep.get(str(lid))
        if ep is None or ep[0] not in nodes or ep[1] not in nodes:
            continue
        lon_a, lat_a = nodes[ep[0]]
        lon_b, lat_b = nodes[ep[1]]
        mid[i] = [0.5 * (lon_a + lon_b), 0.5 * (lat_a + lat_b)]
    return mid


def km_distance(mid, lat0, lon0):
    R_km = 6371.0
    cos_lat = np.cos(np.deg2rad(lat0))
    dlat_km = (mid[:, 1] - lat0) * np.pi / 180.0 * R_km
    dlon_km = (mid[:, 0] - lon0) * np.pi / 180.0 * R_km * cos_lat
    return np.sqrt(dlat_km ** 2 + dlon_km ** 2)


def baseline_popularity(graph_data, links_order, N):
    """Recompute the mean diffused popularity over the two baseline Saturdays."""
    lid_to_idx = {lid: i for i, lid in enumerate(links_order)}
    P_diff = build_diffusion_P(graph_data, links_order)
    b = load_popularity_npz(str(config.DATA_DIR / "popularity_results_osm.npz"))
    matrix = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_link_ids = b["link_ids"]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_link_ids], dtype=np.int64)
    valid = proj >= 0
    time_index = {t: i for i, t in enumerate(times)}
    accum = np.zeros(N, dtype=np.float64); n = 0
    for day in CORRIDOR_RANK_DAYS:
        cur = datetime.strptime(f"{day} {CORRIDOR_RANK_START}", "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(f"{day} {CORRIDOR_RANK_END}",   "%Y-%m-%d %H:%M:%S")
        while cur <= end:
            ti = time_index.get(cur.strftime("%Y-%m-%d %H:%M:%S"))
            if ti is not None:
                row = matrix.getrow(ti).toarray().ravel().astype(np.float64)
                C = np.zeros(N, dtype=np.float64)
                np.add.at(C, proj[valid], row[valid])
                c = C + ALPHA
                c_diff = (1.0 - GAMMA) * c + GAMMA * (P_diff @ c)
                s = c_diff.sum()
                if s > 0:
                    accum += c_diff / s; n += 1
            cur += timedelta(minutes=5)
    accum /= max(n, 1)
    return accum


def build_topk_mask(ring_mask, pop, k):
    score = np.where(ring_mask, pop, -1.0)
    idx = np.argsort(score)[::-1][:k]
    m = np.zeros_like(ring_mask, dtype=bool); m[idx] = True
    return m


def build_neighbor_mask(graph_data, links_order, mid, seed_radius_m, n_hops):
    N = len(links_order)
    lid_to_idx = {lid: i for i, lid in enumerate(links_order)}
    adj_dict = graph_data.get('adjacency', {})
    out_adj = [[] for _ in range(N)]
    in_adj  = [[] for _ in range(N)]
    for lid, out_list in adj_dict.items():
        i = lid_to_idx.get(lid)
        if i is None: continue
        for next_lid in out_list:
            j = lid_to_idx.get(next_lid)
            if j is not None:
                out_adj[i].append(j); in_adj[j].append(i)
    d_km = km_distance(mid, STADIUM_LAT, STADIUM_LON)
    seed = np.where(d_km <= seed_radius_m / 1000.0)[0].tolist()
    visited = set(seed); frontier = set(seed)
    for _ in range(n_hops):
        nxt = set()
        for u in frontier:
            for v in out_adj[u]:
                if v not in visited: nxt.add(v); visited.add(v)
            for v in in_adj[u]:
                if v not in visited: nxt.add(v); visited.add(v)
        frontier = nxt
    m = np.zeros(N, dtype=bool)
    for i in visited: m[i] = True
    return m


# ---------- plot ----------
def plot_one(name, mask, links_order, nodes, links_ep, results_data, save_path):
    cos_lat = np.cos(np.deg2rad(STADIUM_LAT))
    lat_per_km = 1.0 / 111.0
    lon_per_km = 1.0 / (111.0 * cos_lat)
    lat_lo = STADIUM_LAT - VIEW_RADIUS_KM * lat_per_km
    lat_hi = STADIUM_LAT + VIEW_RADIUS_KM * lat_per_km
    lon_lo = STADIUM_LON - VIEW_RADIUS_KM * lon_per_km
    lon_hi = STADIUM_LON + VIEW_RADIUS_KM * lon_per_km

    bg, fg = [], []
    for i, lid in enumerate(links_order):
        ep = links_ep.get(str(lid))
        if ep is None or ep[0] not in nodes or ep[1] not in nodes:
            continue
        lon_a, lat_a = nodes[ep[0]]
        lon_b, lat_b = nodes[ep[1]]
        if (max(lon_a, lon_b) < lon_lo or min(lon_a, lon_b) > lon_hi or
            max(lat_a, lat_b) < lat_lo or min(lat_a, lat_b) > lat_hi):
            continue
        seg = [(lon_a, lat_a), (lon_b, lat_b)]
        (fg if mask[i] else bg).append(seg)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.04, 0.05, 0.62, 0.90])
    ax.add_collection(LineCollection(bg, colors="#c4c4c4", linewidths=0.4, zorder=1))
    ax.add_collection(LineCollection(fg, colors="#c0392b", linewidths=2.0,
                                       zorder=3, label=f"cluster ({int(mask.sum())} links)"))
    ax.plot(STADIUM_LON, STADIUM_LAT, marker="*", markersize=22,
             markerfacecolor="#f1c40f", markeredgecolor="black", markeredgewidth=1.0,
             zorder=5, label="Rice-Eccles Stadium")
    # Optional reference circle: 1 km ring (or seed radius for neighbour cluster)
    if name == "neighbors_3hop":
        rad = NEIGHBOR_SEED_M / 1000.0
        lbl = f"{NEIGHBOR_SEED_M:.0f} m seed radius"
    else:
        rad = CLUSTER_RADIUS_KM
        lbl = f"{CLUSTER_RADIUS_KM:.1f} km candidate ring"
    theta = np.linspace(0, 2 * np.pi, 200)
    circ_lon = STADIUM_LON + rad * lon_per_km * np.cos(theta)
    circ_lat = STADIUM_LAT + rad * lat_per_km * np.sin(theta)
    ax.plot(circ_lon, circ_lat, color="#2c3e50", lw=0.8, ls=":", alpha=0.7,
             zorder=4, label=lbl)

    ax.set_xlim(lon_lo, lon_hi)
    ax.set_ylim(lat_lo, lat_hi)
    ax.set_aspect(1.0 / cos_lat)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    titles = dict(
        top25_1km      = "Top-25 most-trafficked links within 1 km",
        top50_1km      = "Top-50 most-trafficked links within 1 km",
        top100_1km     = "Top-100 most-trafficked links within 1 km",
        all_1km        = "All links within 1 km (no top-K filter)",
        neighbors_3hop = "3-hop neighbour cluster from 200 m seed",
    )
    ax.set_title(f"Salt Lake City road network around Rice-Eccles Stadium\n"
                  f"Cluster: {titles[name]}",
                  fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # side panel
    side = fig.add_axes([0.68, 0.05, 0.30, 0.90])
    side.axis("off")
    # Pull numbers for THIS cluster from results JSON
    e1_event = results_data["experiment_1_default"]["event"][name]
    e1_ctrl  = results_data["experiment_1_default"]["control"][name]
    best     = results_data["experiment_3_tuned_4d"]["best_per_cluster"][name]
    default_mre = e1_event["d2m_mean"]
    tuned_mre   = best["mre"]
    red = (1 - tuned_mre / default_mre) * 100.0 if default_mre > 0 else float("nan")
    bp = best["params"]
    text_lines = [
        f"Cluster: {name}",
        f"  size               : {int(mask.sum())} links",
        "",
        "Event: Sept 8 -> Sept 15 (kickoff 8 PM)",
        "Window: 19:00-19:55 MDT (12 bins)",
        f"Diffusion: gamma={GAMMA}, alpha={ALPHA}",
        "",
        "data->model MRE on event pair",
        f"  default chain      : {default_mre:.4f}",
        f"  tuned 4D best      : {tuned_mre:.4f}",
        f"  reduction          : {red:.1f} %",
        "",
        "Best 4D parameters",
        f"  alpha_s = {bp['alpha_s']:+.2f}",
        f"  alpha_l = {bp['alpha_l']:+.2f}",
        f"  beta    = {bp['beta']:.2f}",
        f"  rho     = {bp['rho']:.2f}",
        "",
        "Reference (control pair, default)",
        f"  data->data corridor MRE: {e1_ctrl['d2d_mean']:.4f}",
        f"  data->model corridor MRE: {e1_ctrl['d2m_mean']:.4f}",
    ]
    side.text(0.0, 1.0, "\n".join(text_lines), fontsize=10, family="monospace",
              va="top", ha="left", transform=side.transAxes,
              bbox=dict(facecolor="white", edgecolor="#999",
                         linewidth=0.6, pad=10))
    fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {save_path}")


def main():
    with open(OUT / "results_cluster_compare.json") as f:
        results = json.load(f)

    print("[load] graph_data ...")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links_order = list(graph_data['links'].keys())
    N = len(links_order)

    print("[load] parsing XML ...")
    nodes, links_ep = parse_matsim_xml(str(config.NETWORK_XML))
    mid = link_midpoints(links_order, nodes, links_ep)

    print("[compute] baseline popularity ...")
    pop = baseline_popularity(graph_data, links_order, N)

    ring_1km = km_distance(mid, STADIUM_LAT, STADIUM_LON) <= CLUSTER_RADIUS_KM
    masks = {
        "top25_1km":      build_topk_mask(ring_1km, pop, 25),
        "top50_1km":      build_topk_mask(ring_1km, pop, 50),
        "top100_1km":     build_topk_mask(ring_1km, pop, 100),
        "all_1km":        ring_1km.copy(),
        "neighbors_3hop": build_neighbor_mask(graph_data, links_order, mid,
                                                NEIGHBOR_SEED_M, NEIGHBOR_HOPS),
    }
    for name, mask in masks.items():
        out_path = OUT / f"fig_stadium_corridors_{name}.png"
        plot_one(name, mask, links_order, nodes, links_ep, results, out_path)


if __name__ == "__main__":
    main()
