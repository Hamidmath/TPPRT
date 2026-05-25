"""Plot the Salt Lake City road network around Rice-Eccles Stadium,
highlight the 100 access-corridor links used in the event-day forecast
experiment, and annotate the figure with the analysis numbers
(default chain corridor MRE, tuned MRE, and the best 4D parameters).

The corridor cluster is identified by the same rule used in the
experiment: within 3 km of the stadium, ranked by mean diffused
popularity over baseline Saturday evenings (Sept 1 and Sept 22,
17:00-23:55), keep the top 100.
"""
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.sparse import csr_matrix, diags

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz

OUT = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project/results/event_sept15")
OUT.mkdir(parents=True, exist_ok=True)

# Stadium location and viewport
STADIUM_LAT = 40.7596
STADIUM_LON = -111.8485
import os as _os
CLUSTER_RADIUS_KM = float(_os.environ.get("RADIUS_KM", 1.0))
CORRIDOR_TOP_K = 100
# Viewport is a bit wider than the cluster so the boundary is visible
VIEW_RADIUS_KM = CLUSTER_RADIUS_KM + 0.5

# Corridor ranking window: baseline Saturday evenings
CORRIDOR_RANK_DAYS  = ["2018-09-01", "2018-09-22"]
CORRIDOR_RANK_START = "17:00:00"
CORRIDOR_RANK_END   = "23:55:00"
GAMMA = 0.20
ALPHA = 0.01

# Experiment numbers are loaded from the corresponding JSON to match
# whichever radius the plot is being made at.
def _load_results_for_radius(radius_km):
    p = OUT / f"results_diffused_{int(radius_km)}km.json"
    if not p.exists():
        # fall back to canonical name
        p = OUT / "results_diffused.json"
    with open(p) as f:
        data = json.load(f)
    e1 = data["experiment_1_default"]["event"]
    e3 = data["experiment_3_tuned_4d"]
    return dict(
        n_total_links = 99716,
        cluster_size  = data.get("cluster_size", 100),
        radius_km     = radius_km,
        window        = "19:00-19:55 MDT, Sept 15 2018 (1 h pre-kickoff)",
        n_bins        = e1["n_bins"],
        default_corridor_mre = e1["d2m_cluster_mean"],
        tuned_corridor_mre   = e3["best_cluster"]["mre"],
        reduction_pct        = 100.0 * (1 - e3["best_cluster"]["mre"] / e1["d2m_cluster_mean"]),
        default_overall_mre  = e1["d2m_overall_mean"],
        tuned_overall_mre    = e3["best_overall"]["mre"],
        overall_reduction_pct = 100.0 * (1 - e3["best_overall"]["mre"] / e1["d2m_overall_mean"]),
        best_corridor = dict(alpha_s=e3["best_cluster"]["alpha_s"],
                              alpha_l=e3["best_cluster"]["alpha_l"],
                              beta=e3["best_cluster"]["beta"],
                              rho=e3["best_cluster"]["rho"]),
        best_overall  = dict(alpha_s=e3["best_overall"]["alpha_s"],
                              alpha_l=e3["best_overall"]["alpha_l"],
                              beta=e3["best_overall"]["beta"],
                              rho=e3["best_overall"]["rho"]),
        diffusion     = dict(gamma=GAMMA, alpha=ALPHA),
    )

EXPT_NUMBERS = _load_results_for_radius(CLUSTER_RADIUS_KM)


# ---------- parsing the MATSim XML ----------
def parse_matsim_xml(xml_path):
    nodes = {}
    link_endpoints = {}
    node_re = re.compile(r'<node\s+id="(\d+)"\s+x="(-?[\d.]+)"\s+y="(-?[\d.]+)"')
    link_re = re.compile(r'<link\s+id="(\d+)"\s+from="(\d+)"\s+to="(\d+)"')
    with open(xml_path) as f:
        for line in f:
            m = node_re.search(line)
            if m:
                nodes[m.group(1)] = (float(m.group(2)), float(m.group(3)))
                continue
            m = link_re.search(line)
            if m:
                link_endpoints[m.group(1)] = (m.group(2), m.group(3))
    return nodes, link_endpoints


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


def corridor_indices(graph_data, links_order, nodes, link_endpoints):
    """Reproduce the corridor cluster used in the experiment."""
    # 1. Geometric ring
    cos_lat = np.cos(np.deg2rad(STADIUM_LAT))
    R_km = 6371.0
    N = len(links_order)
    ring = np.zeros(N, dtype=bool)
    mids = np.full((N, 2), np.nan, dtype=np.float64)
    for i, lid in enumerate(links_order):
        ep = link_endpoints.get(str(lid))
        if ep is None or ep[0] not in nodes or ep[1] not in nodes:
            continue
        lon_a, lat_a = nodes[ep[0]]
        lon_b, lat_b = nodes[ep[1]]
        mid_lon = 0.5 * (lon_a + lon_b)
        mid_lat = 0.5 * (lat_a + lat_b)
        mids[i] = [mid_lon, mid_lat]
        dlat_km = (mid_lat - STADIUM_LAT) * np.pi / 180.0 * R_km
        dlon_km = (mid_lon - STADIUM_LON) * np.pi / 180.0 * R_km * cos_lat
        if dlat_km ** 2 + dlon_km ** 2 <= CLUSTER_RADIUS_KM ** 2:
            ring[i] = True
    # 2. Rank by mean diffused popularity over baseline evenings
    print(f"[ring] {int(ring.sum())} links within {CLUSTER_RADIUS_KM} km")
    P_diff = build_diffusion_P(graph_data, links_order)
    b = load_popularity_npz(str(config.DATA_DIR / "popularity_results_osm.npz"))
    matrix = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_link_ids = b["link_ids"]
    lid_to_idx = {lid: i for i, lid in enumerate(links_order)}
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_link_ids],
                     dtype=np.int64)
    valid = proj >= 0
    time_index = {t: i for i, t in enumerate(times)}
    accum = np.zeros(N, dtype=np.float64)
    n_bins_used = 0
    for day in CORRIDOR_RANK_DAYS:
        cur = datetime.strptime(f"{day} {CORRIDOR_RANK_START}",
                                 "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(f"{day} {CORRIDOR_RANK_END}",
                                 "%Y-%m-%d %H:%M:%S")
        while cur <= end:
            tstr = cur.strftime("%Y-%m-%d %H:%M:%S")
            ti = time_index.get(tstr)
            if ti is not None:
                row = matrix.getrow(ti).toarray().ravel().astype(np.float64)
                C = np.zeros(N, dtype=np.float64)
                np.add.at(C, proj[valid], row[valid])
                c = C + ALPHA
                c_diff = (1.0 - GAMMA) * c + GAMMA * (P_diff @ c)
                s = c_diff.sum()
                if s > 0:
                    accum += c_diff / s
                    n_bins_used += 1
            cur += timedelta(minutes=5)
    accum /= max(n_bins_used, 1)
    in_ring = np.where(ring, accum, -1.0)
    top_idx = np.argsort(in_ring)[::-1][:CORRIDOR_TOP_K]
    mask = np.zeros(N, dtype=bool)
    mask[top_idx] = True
    return mask, mids


def main():
    print("[load] parsing slc_network.xml ...")
    nodes, link_endpoints = parse_matsim_xml(str(config.NETWORK_XML))
    print(f"  parsed {len(nodes):,} nodes, {len(link_endpoints):,} links")

    print("[load] graph_data ...")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links_order = list(graph_data['links'].keys())

    print("[cluster] identifying 100 corridor links ...")
    cluster_mask, mids = corridor_indices(graph_data, links_order,
                                           nodes, link_endpoints)
    print(f"  cluster size = {int(cluster_mask.sum())}")

    # Compute the viewport limits
    cos_lat = np.cos(np.deg2rad(STADIUM_LAT))
    lat_per_km = 1.0 / 111.0
    lon_per_km = 1.0 / (111.0 * cos_lat)
    lat_lo = STADIUM_LAT - VIEW_RADIUS_KM * lat_per_km
    lat_hi = STADIUM_LAT + VIEW_RADIUS_KM * lat_per_km
    lon_lo = STADIUM_LON - VIEW_RADIUS_KM * lon_per_km
    lon_hi = STADIUM_LON + VIEW_RADIUS_KM * lon_per_km

    # Build line segments for non-corridor (visible) and corridor links
    bg_segs, fg_segs = [], []
    in_box = 0
    for i, lid in enumerate(links_order):
        ep = link_endpoints.get(str(lid))
        if ep is None or ep[0] not in nodes or ep[1] not in nodes:
            continue
        lon_a, lat_a = nodes[ep[0]]
        lon_b, lat_b = nodes[ep[1]]
        # Skip links entirely outside the viewport (cheap test)
        if (max(lon_a, lon_b) < lon_lo or min(lon_a, lon_b) > lon_hi or
            max(lat_a, lat_b) < lat_lo or min(lat_a, lat_b) > lat_hi):
            continue
        in_box += 1
        seg = [(lon_a, lat_a), (lon_b, lat_b)]
        if cluster_mask[i]:
            fg_segs.append(seg)
        else:
            bg_segs.append(seg)
    print(f"  links visible in viewport: {in_box:,} "
          f"(bg={len(bg_segs):,}, fg={len(fg_segs):,})")

    # Plot
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.04, 0.05, 0.62, 0.90])

    bg_lc = LineCollection(bg_segs, colors="#c4c4c4", linewidths=0.4,
                            zorder=1)
    fg_lc = LineCollection(fg_segs, colors="#c0392b", linewidths=2.0,
                            zorder=3, label="100 access corridors")
    ax.add_collection(bg_lc)
    ax.add_collection(fg_lc)

    # Stadium marker
    ax.plot(STADIUM_LON, STADIUM_LAT, marker="*", markersize=22,
             markerfacecolor="#f1c40f", markeredgecolor="black",
             markeredgewidth=1.0, zorder=5, label="Rice-Eccles Stadium")
    # Cluster radius circle (3 km)
    theta = np.linspace(0, 2 * np.pi, 200)
    circ_lon = STADIUM_LON + CLUSTER_RADIUS_KM * lon_per_km * np.cos(theta)
    circ_lat = STADIUM_LAT + CLUSTER_RADIUS_KM * lat_per_km * np.sin(theta)
    ax.plot(circ_lon, circ_lat, color="#2c3e50", lw=0.8, ls=":", alpha=0.6,
             zorder=4, label=f"{CLUSTER_RADIUS_KM:.0f} km radius (candidate set)")

    ax.set_xlim(lon_lo, lon_hi)
    ax.set_ylim(lat_lo, lat_hi)
    ax.set_aspect(1.0 / cos_lat)  # equal-distance aspect at this latitude
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Salt Lake City road network around Rice-Eccles Stadium\n"
                  "Red = 100 access-corridor links used in the event forecast",
                  fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # --- side annotation panel ---
    side = fig.add_axes([0.68, 0.05, 0.30, 0.90])
    side.axis("off")
    text_lines = [
        "Event-day forecast experiment",
        "Utah vs. Washington, Sat Sept 15 2018, kickoff 8 PM MDT",
        "",
        "Setup",
        f"  Network         : {EXPT_NUMBERS['n_total_links']:,} road links",
        f"  Corridor cluster: {EXPT_NUMBERS['cluster_size']} links",
        f"                    (top by mean baseline traffic within",
        f"                    {EXPT_NUMBERS['radius_km']:.0f} km of stadium)",
        f"  Window          : {EXPT_NUMBERS['window']}",
        f"                    ({EXPT_NUMBERS['n_bins']} five-minute bins)",
        f"  Diffusion       : gamma = {EXPT_NUMBERS['diffusion']['gamma']}, "
        f"alpha = {EXPT_NUMBERS['diffusion']['alpha']}",
        "  Forecast        : chain(Sept 8 prior) vs Sept 15 truth",
        "",
        "Result: corridor MRE",
        f"  default chain      : {EXPT_NUMBERS['default_corridor_mre']:.2f}",
        f"  best tuned (4D)    : {EXPT_NUMBERS['tuned_corridor_mre']:.2f}",
        f"  reduction          : {EXPT_NUMBERS['reduction_pct']:.1f} %",
        "",
        "Best 4D corridor parameters",
        f"  alpha_s = {EXPT_NUMBERS['best_corridor']['alpha_s']:+.2f}",
        f"  alpha_l = {EXPT_NUMBERS['best_corridor']['alpha_l']:+.2f}",
        f"  beta    = {EXPT_NUMBERS['best_corridor']['beta']:.2f}",
        f"  rho     = {EXPT_NUMBERS['best_corridor']['rho']:.2f}",
        "",
        "For reference: all-links MRE",
        f"  default      : {EXPT_NUMBERS['default_overall_mre']:.3f}",
        f"  tuned        : {EXPT_NUMBERS['tuned_overall_mre']:.3f}",
        f"  reduction    : {EXPT_NUMBERS['overall_reduction_pct']:.1f} %",
        f"  best params  : alpha_s = {EXPT_NUMBERS['best_overall']['alpha_s']:+.2f},",
        f"                 alpha_l = {EXPT_NUMBERS['best_overall']['alpha_l']:+.2f},",
        f"                 beta    = {EXPT_NUMBERS['best_overall']['beta']:.2f},",
        f"                 rho     = {EXPT_NUMBERS['best_overall']['rho']:.2f}",
    ]
    side.text(0.0, 1.0, "\n".join(text_lines),
              fontsize=10, family="monospace",
              va="top", ha="left",
              transform=side.transAxes,
              bbox=dict(facecolor="white", edgecolor="#999",
                         linewidth=0.6, pad=10))

    out_png = OUT / f"fig_stadium_corridors_{int(CLUSTER_RADIUS_KM)}km.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight",
                facecolor="white")
    # also keep a canonical filename
    canon = OUT / "fig_stadium_corridors.png"
    fig.savefig(canon, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_png}")
    print(f"saved {canon}")


if __name__ == "__main__":
    main()
