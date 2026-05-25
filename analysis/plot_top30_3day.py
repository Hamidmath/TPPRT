"""Plot just the map of the top-30 busiest cluster (3-day ranking).
No side panel.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

OUT = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project/results/event_sept15")

STADIUM_LAT = 40.7596
STADIUM_LON = -111.8485
CLUSTER_RADIUS_KM = 1.0
VIEW_RADIUS_KM = 1.5


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


def main():
    mask = np.load(OUT / "top30_busiest_3day_with_deadends_mask.npy")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links_order = list(graph_data["links"].keys())
    nodes, links_ep = parse_matsim_xml(str(config.NETWORK_XML))

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
        if ep is None or ep[0] not in nodes or ep[1] not in nodes: continue
        lon_a, lat_a = nodes[ep[0]]; lon_b, lat_b = nodes[ep[1]]
        if (max(lon_a, lon_b) < lon_lo or min(lon_a, lon_b) > lon_hi or
            max(lat_a, lat_b) < lat_lo or min(lat_a, lat_b) > lat_hi):
            continue
        seg = [(lon_a, lat_a), (lon_b, lat_b)]
        (fg if mask[i] else bg).append(seg)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.add_collection(LineCollection(bg, colors="#7f7f7f", linewidths=0.5, zorder=1))
    ax.add_collection(LineCollection(
        fg, colors="#c0392b", linewidths=2.8, zorder=3,
        label="top-30 busiest"))
    ax.plot(STADIUM_LON, STADIUM_LAT, marker="*", markersize=22,
             markerfacecolor="#f1c40f", markeredgecolor="black",
             markeredgewidth=1.0, zorder=5, label="Rice-Eccles Stadium")
    theta = np.linspace(0, 2 * np.pi, 200)
    circ_lon = STADIUM_LON + CLUSTER_RADIUS_KM * lon_per_km * np.cos(theta)
    circ_lat = STADIUM_LAT + CLUSTER_RADIUS_KM * lat_per_km * np.sin(theta)
    ax.plot(circ_lon, circ_lat, color="black", lw=1.8, ls="--", alpha=0.95,
             zorder=4, label=f"{CLUSTER_RADIUS_KM:.1f} km ring")
    ax.set_xlim(lon_lo, lon_hi); ax.set_ylim(lat_lo, lat_hi)
    ax.set_aspect(1.0 / cos_lat)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    out_png = OUT / "fig_stadium_corridors_top30_3day.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
