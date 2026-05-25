"""Plot the all_1km cluster figure with dead-end links excluded.

Reads the saved 283-link boolean mask produced by
event_sept15_all_1km_no_dead.py and uses it to colour the road
network around Rice-Eccles. Side panel reflects the no-dead-end
numbers (event d2d / d2m default / d2m tuned / reduction / best 4D).
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
GAMMA = 0.20
ALPHA = 0.01


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
    print("[load] mask + results ...")
    mask = np.load(OUT / "all_1km_no_dead_mask.npy")
    with open(OUT / "all_1km_no_dead_results.json") as f:
        results = json.load(f)

    print("[load] graph + xml ...")
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
    ax.add_collection(LineCollection(
        fg, colors="#c0392b", linewidths=2.0, zorder=3,
        label=f"cluster ({int(mask.sum())} links, no dead-ends)"))
    ax.plot(STADIUM_LON, STADIUM_LAT, marker="*", markersize=22,
             markerfacecolor="#f1c40f", markeredgecolor="black",
             markeredgewidth=1.0, zorder=5, label="Rice-Eccles Stadium")
    theta = np.linspace(0, 2 * np.pi, 200)
    circ_lon = STADIUM_LON + CLUSTER_RADIUS_KM * lon_per_km * np.cos(theta)
    circ_lat = STADIUM_LAT + CLUSTER_RADIUS_KM * lat_per_km * np.sin(theta)
    ax.plot(circ_lon, circ_lat, color="#2c3e50", lw=0.8, ls=":", alpha=0.7,
             zorder=4, label=f"{CLUSTER_RADIUS_KM:.1f} km ring")
    ax.set_xlim(lon_lo, lon_hi)
    ax.set_ylim(lat_lo, lat_hi)
    ax.set_aspect(1.0 / cos_lat)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title("Salt Lake City road network around Rice-Eccles Stadium\n"
                  "Cluster: all links within 1 km, dead-ends (cul-de-sacs, "
                  "driveways, stubs) excluded",
                  fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # Side panel
    side = fig.add_axes([0.68, 0.05, 0.30, 0.90])
    side.axis("off")
    bp = results["tuned_best"]["params"]
    text = [
        "Cluster: all_1km_no_dead",
        f"  size               : {results['cluster_size']} links",
        f"  dead-ends dropped  : {results['n_dead_dropped']}",
        f"                       (from 305 raw)",
        f"  pendant nodes in graph: {results['n_pendant_nodes']:,}",
        "",
        "Event: Sept 8 -> Sept 15 (kickoff 8 PM)",
        "Window: 19:00-19:55 MDT (12 bins)",
        f"Diffusion: gamma={GAMMA}, alpha={ALPHA}",
        "",
        "MRE on the event pair",
        f"  data->data (noise floor) : {results['event_d2d']:.4f}",
        f"  data->model default chain: {results['event_d2m']:.4f}",
        f"  data->model tuned 4D     : {results['tuned_best']['mre']:.4f}",
        f"  reduction                : {results['reduction_pct']:.1f} %",
        "",
        "Best 4D parameters",
        f"  alpha_s = {bp['alpha_s']:+.2f}",
        f"  alpha_l = {bp['alpha_l']:+.2f}",
        f"  beta    = {bp['beta']:.2f}",
        f"  rho     = {bp['rho']:.2f}",
        "",
        "Reference (control pair, default)",
        f"  data->data : {results['control_d2d']:.4f}",
        f"  data->model: {results['control_d2m']:.4f}",
    ]
    side.text(0.0, 1.0, "\n".join(text), fontsize=10, family="monospace",
              va="top", ha="left", transform=side.transAxes,
              bbox=dict(facecolor="white", edgecolor="#999",
                         linewidth=0.6, pad=10))
    out_png = OUT / "fig_stadium_corridors_all_1km_no_dead.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
