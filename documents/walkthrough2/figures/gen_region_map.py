"""Geographic map of the 20-link anomaly region in Cottonwood Heights.

Style: tiered road background (residential / arterial / primary),
thin highlight for the 20 anomaly links, light publication palette.
"""
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NETWORK_XML = str(ROOT / "data/slc_network.xml")
OUT = Path(__file__).parent / "region_map.pdf"

REGION_IDS = {
    "77400","59062","59055","77398","85822","3614","77405","77404",
    "77401","59063","77396","77395","85820","77407","59065","59064",
    "59049","77391","59068","59059",
}

# Center = midpoint of link 59063 (the highest-lift link, used as the
# Dijkstra anchor for the 1 km neighborhood that defines the region).
CENTER_LAT = (40.63660 + 40.63749) / 2  # = 40.637045
CENTER_LON = (-111.80697 + -111.80872) / 2  # = -111.807845
# Used for cropping the plot window; matches the center.
CENTROID_LAT = CENTER_LAT
CENTROID_LON = CENTER_LON
COL_W = 3.35

def parse_network(xml_path, region_ids):
    nodes = {}
    region_segs = []
    bg_segs = []  # list of (seg, freespeed)
    for ev, el in ET.iterparse(xml_path, events=("start",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("x")), float(el.get("y")))
        elif el.tag == "link":
            f, t = el.get("from"), el.get("to")
            if f in nodes and t in nodes:
                seg = (nodes[f], nodes[t])
                fs = float(el.get("freespeed", 0.0))
                if el.get("id") in region_ids:
                    region_segs.append((seg, fs))
                else:
                    bg_segs.append((seg, fs))
            el.clear()
    return nodes, region_segs, bg_segs


def main():
    nodes, region_segs, bg_segs = parse_network(NETWORK_XML, REGION_IDS)

    # Project (lon, lat) to local meters around the center, so 1 m on
    # the x axis equals 1 m on the y axis and a 1 km radius is a true
    # circle on the plot.
    lat0_rad = np.deg2rad(CENTER_LAT)
    K_LAT = 111_000.0
    K_LON = 111_000.0 * np.cos(lat0_rad)
    def to_xy(lon, lat):
        return ((lon - CENTER_LON) * K_LON,
                (lat - CENTER_LAT) * K_LAT)

    HALF = 1300.0  # meters; the plot window is +/- HALF on each axis
    def in_box_xy(seg):
        (x1, y1), (x2, y2) = seg
        a = to_xy(x1, y1); b = to_xy(x2, y2)
        return (abs(a[0]) <= HALF and abs(a[1]) <= HALF
                 and abs(b[0]) <= HALF and abs(b[1]) <= HALF)

    bg_visible = [(s, fs) for s, fs in bg_segs if in_box_xy(s)]
    region_visible = [(s, fs) for s, fs in region_segs if in_box_xy(s)]
    print(f"bg in-box={len(bg_visible):,}  region in-box={len(region_visible)}")

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.95))
    ax.set_facecolor("#fbfbfb")

    # All background links: same color and line width.
    for seg, fs in bg_visible:
        (xa, ya) = to_xy(*seg[0]); (xb, yb) = to_xy(*seg[1])
        ax.plot([xa, xb], [ya, yb], color="#bdbdbd", lw=0.5,
                 zorder=1, solid_capstyle="round")

    # 20 region links: slightly heavier than background, red.
    endpoint_set = set()
    for seg, fs in region_visible:
        a, b = to_xy(*seg[0]), to_xy(*seg[1])
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#c92420", lw=0.5,
                 zorder=5, solid_capstyle="round")
        endpoint_set.add((round(a[0], 2), round(a[1], 2)))
        endpoint_set.add((round(b[0], 2), round(b[1], 2)))

    # Small dots on each unique endpoint of the region links.
    if endpoint_set:
        xs, ys = zip(*endpoint_set)
        ax.scatter(xs, ys, s=4.5, color="#c92420",
                    edgecolor="none", zorder=6)

    # 1 km radius circle (true circle now, since axes are in meters).
    from matplotlib.patches import Circle
    ax.add_patch(Circle((0, 0), radius=1000.0,
                         fill=False, edgecolor="#1f77b4",
                         linestyle=(0, (4, 2)), lw=0.9, zorder=4,
                         alpha=0.9))


    ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#888888"); spine.set_linewidth(0.6)

    # North arrow (axes coords)
    nx, ny = 0.93, 0.93
    ax.annotate("N", xy=(nx, ny), xycoords="axes fraction",
                 fontsize=7.5, ha="center", va="center", fontweight="bold")
    ax.annotate("", xy=(nx, ny - 0.02), xytext=(nx, ny - 0.10),
                 xycoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="black"))


    # Legend in the lower-right.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#c92420", lw=0.5,
                marker="o", markersize=2.5, markevery=[0, 1],
                markerfacecolor="#c92420", markeredgecolor="none",
                label="20 top-busy links"),
        Line2D([0], [0], color="#1f77b4", lw=0.9,
                linestyle=(0, (4, 2)), label="1 km radius"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False,
               fontsize=5.5, handlelength=1.2, borderaxespad=0.4)

    fig.tight_layout(pad=0.2)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
