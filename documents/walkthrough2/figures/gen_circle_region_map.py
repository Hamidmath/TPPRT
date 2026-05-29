"""Zoomed-in map around the chosen 2 km circle in SLC.

Renders only the area around the circle (not the whole network),
highlights the 15 top-difference links inside it, and tints the
70 mph freeway segments lightly so the I-15 corridor is visible.
"""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NETWORK_XML = str(ROOT / "data/slc_network.xml")
OUT = Path(__file__).parent / "circle_region_map.pdf"

CENTER_LAT = 40.722348
CENTER_LON = -111.904691
R_KM = 2.0

# The 15 top-200 links whose midpoints sit inside the 2 km circle
# (verified earlier; see analysis log).
REGION15 = ["58905","58889","58864","58870","58898","58888","58869","58862",
            "59159","59261","59071","59157","58874","58923","59260"]

COL_W = 3.35


def main():
    region_set = set(REGION15)
    print(f"highlight region: {len(region_set)} links")

    lat0_rad = np.deg2rad(CENTER_LAT)
    K_LAT = 111_000.0
    K_LON = 111_000.0 * np.cos(lat0_rad)
    def to_xy(lon, lat):
        return ((lon - CENTER_LON) * K_LON,
                (lat - CENTER_LAT) * K_LAT)

    HALF = 2400.0  # metres; plot window ~ 4.8 km square

    nodes = {}
    region_segs = []
    bg_segs = []
    for ev, el in ET.iterparse(NETWORK_XML, events=("start",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("x")), float(el.get("y")))
        elif el.tag == "link":
            f, t = el.get("from"), el.get("to")
            if f in nodes and t in nodes:
                xa, ya = to_xy(*nodes[f])
                xb, yb = to_xy(*nodes[t])
                if abs(xa) > HALF or abs(ya) > HALF: el.clear(); continue
                if abs(xb) > HALF or abs(yb) > HALF: el.clear(); continue
                if el.get("id") in region_set:
                    region_segs.append(((xa, ya), (xb, yb)))
                else:
                    bg_segs.append(((xa, ya), (xb, yb)))
            el.clear()
    print(f"  segs in window: top-15 = {len(region_segs)}, "
          f"bg = {len(bg_segs):,}")

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.95))
    ax.set_facecolor("white")

    # All background streets (uniform grey, no freeway tint)
    for (xa, ya), (xb, yb) in bg_segs:
        ax.plot([xa, xb], [ya, yb], color="#d3d3d3", lw=0.4,
                 zorder=1, solid_capstyle="round")
    # 15 highlighted links + endpoint dots
    endpoints = set()
    for (xa, ya), (xb, yb) in region_segs:
        ax.plot([xa, xb], [ya, yb], color="#c92420", lw=1.4,
                 zorder=4, solid_capstyle="round")
        endpoints.add((round(xa, 1), round(ya, 1)))
        endpoints.add((round(xb, 1), round(yb, 1)))
    if endpoints:
        xs, ys = zip(*endpoints)
        ax.scatter(xs, ys, s=5, color="#c92420",
                    edgecolor="none", zorder=5)

    # 2 km radius circle (true circle in metric projection)
    ax.add_patch(Circle((0, 0), radius=R_KM * 1000.0,
                         fill=False, edgecolor="#1f77b4",
                         linestyle=(0, (4, 2)), lw=1.0, zorder=3))

    ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#888888"); spine.set_linewidth(0.6)

    # Legend
    handles = [
        Line2D([0], [0], color="#c92420", lw=1.4,
                marker="o", markersize=3, markevery=[0, 1],
                markerfacecolor="#c92420", markeredgecolor="none",
                label=f"{len(REGION15)} top-difference links"),
        Line2D([0], [0], color="#1f77b4", lw=1.0,
                linestyle=(0, (4, 2)),
                label=f"{R_KM:.0f} km radius"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False,
               fontsize=6.5, handlelength=2.0, borderaxespad=0.3)

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"  saved {OUT}")


if __name__ == "__main__":
    main()
