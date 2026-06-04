"""I-15 NB crash event-link map (single-panel, no screenshot).

Event:    rear-end pile-up on northbound I-15 near 14400 S, Draper /
          South Jordan border, on 2018-09-20 (Thursday morning).
Crash pt: (40.489415, -111.893172) [user-supplied].

Same colour table / line widths / aspect handling as the festival
gen_festival_event_links.py left panel: background road graph in
grey, freeways in light blue, the eval-box surroundings in orange,
the one-hop graph neighbors of the closed set in the same orange
(non-expanded mode) or in red (expanded mode), the seed closure in
red, and a green marker at the crash coordinates. Eval box drawn as
a dashed black rectangle.
"""
import json
import os
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NETWORK_XML = str(ROOT / "data/slc_network.xml")
GRAPH_JSON  = str(ROOT / "data/city_graph_full.json")

EXPAND_NEIGHBORS = os.environ.get("TPPR_EXPAND", "0") == "1"
SUFFIX = "_expanded" if EXPAND_NEIGHBORS else ""
OUT_PDF = Path(__file__).parent / f"i15_crash_event_links{SUFFIX}.pdf"
OUT_PNG = Path(__file__).parent / f"i15_crash_event_links{SUFFIX}.png"

# Crash point (user-supplied: 40.489415, -111.893172). Snapped to
# the closest point on the closed NB I-15 link 59076 so the marker
# sits exactly on the red line; the physical crash lies a few
# tens of metres east of the link's centreline at this latitude.
CRASH_LAT, CRASH_LON = 40.489741, -111.894147

# Closure box for picking the affected I-15 NB segments. The SLC
# network represents I-15 here as a single ~1.3 km aggregated link
# (lid 59076) running from (40.4877, -111.8948) to (40.4992,
# -111.8910); the crash point sits inside that span. The box is
# sized so that link's midpoint (40.4935, -111.8929) falls inside,
# and the orientation + freespeed filters (|dy| > 1.5*|dx|, dy > 0,
# fs >= 31 m/s) pin the seed to the NB carriageway.
CLOSURE_LAT_LO = 40.486
CLOSURE_LAT_HI = 40.501
CLOSURE_LON_LO = -111.895
CLOSURE_LON_HI = -111.890

# Evaluation box. Centred at (40.492575, -111.891102) on I-15 NB
# just north of the crash. Lon half-extent shrunk 10% from the
# prior version (left and right sides pulled in), and the top
# (north) lat half-extent shrunk 3%. South side and centre lat
# unchanged. Span: ~3.45 km N-S, ~3.21 km E-W.
BOX_LAT_LO = 40.476825
BOX_LAT_HI = 40.507853
BOX_LON_LO = -111.910002
BOX_LON_HI = -111.872202

# Plot frame. A little larger than the eval box on every side.
FRAME_LAT_LO = 40.469
FRAME_LAT_HI = 40.515
FRAME_LON_LO = -111.918
FRAME_LON_HI = -111.864


def main():
    print("[start] i15 crash event-link single-panel")

    print("[load] adjacency for graph neighbors ...")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    adj = graph.get("adjacency", {})

    print("[scan] network XML for plot segments + closed/box masks ...")
    nodes = {}
    for _, el in ET.iterparse(NETWORK_XML, events=("start",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("x")), float(el.get("y")))
        elif el.tag == "link":
            el.clear()

    seed_lids = set()
    for _, el in ET.iterparse(NETWORK_XML, events=("start",)):
        if el.tag != "link":
            el.clear()
            continue
        f, t = el.get("from"), el.get("to")
        if f not in nodes or t not in nodes:
            el.clear(); continue
        x1, y1 = nodes[f]; x2, y2 = nodes[t]
        dx, dy = x2 - x1, y2 - y1
        mlat = 0.5 * (y1 + y2); mlon = 0.5 * (x1 + x2)
        fs = float(el.get("freespeed", 0.0))
        lid = el.get("id")
        in_box = (
            CLOSURE_LAT_LO <= mlat <= CLOSURE_LAT_HI
            and CLOSURE_LON_LO <= mlon <= CLOSURE_LON_HI
        )
        nb_freeway = (
            fs >= 31.0
            and abs(dy) > 1.5 * abs(dx)
            and dy > 0
        )
        if in_box and nb_freeway:
            seed_lids.add(lid)
        el.clear()

    neighbor_lids = set()
    for lid in seed_lids:
        for ol in adj.get(lid, []):
            neighbor_lids.add(ol)
    for src, outs in adj.items():
        if any(o in seed_lids for o in outs):
            neighbor_lids.add(src)
    neighbor_lids -= seed_lids

    closed_segs, neighbor_segs, box_segs, fw_segs, bg_segs = [], [], [], [], []
    for _, el in ET.iterparse(NETWORK_XML, events=("start",)):
        if el.tag != "link":
            el.clear()
            continue
        f, t = el.get("from"), el.get("to")
        if f not in nodes or t not in nodes:
            el.clear(); continue
        x1, y1 = nodes[f]; x2, y2 = nodes[t]
        mlat = 0.5 * (y1 + y2); mlon = 0.5 * (x1 + x2)
        fs = float(el.get("freespeed", 0.0))
        lid = el.get("id")
        seg = ((x1, y1), (x2, y2))
        in_box = (
            BOX_LAT_LO <= mlat <= BOX_LAT_HI
            and BOX_LON_LO <= mlon <= BOX_LON_HI
        )
        if lid in seed_lids:
            closed_segs.append(seg)
        elif lid in neighbor_lids:
            neighbor_segs.append(seg)
        elif in_box:
            box_segs.append(seg)
        elif fs >= 31.0:
            fw_segs.append(seg)
        else:
            bg_segs.append(seg)
        el.clear()

    print(f"  seed closed={len(closed_segs)}  "
          f"neighbors={len(neighbor_segs)}  "
          f"box(no-closed)={len(box_segs)}  "
          f"freeway={len(fw_segs):,}  bg={len(bg_segs):,}")

    lat_lo, lat_hi = FRAME_LAT_LO, FRAME_LAT_HI
    lon_lo, lon_hi = FRAME_LON_LO, FRAME_LON_HI

    fig, axL = plt.subplots(1, 1, figsize=(6.5, 5.6))
    axL.set_facecolor("white")
    for (x1, y1), (x2, y2) in bg_segs:
        axL.plot([x1, x2], [y1, y2], color="#d6d6d6", lw=0.30,
                  zorder=1, solid_capstyle="round")
    for (x1, y1), (x2, y2) in fw_segs:
        axL.plot([x1, x2], [y1, y2], color="#a9c5e4", lw=0.9,
                  zorder=2, solid_capstyle="round")
    for (x1, y1), (x2, y2) in box_segs:
        axL.plot([x1, x2], [y1, y2], color="#e8a778", lw=1.2,
                  zorder=3, solid_capstyle="round")
    if EXPAND_NEIGHBORS:
        for (x1, y1), (x2, y2) in neighbor_segs:
            axL.plot([x1, x2], [y1, y2], color="#b41010", lw=2.4,
                      zorder=4, solid_capstyle="round")
    else:
        for (x1, y1), (x2, y2) in neighbor_segs:
            axL.plot([x1, x2], [y1, y2], color="#e8a778", lw=1.2,
                      zorder=3, solid_capstyle="round")
    for (x1, y1), (x2, y2) in closed_segs:
        axL.plot([x1, x2], [y1, y2], color="#b41010", lw=2.4,
                  zorder=5, solid_capstyle="round")

    axL.plot(
        [BOX_LON_LO, BOX_LON_HI, BOX_LON_HI, BOX_LON_LO, BOX_LON_LO],
        [BOX_LAT_LO, BOX_LAT_LO, BOX_LAT_HI, BOX_LAT_HI, BOX_LAT_LO],
        color="black", lw=0.8, ls="--", alpha=0.55, zorder=4,
    )

    axL.plot(CRASH_LON, CRASH_LAT, marker="o", color="#1a7a3a",
              markeredgecolor="black", markeredgewidth=0.8,
              ms=8, zorder=6)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#b41010", lw=2.6,
                label="closed I-15 NB links"),
        Line2D([0], [0], color="black", lw=0.9, ls="--",
                label="eval box"),
        Line2D([0], [0], marker="o", color="#1a7a3a",
                markeredgecolor="black", markeredgewidth=0.6,
                lw=0, ms=7, label="crash point"),
    ]
    axL.legend(handles=handles, loc="lower left", frameon=False,
                fontsize=8, handlelength=2.2, borderaxespad=0.4)
    axL.set_xlim(lon_lo, lon_hi); axL.set_ylim(lat_lo, lat_hi)
    axL.set_aspect(1.0 / np.cos(np.deg2rad(CRASH_LAT)))
    axL.set_xticks([]); axL.set_yticks([])
    for s in axL.spines.values():
        s.set_color("#888"); s.set_linewidth(0.6)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.02)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  saved {OUT_PDF}")
    print(f"  saved {OUT_PNG}")


if __name__ == "__main__":
    main()
