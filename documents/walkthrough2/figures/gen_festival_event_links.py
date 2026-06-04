"""9th and 9th Street Festival event-link map (two-panel).

LEFT  panel: the local SLC road graph with the closed-link set drawn
            in red and their graph neighbors drawn in a less-bold orange.
RIGHT panel: an OpenStreetMap-tile basemap of the same extent, fetched
            from tile.openstreetmap.org and stitched with PIL.

Closure: 900 South from roughly 700 East to 1100 East on 2018-09-15.
"""
import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NETWORK_XML = str(ROOT / "data/slc_network.xml")
GRAPH_JSON  = str(ROOT / "data/city_graph_full.json")
import os
EXPAND_NEIGHBORS = os.environ.get("TPPR_EXPAND", "0") == "1"
SUFFIX = "_expanded" if EXPAND_NEIGHBORS else ""
OUT_PDF = Path(__file__).parent / f"festival_event_links{SUFFIX}.pdf"
OUT_PNG = Path(__file__).parent / f"festival_event_links{SUFFIX}.png"

# Two-box closure: a short EW segment on 900 S and a short NS segment on
# 900 E. Boxes from user (corners): each closure box is the lat/lon
# rectangle spanning its 4 corner points.
# Box 1 (EW along 900 S): corners (40.749940, -111.863884),
#   (40.749590, -111.863906), (40.749883, -111.868326),
#   (40.749671, -111.868348).
CLOSURE_BOX1_LAT_LO = 40.749590
CLOSURE_BOX1_LAT_HI = 40.749940
CLOSURE_BOX1_LON_LO = -111.868348
CLOSURE_BOX1_LON_HI = -111.863884
# Box 2 (NS along 900 E): corners (40.747956, -111.865172),
#   (40.747908, -111.865494), (40.752069, -111.865140),
#   (40.752069, -111.865526).
CLOSURE_BOX2_LAT_LO = 40.747908
CLOSURE_BOX2_LAT_HI = 40.752069
CLOSURE_BOX2_LON_LO = -111.865526
CLOSURE_BOX2_LON_HI = -111.865140

# Evaluation box (drives the surgery MRE). Corners from user:
#   NW (40.756455, -111.876984)
#   NE (40.756409, -111.853615)
#   SE (40.741489, -111.853707)
#   SW (40.741419, -111.876953)
BOX_LAT_LO = 40.741419
BOX_LAT_HI = 40.756455
BOX_LON_LO = -111.876984
BOX_LON_HI = -111.853615

# Plot frame. Corners from user:
#   NW (40.758520, -111.879710)
#   NE (40.758566, -111.845131)
#   SE (40.739957, -111.844702)
#   SW (40.739841, -111.879771)
FRAME_LAT_LO = 40.739841
FRAME_LAT_HI = 40.758566
FRAME_LON_LO = -111.879771
FRAME_LON_HI = -111.844702

# 9th and 9th festival hub: coordinates given by the user.
CENTER_LAT, CENTER_LON = 40.749792, -111.865271

GMAP_PNG = Path(__file__).parent / "festival_googlemap.png"


def main():
    print(f"[start] festival event-link two-panel")

    print("[load] adjacency for graph neighbors ...")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    adj = graph.get("adjacency", {})
    print("[scan] network XML for plot segments + closed/box masks ...")
    nodes = {}
    # First pass: nodes only.
    for _, el in ET.iterparse(NETWORK_XML, events=("start",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("x")), float(el.get("y")))
        elif el.tag == "link":
            el.clear()
    # Second pass: identify the seed closure (lids only).
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
        lid  = el.get("id")
        in_box1 = (
            CLOSURE_BOX1_LAT_LO <= mlat <= CLOSURE_BOX1_LAT_HI
            and CLOSURE_BOX1_LON_LO <= mlon <= CLOSURE_BOX1_LON_HI
            and abs(dx) > 1.5 * abs(dy)
        )
        in_box2 = (
            CLOSURE_BOX2_LAT_LO <= mlat <= CLOSURE_BOX2_LAT_HI
            and CLOSURE_BOX2_LON_LO <= mlon <= CLOSURE_BOX2_LON_HI
            and abs(dy) > 1.5 * abs(dx)
        )
        if in_box1 or in_box2:
            seed_lids.add(lid)
        el.clear()

    # One-hop graph neighbors of the seed (always computed; used in legend
    # only when EXPAND is on, otherwise still drawn so the reader sees
    # the surrounding stubs that get closed in the expanded experiment).
    neighbor_lids = set()
    for lid in seed_lids:
        for ol in adj.get(lid, []):
            neighbor_lids.add(ol)
    for src, outs in adj.items():
        if any(o in seed_lids for o in outs):
            neighbor_lids.add(src)
    neighbor_lids -= seed_lids

    # Third pass: classify each link using the seed + neighbor sets.
    closed_lids   = set()
    neighbor_segs = []
    box_lids      = set()
    bg_segs, fw_segs, box_segs, closed_segs = [], [], [], []
    for _, el in ET.iterparse(NETWORK_XML, events=("start",)):
        if el.tag != "link":
            el.clear()
            continue
        f, t = el.get("from"), el.get("to")
        if f not in nodes or t not in nodes:
            el.clear(); continue
        x1, y1 = nodes[f]; x2, y2 = nodes[t]
        mlat = 0.5 * (y1 + y2); mlon = 0.5 * (x1 + x2)
        fs   = float(el.get("freespeed", 0.0))
        lid  = el.get("id")
        seg  = ((x1, y1), (x2, y2))
        in_box = (
            BOX_LAT_LO <= mlat <= BOX_LAT_HI
            and BOX_LON_LO <= mlon <= BOX_LON_HI
        )
        if lid in seed_lids:
            closed_lids.add(lid)
            closed_segs.append(seg)
        elif lid in neighbor_lids:
            neighbor_segs.append(seg)
        elif in_box:
            box_lids.add(lid)
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

    # Two-panel layout: road-graph view (left) + Google Maps screenshot
    # (right), same wspace=0 styling as the earlier figure.
    print(f"[load] google map screenshot at {GMAP_PNG} ...")
    from PIL import Image
    gmap = np.array(Image.open(GMAP_PNG).convert("RGB"))
    print(f"  screenshot shape: {gmap.shape}")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 3.8),
                                     gridspec_kw={"wspace": 0.0})

    # LEFT: graph view
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
        # In the expanded variant, the 1-hop neighbors are also closed,
        # drawn in the exact same red as the seed closure.
        for (x1, y1), (x2, y2) in neighbor_segs:
            axL.plot([x1, x2], [y1, y2], color="#b41010", lw=2.4,
                      zorder=4, solid_capstyle="round")
    else:
        # In the seed-only variant the neighbors are NOT closed; they
        # still belong to the eval-box surroundings, so draw them in
        # the same orange as the box stubs.
        for (x1, y1), (x2, y2) in neighbor_segs:
            axL.plot([x1, x2], [y1, y2], color="#e8a778", lw=1.2,
                      zorder=3, solid_capstyle="round")
    for (x1, y1), (x2, y2) in closed_segs:
        axL.plot([x1, x2], [y1, y2], color="#b41010", lw=2.4,
                  zorder=5, solid_capstyle="round")
    # Distinct colored dot at the user-supplied 9th-and-9th coordinates.
    axL.plot(CENTER_LON, CENTER_LAT, marker="o", color="#1a7a3a",
              markeredgecolor="black", markeredgewidth=0.8,
              ms=8, zorder=6)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#b41010", lw=2.6, label="closed links"),
        Line2D([0], [0], marker="o", color="#1a7a3a",
                markeredgecolor="black", markeredgewidth=0.6,
                lw=0, ms=7, label="9th and 9th"),
    ]
    axL.legend(handles=handles, loc="lower left", frameon=False,
                fontsize=8, handlelength=2.2, borderaxespad=0.4)
    axL.set_xlim(lon_lo, lon_hi); axL.set_ylim(lat_lo, lat_hi)
    axL.set_aspect(1.0 / np.cos(np.deg2rad(CENTER_LAT)))
    axL.set_xticks([]); axL.set_yticks([])
    for s in axL.spines.values(): s.set_color("#888"); s.set_linewidth(0.6)

    # RIGHT panel: Google Maps screenshot, full native resolution.
    axR.imshow(gmap, origin="upper", zorder=0,
                interpolation="none", resample=False)
    axR.set_xticks([]); axR.set_yticks([])
    for s in axR.spines.values(): s.set_color("#888"); s.set_linewidth(0.6)
    axR.set_aspect("equal")

    fig.subplots_adjust(left=0.03, right=0.97, top=0.99,
                         bottom=0.04, wspace=0.0)
    # bbox_inches='tight' trims the whitespace each aspect-locked axes
    # leaves around itself, so the two panels touch with no visible
    # interior gap in the saved file.
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  saved {OUT_PDF}")
    print(f"  saved {OUT_PNG}")


if __name__ == "__main__":
    main()
