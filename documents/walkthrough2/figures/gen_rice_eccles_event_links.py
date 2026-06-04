"""Rice-Eccles Stadium event-link map (single-panel, no screenshot).

Event:    Washington at Utah football, Rice-Eccles Stadium, Salt Lake
          City, Sat 2018-09-15, kickoff 19:00 PDT (20:00 local MDT),
          attendance 47,445.
Stadium:  (40.76075, -111.84855).

Same colour table / line widths / aspect handling as the festival and
I-15 crash figures: background road graph in grey, freeways in light
blue, the eval-box surroundings in orange, the one-hop graph
neighbors of the closed set in orange (non-expanded) or red
(expanded), the stadium-adjacent closed streets in red, and a green
marker at the stadium. Eval box drawn as a dashed black rectangle.
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
OUT_PDF = Path(__file__).parent / f"rice_eccles_event_links{SUFFIX}.pdf"
OUT_PNG = Path(__file__).parent / f"rice_eccles_event_links{SUFFIX}.png"

# Stadium center (exact, user-supplied).
STADIUM_LAT, STADIUM_LON = 40.759692, -111.848837

# Closure box: the streets immediately around Rice-Eccles that close to
# through traffic on game day (Guardsman Way, South Campus Dr, Wasatch
# Dr, Mario Capecchi Dr, the 500 S frontage). ~+/-380 m N-S, +/-340 m
# E-W around the stadium. All links here are surface streets.
CLOSURE_LAT_LO = 40.7573
CLOSURE_LAT_HI = 40.7642
CLOSURE_LON_LO = -111.8525
CLOSURE_LON_HI = -111.8446

# Evaluation box, centred on the stadium: ~3.1 km N-S x ~3.0 km E-W,
# so the game-day approach arterials (Foothill Dr, 1300 E, 500/400 S,
# Sunnyside Ave, Guardsman Way, Wasatch Dr) are inside the box.
BOX_LAT_LO = 40.7468
BOX_LAT_HI = 40.7748
BOX_LON_LO = -111.8666
BOX_LON_HI = -111.8306

# Plot frame, a little larger than the eval box on every side.
FRAME_LAT_LO = 40.740
FRAME_LAT_HI = 40.781
FRAME_LON_LO = -111.874
FRAME_LON_HI = -111.823


def main():
    print("[start] rice-eccles event-link single-panel")

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

    # Seed closure: every link whose midpoint is in the closure box.
    seed_lids = set()
    for _, el in ET.iterparse(NETWORK_XML, events=("start",)):
        if el.tag != "link":
            el.clear()
            continue
        f, t = el.get("from"), el.get("to")
        if f not in nodes or t not in nodes:
            el.clear(); continue
        x1, y1 = nodes[f]; x2, y2 = nodes[t]
        mlat = 0.5 * (y1 + y2); mlon = 0.5 * (x1 + x2)
        lid = el.get("id")
        if (CLOSURE_LAT_LO <= mlat <= CLOSURE_LAT_HI
                and CLOSURE_LON_LO <= mlon <= CLOSURE_LON_HI):
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

    seed_lids = set(); neighbor_lids = set()   # no closure: show only box + stadium
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

    axL.plot(STADIUM_LON, STADIUM_LAT, marker="o", color="#1a7a3a",
              markeredgecolor="black", markeredgewidth=0.8,
              ms=8, zorder=6)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="black", lw=0.9, ls="--",
                label="eval box"),
        Line2D([0], [0], marker="o", color="#1a7a3a",
                markeredgecolor="black", markeredgewidth=0.6,
                lw=0, ms=7, label="Rice-Eccles Stadium"),
    ]
    axL.legend(handles=handles, loc="lower left", frameon=False,
                fontsize=8, handlelength=2.2, borderaxespad=0.4)
    axL.set_xlim(lon_lo, lon_hi); axL.set_ylim(lat_lo, lat_hi)
    axL.set_aspect(1.0 / np.cos(np.deg2rad(STADIUM_LAT)))
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
