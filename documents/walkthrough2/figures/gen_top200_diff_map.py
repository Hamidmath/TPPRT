"""Whole-network map highlighting the 200 links with the largest
per-slot mean difference between Sep 11 and Sep 18, 16:00-16:55.

Per-link statistic:
  for each 5-min slot t in 16:00, 16:05, ..., 16:55:
    if c_Sep11(t, i) >= 5:
      diff_t(i) = c_Sep11(t, i) - c_Sep18(t, i)
  mean_diff(i) = mean of diff_t(i) over eligible slots
  rank links by mean_diff(i) descending, take top 200.

Output: documents/walkthrough2/figures/top200_diff_map.pdf
"""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from core.io import load_popularity_npz

NETWORK_XML = str(ROOT / "data/slc_network.xml")
RAW_NPZ     = str(ROOT / "data/popularity_results_osm.npz")

import os
TOP_N = int(os.environ.get("TPPR_TOP_N", "200"))
OUT = Path(__file__).parent / f"top{TOP_N}_diff_map.pdf"

FLOOR = 5


def main():
    print(f"[start] top-{TOP_N} per-slot mean RELATIVE error, "
          "Sep 11 vs Sep 18 16:00-17:00")
    b = load_popularity_npz(RAW_NPZ)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    t2idx = {t: i for i, t in enumerate(times)}
    link_ids = [str(x) for x in b["link_ids"]]

    slots11 = [f"2018-09-11 16:{m:02d}:00" for m in range(0, 60, 5)]
    slots18 = [f"2018-09-18 16:{m:02d}:00" for m in range(0, 60, 5)]
    idx11 = [t2idx[t] for t in slots11]
    idx18 = [t2idx[t] for t in slots18]
    print(f"  slots: {len(idx11)} Sep 11, {len(idx18)} Sep 18")

    M11 = M[idx11].toarray().astype(np.float64)
    M18 = M[idx18].toarray().astype(np.float64)
    print(f"  block shapes: {M11.shape}, {M18.shape}")

    # Per slot t with c_Sep11(t, i) >= FLOOR:
    #   rel_t(i) = |c_Sep11(t,i) - c_Sep18(t,i)| / c_Sep11(t,i)
    # Per link: mean rel_t over eligible slots. Rank descending.
    elig = M11 >= FLOOR                                    # (12, N) boolean
    rel_per_slot = np.where(elig,
                             np.abs(M11 - M18) / np.maximum(M11, 1.0),
                             0.0)
    n_elig = elig.sum(axis=0)                              # per-link eligible count
    sum_rel = rel_per_slot.sum(axis=0)                     # per-link sum of rel errs
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_rel = np.where(n_elig > 0, sum_rel / n_elig, np.nan)

    valid = ~np.isnan(mean_rel)
    n_valid = int(valid.sum())
    print(f"  links with >=1 eligible slot: {n_valid:,}")

    # Never include ineligible links: cap selection at n_valid.
    take = min(TOP_N, n_valid)
    if take < TOP_N:
        print(f"  requested top-{TOP_N} but only {n_valid} valid; "
              f"using all {n_valid}")
    order = np.argsort(np.where(valid, mean_rel, -np.inf))[::-1]
    top_idx = order[:take]
    top_link_ids = set(link_ids[i] for i in top_idx)
    print(f"  top-{take} mean |rel err| range: "
          f"[{mean_rel[top_idx].min():.3f}, {mean_rel[top_idx].max():.3f}]")
    print(f"  10 largest:")
    for k in range(10):
        i = top_idx[k]
        print(f"    link {link_ids[i]:>6s}  mean_rel={mean_rel[i]:.3f}  "
              f"eligible_bins={int(n_elig[i])}/12")

    # ---- Map ----
    print("\n  parsing road network for plotting ...")
    nodes = {}
    region_segs = []   # top-200
    fw_segs = []       # motorway / freeway (light highlight, includes I-15)
    bg_segs = []       # everything else
    for ev, el in ET.iterparse(NETWORK_XML, events=("start",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("x")), float(el.get("y")))
        elif el.tag == "link":
            f, t = el.get("from"), el.get("to")
            if f in nodes and t in nodes:
                seg = (nodes[f], nodes[t])
                fs = float(el.get("freespeed", 0.0))
                if el.get("id") in top_link_ids:
                    region_segs.append(seg)
                elif fs >= 31.0:
                    fw_segs.append(seg)
                else:
                    bg_segs.append(seg)
            el.clear()
    print(f"  road segments: top-{TOP_N} = {len(region_segs)},  "
          f"freeway = {len(fw_segs):,},  bg = {len(bg_segs):,}")

    # Bounding box from all segments.
    pts = [p for s in bg_segs + region_segs for p in s]
    lons = [p[0] for p in pts]; lats = [p[1] for p in pts]
    lon_lo, lon_hi = min(lons), max(lons)
    lat_lo, lat_hi = min(lats), max(lats)

    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.set_facecolor("white")
    for (x1, y1), (x2, y2) in bg_segs:
        ax.plot([x1, x2], [y1, y2], color="#d3d3d3", lw=0.25,
                 zorder=1, solid_capstyle="round")
    for (x1, y1), (x2, y2) in fw_segs:
        ax.plot([x1, x2], [y1, y2], color="#9ec5e8", lw=0.7,
                 zorder=2, solid_capstyle="round")
    for (x1, y1), (x2, y2) in region_segs:
        ax.plot([x1, x2], [y1, y2], color="#c92420", lw=0.4,
                 zorder=3, solid_capstyle="round", alpha=0.95)

    # 2 km radius circle at (40.722348, -111.904691).
    from matplotlib.patches import Ellipse
    cx, cy = -111.904691, 40.722348
    r_km = 2.0
    deg_per_km_lat = 1.0 / 111.0
    deg_per_km_lon = 1.0 / (111.0 * np.cos(np.deg2rad(cy)))
    ax.add_patch(Ellipse((cx, cy),
                          width=2 * r_km * deg_per_km_lon,
                          height=2 * r_km * deg_per_km_lat,
                          fill=False, edgecolor="#1f77b4",
                          linestyle=(0, (4, 2)), lw=1.1, zorder=4))

    # Legend describing the freeway tint and the circle.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#9ec5e8", lw=2.0,
                label="freeway (70 mph)"),
        Line2D([0], [0], color="#1f77b4", lw=1.1,
                linestyle=(0, (4, 2)), label=f"{r_km:.0f} km radius"),
        Line2D([0], [0], color="#c92420", lw=0.9,
                label=f"top-{TOP_N} links"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False,
               fontsize=8, handlelength=2.0, borderaxespad=0.4)
    ax.set_xlim(lon_lo, lon_hi); ax.set_ylim(lat_lo, lat_hi)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(lats))))
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#888888"); spine.set_linewidth(0.6)
    ax.set_title(f"Top-{TOP_N} links by per-slot mean $|$rel err$|$ "
                  "(Sep 11 vs Sep 18), 16:00-17:00 (Sep 11 >= 5 filter)",
                  fontsize=9, pad=4)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  saved {OUT}")


if __name__ == "__main__":
    main()
