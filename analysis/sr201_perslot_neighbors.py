"""Per-5-min-slot trace for WB SR-201 and its immediate neighbors,
Sep 21 vs Sep 28, 13:00-14:55.

Three nested groups:
  G1  WB SR-201 mainline + on/off ramps   (lat 40.700-40.715, fs>=20, x2<x1)
  G2  WB+EB SR-201 corridor                (lat 40.700-40.715, fs>=20)
  G3  Corridor + frontage / parallels      (lat 40.695-40.725, fs>=15)

For each slot, report the sum across each group on both days, the delta,
the relative error, and (for G1 only) the link-level breakdown of the
top three drops.
"""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.io import load_popularity_npz

RAW_NPZ = str(ROOT / "data/popularity_results_osm.npz")
NETWORK_XML = str(ROOT / "data/slc_network.xml")


def slot(day, h, m):  return f"2018-09-{day:02d} {h:02d}:{m:02d}:00"


def main():
    b = load_popularity_npz(RAW_NPZ)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    link_ids = [str(x) for x in b["link_ids"]]
    t2i = {t: i for i, t in enumerate(times)}

    HMS = [(h, m) for h in (13, 14) for m in range(0, 60, 5)]  # 24 slots
    idx21 = [t2i[slot(21, h, m)] for h, m in HMS]
    idx28 = [t2i[slot(28, h, m)] for h, m in HMS]
    M21 = M[idx21].toarray().astype(np.float64)
    M28 = M[idx28].toarray().astype(np.float64)

    # Network geometry.
    nodes = {}
    geom = {}
    for ev, el in ET.iterparse(NETWORK_XML, events=("start",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("x")), float(el.get("y")))
        elif el.tag == "link":
            f, t = el.get("from"), el.get("to")
            if f in nodes and t in nodes:
                x1, y1 = nodes[f]; x2, y2 = nodes[t]
                fs = float(el.get("freespeed", 0.0))
                geom[el.get("id")] = dict(
                    mlon=0.5*(x1+x2), mlat=0.5*(y1+y2), fs=fs,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                )
            el.clear()

    def filt(lat_lo, lat_hi, lon_lo, lon_hi, fs_min, dir_filter=None):
        out = []
        for i, lid in enumerate(link_ids):
            g = geom.get(lid)
            if g is None: continue
            if not (lat_lo <= g["mlat"] <= lat_hi): continue
            if not (lon_lo <= g["mlon"] <= lon_hi): continue
            if g["fs"] < fs_min: continue
            if dir_filter == "WB" and not (g["x2"] < g["x1"]): continue
            if dir_filter == "EB" and not (g["x2"] > g["x1"]): continue
            out.append(i)
        return np.array(out, dtype=np.int64)

    G1 = filt(40.700, 40.715, -112.15, -111.85, 20.0, "WB")
    G2 = filt(40.700, 40.715, -112.15, -111.85, 20.0, None)
    G3 = filt(40.695, 40.725, -112.20, -111.80, 15.0, None)

    print(f"G1 (WB SR-201, fs>=20)        : {len(G1):3d} links")
    print(f"G2 (WB+EB SR-201)              : {len(G2):3d} links")
    print(f"G3 (corridor + neighbors)      : {len(G3):3d} links")

    eps = 1e-6

    print(f"\n{'slot':>5s}  "
          f"{'G1.s21':>6s} {'G1.s28':>6s} {'G1.d':>5s} {'G1.r':>6s}  | "
          f"{'G2.s21':>6s} {'G2.s28':>6s} {'G2.d':>5s} {'G2.r':>6s}  | "
          f"{'G3.s21':>6s} {'G3.s28':>6s} {'G3.d':>5s} {'G3.r':>6s}")
    flags_for_slot = []
    for k, (h, m) in enumerate(HMS):
        row = []
        for G in (G1, G2, G3):
            s21 = M21[k, G].sum(); s28 = M28[k, G].sum()
            d   = s21 - s28
            r   = d / (s28 + eps)
            row.extend([s21, s28, d, r])
        # Flag: WB G1 |rel| >= 0.40 OR delta <= -10
        flag = (abs(row[3]) >= 0.40) or (row[2] <= -10)
        mark = " *" if flag else "  "
        print(f" {h:02d}:{m:02d}  "
              f"{row[0]:6.0f} {row[1]:6.0f} {row[2]:+5.0f} {row[3]:+6.2f}{mark}| "
              f"{row[4]:6.0f} {row[5]:6.0f} {row[6]:+5.0f} {row[7]:+6.2f}  | "
              f"{row[8]:6.0f} {row[9]:6.0f} {row[10]:+5.0f} {row[11]:+6.2f}")
        if flag:
            flags_for_slot.append((h, m, k))

    # For each flagged slot, list top 3 WB SR-201 links by |delta|.
    print(f"\nSlots with WB G1 |rel|>=0.40 or delta<=-10 trips: "
          f"{len(flags_for_slot)} of {len(HMS)}")
    for h, m, k in flags_for_slot:
        print(f"\n--- {h:02d}:{m:02d} ---  WB G1 delta / rel breakdown (top 5 links by |delta|):")
        deltas = M21[k, G1] - M28[k, G1]
        order = np.argsort(-np.abs(deltas))
        for o in order[:5]:
            j = G1[o]
            s21 = M21[k, j]; s28 = M28[k, j]
            r = (s21 - s28) / (s28 + eps)
            g = geom[link_ids[j]]
            print(f"   link {link_ids[j]:>6s}  ({g['mlat']:.4f}, {g['mlon']:.4f})  "
                  f"fs={g['fs']:4.1f}  s21={s21:3.0f}  s28={s28:3.0f}  "
                  f"d={s21-s28:+4.0f}  rel={r:+6.2f}")


if __name__ == "__main__":
    main()
