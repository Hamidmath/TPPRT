"""Compare raw per-link counts on Sept 21 vs Sept 28 around the
westbound SR-201 crash that occurred just before 13:00.

Three windows are reported:
  W1 = 13:00 to 13:55   (12 slots, immediate aftermath)
  W2 = 14:00 to 14:55   (12 slots, recovery)
  WA = 13:00 to 14:55   (24 slots, full post-crash hour-pair)

SR-201 (the 2100 South Freeway) runs east-west at lat ~40.706 - 40.712,
from I-15 (lon ~ -111.91) west to about Magna (lon ~ -112.13). We use
that lat band and split each link's direction by edge orientation
(x2 < x1 -> westbound).
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
    print("[load] raw popularity ...")
    b = load_popularity_npz(RAW_NPZ)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    link_ids = [str(x) for x in b["link_ids"]]
    t2i = {t: i for i, t in enumerate(times)}

    # Three slot lists per day.
    W1 = [(13, m) for m in range(0, 60, 5)]              # 13:00 .. 13:55
    W2 = [(14, m) for m in range(0, 60, 5)]              # 14:00 .. 14:55
    WA = W1 + W2                                          # 13:00 .. 14:55

    def block(day, hms):
        idx = [t2i[slot(day, h, m)] for h, m in hms]
        return M[idx].toarray().astype(np.float64)

    print("\n[parse] network ...")
    nodes = {}
    link_geom = {}
    for ev, el in ET.iterparse(NETWORK_XML, events=("start",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("x")), float(el.get("y")))
        elif el.tag == "link":
            f, t = el.get("from"), el.get("to")
            if f in nodes and t in nodes:
                x1, y1 = nodes[f]; x2, y2 = nodes[t]
                fs = float(el.get("freespeed", 0.0))
                link_geom[el.get("id")] = dict(
                    mlon=0.5*(x1+x2), mlat=0.5*(y1+y2), fs=fs,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                )
            el.clear()

    # SR-201 corridor.
    LAT_LO, LAT_HI = 40.700, 40.715
    LON_LO, LON_HI = -112.15, -111.85
    FS_MIN = 20.0

    wb, eb = [], []
    for i, lid in enumerate(link_ids):
        g = link_geom.get(lid)
        if g is None: continue
        if not (LAT_LO <= g["mlat"] <= LAT_HI): continue
        if not (LON_LO <= g["mlon"] <= LON_HI): continue
        if g["fs"] < FS_MIN: continue
        (wb if g["x2"] < g["x1"] else eb).append(i)
    wb, eb = np.array(wb), np.array(eb)
    print(f"\n[SR-201] WB={len(wb)} links   EB={len(eb)} links")

    eps = 1e-6
    for tag, hms in [("W1 (13:00-13:55)", W1),
                     ("W2 (14:00-14:55)", W2),
                     ("WA (13:00-14:55)", WA)]:
        B21 = block(21, hms); B28 = block(28, hms)
        s21 = B21.sum(axis=0); s28 = B28.sum(axis=0)
        diff = s21 - s28
        valid = s28 >= 5
        rel = np.where(valid, (s21 - s28) / (s28 + eps), 0.0)

        print(f"\n========== {tag} ==========")
        print(f"  whole network: Sep21={s21.sum():,.0f}  Sep28={s28.sum():,.0f}  "
              f"delta={diff.sum():+,.0f} ({100*diff.sum()/(s28.sum()+eps):+.2f}%)")
        print(f"  valid links: {int(valid.sum()):,}   "
              f"Overall MRE = {np.abs(rel[valid]).mean():.4f}   "
              f"signed mean = {rel[valid].mean():+.4f}")

        for label, sub in [("WB SR-201", wb), ("EB SR-201", eb)]:
            ss21 = s21[sub]; ss28 = s28[sub]
            v = ss28 >= 5
            print(f"  {label}: Sep21={ss21.sum():,.0f}  Sep28={ss28.sum():,.0f}  "
                  f"delta={(ss21-ss28).sum():+.0f} "
                  f"({100*(ss21.sum()-ss28.sum())/(ss28.sum()+eps):+.1f}%)  "
                  f"MRE(v={int(v.sum())})="
                  f"{(np.abs((ss21[v]-ss28[v])/(ss28[v]+eps))).mean():.4f}  "
                  f"signed={((ss21[v]-ss28[v])/(ss28[v]+eps)).mean():+.4f}")

        # Top 10 WB SR-201 links by |rel err| in this window.
        ss21 = s21[wb]; ss28 = s28[wb]
        v = ss28 >= 5
        if v.any():
            r = (ss21 - ss28) / (ss28 + eps)
            order = np.argsort(-np.abs(r) * v)
            print(f"  -- top 10 WB SR-201 links by |rel err| --")
            print(f"     {'link':>6s}  {'lat':>7s}  {'lon':>9s}  {'fs':>5s}  "
                  f"{'sum21':>5s}  {'sum28':>5s}  {'delta':>6s}  {'rel':>7s}")
            shown = 0
            for k in range(len(order)):
                j = wb[order[k]]
                if s28[j] < 5: continue
                g = link_geom[link_ids[j]]
                rr = (s21[j]-s28[j]) / (s28[j]+eps)
                print(f"     {link_ids[j]:>6s}  {g['mlat']:7.4f}  "
                      f"{g['mlon']:9.4f}  {g['fs']:5.1f}  "
                      f"{s21[j]:5.0f}  {s28[j]:5.0f}  "
                      f"{s21[j]-s28[j]:+6.0f}  {rr:+7.3f}")
                shown += 1
                if shown >= 10: break

    # Per-slot WB SR-201 trace for the 24 post-crash slots.
    print("\n[per-slot WB SR-201, 13:00-14:55]")
    B21A = block(21, WA); B28A = block(28, WA)
    print(f"  {'slot':>5s}  {'Sep21':>5s}  {'Sep28':>5s}  {'delta':>6s}  {'rel':>7s}")
    for k, (h, m) in enumerate(WA):
        s21w = B21A[k, wb].sum(); s28w = B28A[k, wb].sum()
        r = (s21w - s28w) / (s28w + eps)
        print(f"  {h:02d}:{m:02d}  {s21w:5.0f}  {s28w:5.0f}  "
              f"{s21w-s28w:+6.0f}  {r:+7.3f}")

    # ---- Network-wide top drops in W1 (immediate aftermath only) ----
    print("\n[network] top 20 absolute-drop links in W1 (13:00-13:55), Sep28>=5:")
    B21W1 = block(21, W1); B28W1 = block(28, W1)
    s21w1 = B21W1.sum(axis=0); s28w1 = B28W1.sum(axis=0)
    validw1 = s28w1 >= 5
    drop = np.where(validw1, s21w1 - s28w1, 0.0)
    order = np.argsort(drop)   # most-negative first
    print(f"  {'link':>6s}  {'lat':>7s}  {'lon':>9s}  {'fs':>5s}  "
          f"{'sum21':>5s}  {'sum28':>5s}  {'delta':>6s}  {'rel':>7s}  on_SR201")
    sr201_set = set(int(x) for x in wb.tolist() + eb.tolist())
    for k in range(20):
        j = order[k]
        if s28w1[j] < 5: break
        g = link_geom.get(link_ids[j])
        if g is None: continue
        r = (s21w1[j]-s28w1[j]) / (s28w1[j]+eps)
        tag = ("WB" if j in set(wb.tolist())
                else "EB" if j in set(eb.tolist()) else "-")
        print(f"  {link_ids[j]:>6s}  {g['mlat']:7.4f}  {g['mlon']:9.4f}  "
              f"{g['fs']:5.1f}  {s21w1[j]:5.0f}  {s28w1[j]:5.0f}  "
              f"{s21w1[j]-s28w1[j]:+6.0f}  {r:+7.3f}  {tag}")


if __name__ == "__main__":
    main()
