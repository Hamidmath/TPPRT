"""WB SR-201 Sep 21 13:00 crash: graph-surgery intervention.

Per professor: a crash is a GRAPH change, not a DEMAND change. We use last
week's diffused popularity (Sep 14 same slot) as input E_b, the chain's
already-calibrated dynamics (no retraining), and compare three predictions
against the actual Sep 21 diffused popularity:

  L1 = last-week-as-is        target ~ Sep 14 (no chain at all)
  L2 = chain on ORIGINAL graph (Sep 14 -> Sep 21, P unchanged)
  L3 = chain on SURGERY graph  (Sep 14 -> Sep 21, closed links removed)

Two chains:
  SP    : alpha = 0.0508, alpha_s = 3.0504 (from sp_prof_ladder.json, speed_only)
  TP-new: beta = 0.102, rho = 0.101, alpha_s = 3.0523 (from tp_new_prof_ladder.json)

Two closure sets:
  A focused : 58979 (WB SR-201 mainline approach to I-15)
  B broad   : A + Redwood Rd cluster (74247, 74249, 31585, 83108)

Three windows:
  W1 : 13:00-13:55  (12 slots, immediate aftermath)
  W2 : 14:00-14:55  (12 slots, recovery hour)
  WA : 13:00-14:55  (24 slots, full event)

Three evaluation regions:
  R1 : whole network (links with target>=eps)
  R2 : WB SR-201 corridor (23 links, lat 40.700-40.715, fs>=20, x2<x1)
  R3 : corridor + neighbors (733 links, lat 40.695-40.725, lon -112.20 to -111.80, fs>=15)
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.io import load_popularity_npz

SMOOTH_NPZ = str(ROOT / "data/popularity_results_smoothed_osm_gamma020.npz")
NETWORK_XML = str(ROOT / "data/slc_network.xml")
GRAPH_JSON = str(ROOT / "data/city_graph_full.json")

EPS = 1e-6
TOL, MAX_ITER = 1e-7, 300

# Calibrated dynamics (no retraining).
SP_ALPHA   = 0.0508
SP_AS      = 3.0504
TP_BETA    = 0.102
TP_RHO     = 0.101
TP_AS      = 3.0523

CLOSURE_A = ["58979"]
CLOSURE_B = ["58979", "74247", "74249", "31585", "83108"]


def slot(day, h, m): return f"2018-09-{day:02d} {h:02d}:{m:02d}:00"


def build_P_weighted(graph, links, lid_to_idx, weights, closed=None):
    closed = set(closed or [])
    adj = graph.get("adjacency", {})
    rows, cols, data = [], [], []
    for i, lid in enumerate(links):
        if lid in closed:
            continue
        succ_lids = [ol for ol in adj.get(lid, []) if ol in lid_to_idx and ol not in closed]
        if not succ_lids:
            continue
        succ_idx = [lid_to_idx[ol] for ol in succ_lids]
        out_w = np.array([weights[j] for j in succ_idx], dtype=np.float64)
        total = out_w.sum()
        if total <= 0:
            continue
        norm = out_w / total
        for k, j in enumerate(succ_idx):
            rows.append(j); cols.append(i); data.append(float(norm[k]))
    return csr_matrix((data, (rows, cols)),
                       shape=(len(links), len(links)), dtype=np.float64)


def get_speed_lane(graph, links):
    s, l = [], []
    for lid in links:
        ed = graph["links"].get(lid, {})
        s.append(ed.get("speed", 11.17))
        l.append(ed.get("lanes", 1.0))
    s = np.array(s, dtype=np.float64); l = np.array(l, dtype=np.float64)
    if s.mean() > 0: s /= s.mean()
    if l.mean() > 0: l /= l.mean()
    return s, l


def sp_iter(P_cs, u, alpha, tol=TOL, max_iter=MAX_ITER):
    v = u.copy()
    for _ in range(max_iter):
        v_new = (1.0 - alpha) * (P_cs @ v) + alpha * u
        s = float(v_new.sum())
        if s > 0: v_new /= s
        if float(np.abs(v_new - v).sum()) < tol:
            v = v_new; break
        v = v_new
    return v


def tp_new_iter(P_cs, u, beta, rho, tol=TOL, max_iter=MAX_ITER):
    N = u.shape[0]
    v_up = u.copy(); v_down = np.zeros(N)
    for _ in range(max_iter):
        s_down = float(v_down.sum())
        v_up_new   = (1.0 - beta) * (P_cs @ v_up) + rho * u * s_down
        v_down_new = beta * v_up + (1.0 - rho) * (P_cs @ v_down)
        s = float(v_up_new.sum() + v_down_new.sum())
        if s > 0: v_up_new /= s; v_down_new /= s
        diff = float(np.abs(v_up_new - v_up).sum() + np.abs(v_down_new - v_down).sum())
        v_up, v_down = v_up_new, v_down_new
        if diff < tol: break
    return v_up + v_down


def load_diffused(npz_path, N, lid_to_idx, needed_times):
    b = load_popularity_npz(npz_path)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_ids = [str(lid) for lid in b["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_ids], dtype=np.int64)
    valid = proj >= 0
    needed = set(needed_times)
    rows = {}
    for ti, t in enumerate(times):
        if t not in needed: continue
        row = M.getrow(ti).toarray().ravel().astype(np.float64)
        E = np.zeros(N, dtype=np.float64)
        np.add.at(E, proj[valid], row[valid])
        s = E.sum()
        if s > 0: E /= s
        rows[t] = E
    return rows


def mre(target, pred, mask=None, target_floor=1e-9):
    if mask is None:
        mask = target > target_floor
    if mask.sum() == 0:
        return float("nan")
    rel = np.abs(target[mask] - pred[mask]) / (target[mask] + EPS)
    return float(rel.mean())


def main():
    print("[load] graph ...")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes = get_speed_lane(graph, links)
    print(f"  N = {N:,}")
    print(f"  closure A: {CLOSURE_A}")
    print(f"  closure B: {CLOSURE_B}")
    for c in CLOSURE_B:
        if c not in lid_to_idx:
            print(f"  WARNING: closure link {c} not in graph")

    # Region masks (computed from network XML midpoints).
    print("[parse] network for region masks ...")
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
                geom[el.get("id")] = dict(mlon=0.5*(x1+x2), mlat=0.5*(y1+y2),
                                            fs=fs, x1=x1, y1=y1, x2=x2, y2=y2)
            el.clear()

    def make_mask(lat_lo, lat_hi, lon_lo, lon_hi, fs_min, dir_filter=None):
        m = np.zeros(N, dtype=bool)
        for i, lid in enumerate(links):
            g = geom.get(lid)
            if g is None: continue
            if not (lat_lo <= g["mlat"] <= lat_hi): continue
            if not (lon_lo <= g["mlon"] <= lon_hi): continue
            if g["fs"] < fs_min: continue
            if dir_filter == "WB" and not (g["x2"] < g["x1"]): continue
            m[i] = True
        return m

    R2_mask = make_mask(40.700, 40.715, -112.15, -111.85, 20.0, "WB")  # WB SR-201
    R3_mask = make_mask(40.695, 40.725, -112.20, -111.80, 15.0, None)  # neighbors
    print(f"  R2 (WB SR-201)  : {int(R2_mask.sum())} links")
    print(f"  R3 (neighbors)  : {int(R3_mask.sum())} links")

    # Slot windows.
    HMS_W1 = [(13, m) for m in range(0, 60, 5)]
    HMS_W2 = [(14, m) for m in range(0, 60, 5)]
    HMS_WA = HMS_W1 + HMS_W2
    WINDOWS = [("W1 13:00-13:55", HMS_W1),
                ("W2 14:00-14:55", HMS_W2),
                ("WA 13:00-14:55", HMS_WA)]

    needed = set()
    for _, hms in WINDOWS:
        for h, m in hms:
            needed.add(slot(14, h, m)); needed.add(slot(21, h, m))
    print(f"[load] diffused popularity for {len(needed)} time stamps ...")
    rows = load_diffused(SMOOTH_NPZ, N, lid_to_idx, needed)

    # Pre-build speed weights (alpha_s applied to normalized speeds).
    w_sp = np.power(speeds, SP_AS)
    w_tp = np.power(speeds, TP_AS)

    # Build P matrices: original and surgery A/B for each weight set.
    print("[build] P matrices ...")
    P = {}
    for tag_w, w in [("SP", w_sp), ("TP", w_tp)]:
        P[(tag_w, "orig")] = build_P_weighted(graph, links, lid_to_idx, w)
        P[(tag_w, "A")]    = build_P_weighted(graph, links, lid_to_idx, w, closed=CLOSURE_A)
        P[(tag_w, "B")]    = build_P_weighted(graph, links, lid_to_idx, w, closed=CLOSURE_B)
        print(f"  P[{tag_w},orig] nnz={P[(tag_w,'orig')].nnz:,}")
        print(f"  P[{tag_w},A]    nnz={P[(tag_w,'A')].nnz:,}")
        print(f"  P[{tag_w},B]    nnz={P[(tag_w,'B')].nnz:,}")

    # For each window x chain x graph_variant, compute predictions and MRE.
    print(f"\n{'='*100}")
    print(f"COMPARISON  (MRE on each region; lower is better)")
    print(f"{'='*100}\n")

    results = []
    for wlabel, hms in WINDOWS:
        inputs = [rows[slot(14, h, m)] for h, m in hms]
        targets = [rows[slot(21, h, m)] for h, m in hms]

        # L1: just predict last-week-as-is (input == prediction).
        mre_L1 = dict(R1=[], R2=[], R3=[])
        for u, tgt in zip(inputs, targets):
            mre_L1["R1"].append(mre(tgt, u))
            mre_L1["R2"].append(mre(tgt, u, mask=R2_mask))
            mre_L1["R3"].append(mre(tgt, u, mask=R3_mask))

        for chain in ("SP", "TP"):
            for variant in ("orig", "A", "B"):
                mres = dict(R1=[], R2=[], R3=[])
                Pmat = P[(chain, variant)]
                for u, tgt in zip(inputs, targets):
                    if chain == "SP":
                        v = sp_iter(Pmat, u, SP_ALPHA)
                    else:
                        v = tp_new_iter(Pmat, u, TP_BETA, TP_RHO)
                    mres["R1"].append(mre(tgt, v))
                    mres["R2"].append(mre(tgt, v, mask=R2_mask))
                    mres["R3"].append(mre(tgt, v, mask=R3_mask))
                results.append((wlabel, chain, variant,
                                 float(np.mean(mres["R1"])),
                                 float(np.mean(mres["R2"])),
                                 float(np.mean(mres["R3"]))))

        # Print this window block.
        print(f"--- {wlabel} ---")
        print(f"  {'prediction':>22s}  {'R1 net':>8s}  {'R2 WB':>8s}  {'R3 nbr':>8s}")
        print(f"  {'L1 last-week-as-is':>22s}  "
              f"{np.mean(mre_L1['R1']):8.4f}  "
              f"{np.mean(mre_L1['R2']):8.4f}  "
              f"{np.mean(mre_L1['R3']):8.4f}")
        for chain in ("SP", "TP"):
            for variant in ("orig", "A", "B"):
                row = [r for r in results if r[0] == wlabel and r[1] == chain and r[2] == variant][0]
                label = f"L2 {chain} orig" if variant == "orig" else f"L3 {chain} surgery-{variant}"
                print(f"  {label:>22s}  {row[3]:8.4f}  {row[4]:8.4f}  {row[5]:8.4f}")
        print()


if __name__ == "__main__":
    main()
