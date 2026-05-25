"""Estimate per-trip L_up / L_down at OSM sub-edge granularity from
matched_routes_osm.json + network_osm.pkl.

The matched JSON only stores MATSim link IDs (sub-edges were aggregated up).
For each MATSim link the chain-walk mapping records how many OSM sub-edges
make up that link; we use that count as the "OSM-level link count" the
trip touched on that MATSim link. This is a tight upper bound: the matcher
with non_emitting_states=True traverses every sub-edge of any MATSim chain
the trip actually crosses.

Output: same schema as empirical_phase_split.npz but in OSM-sub-edge units.
"""
import argparse
import json
import os
import pickle
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict, Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-all-k", default=None)
    args = ap.parse_args()

    print(f"Loading mapping: {args.pkl}")
    with open(args.pkl, "rb") as f:
        net = pickle.load(f)
    e2m = net["edge_to_matsim"]
    chain_len = Counter(e2m.values())
    print(f"  unique matsim links covered: {len(chain_len):,}")
    print(f"  mean sub-edges per matsim link (directed): "
          f"{np.mean(list(chain_len.values())):.2f}")

    print(f"Loading network XML for speed/lanes: {args.xml}")
    root = ET.parse(args.xml).getroot()
    speed, lanes = {}, {}
    for lk in root.find("links"):
        if "car" not in lk.get("modes", ""):
            continue
        speed[lk.get("id")] = float(lk.get("freespeed"))
        lanes[lk.get("id")] = float(lk.get("permlanes"))

    print(f"Loading matched: {args.matched}")
    t0 = time.time()
    with open(args.matched) as f:
        data = json.load(f)
    print(f"  {len(data):,} routes loaded ({time.time()-t0:.0f}s)")

    L_up, L_down, peak_pos, K_arr = [], [], [], []
    L_up_all, L_down_all, K_all = [], [], []
    n_short = n_zero = 0
    for item in data:
        if "route" not in item:
            continue
        seen = OrderedDict()
        for lid, _t in item["route"]["fixes"]:
            seen.setdefault(str(lid), True)
        matsim_links = list(seen.keys())
        if len(matsim_links) < 1:
            continue
        # OSM sub-edge length contribution per matsim link in this trip
        osm_lens = [int(chain_len.get(lid, 1)) for lid in matsim_links]
        K_osm = int(sum(osm_lens))

        if len(matsim_links) >= 3:
            sp = np.array([speed.get(l, 0.0) for l in matsim_links])
            ln = np.array([lanes.get(l, 0.0) for l in matsim_links])
            score = sp * ln
            if score.max() <= 0:
                n_zero += 1
                forced_peak_osm = 0
                L_up_all.append(forced_peak_osm)
                L_down_all.append(K_osm - 1 - forced_peak_osm)
                K_all.append(K_osm)
                continue
            max_score = score.max()
            cands = np.where(score == max_score)[0]
            middle = (len(matsim_links) - 1) / 2.0
            peak_idx_matsim = int(cands[np.argmin(np.abs(cands - middle))])
            # Convert peak index from MATSim-link space to OSM-sub-edge space
            peak_idx_osm = int(sum(osm_lens[:peak_idx_matsim]))
            Lu = peak_idx_osm
            Ld = K_osm - 1 - peak_idx_osm
            L_up.append(Lu)
            L_down.append(Ld)
            peak_pos.append(peak_idx_osm / max(K_osm - 1, 1))
            K_arr.append(K_osm)
            L_up_all.append(Lu)
            L_down_all.append(Ld)
            K_all.append(K_osm)
        else:
            n_short += 1
            forced_peak_osm = 0
            L_up_all.append(forced_peak_osm)
            L_down_all.append(K_osm - 1 - forced_peak_osm if K_osm >= 1 else 0)
            K_all.append(K_osm)

    L_up = np.array(L_up, dtype=np.int32)
    L_down = np.array(L_down, dtype=np.int32)
    peak_pos = np.array(peak_pos, dtype=np.float64)
    K_arr = np.array(K_arr, dtype=np.int32)

    print(f"\n=== OSM sub-edge granularity ===")
    print(f"Trips with >=3 matsim links (kept):   {len(L_up):,}")
    print(f"Trips skipped (<3 matsim links):      {n_short:,}")
    print(f"Trips skipped (zero speed/lanes):     {n_zero:,}")
    print(f"E[K_osm]   = {K_arr.mean():.2f}")
    print(f"E[L_up]    = {L_up.mean():.2f}  -> beta = "
          f"{1/(1+L_up.mean()):.4f}")
    print(f"E[L_down]  = {L_down.mean():.2f}  -> rho  = "
          f"{1/(1+L_down.mean()):.4f}")
    print(f"E[peak_pos] = {peak_pos.mean():.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        L_up=L_up,
        L_down=L_down,
        peak_pos=peak_pos,
        K=K_arr,
    )
    print(f"Wrote {args.out}")

    if args.out_all_k is not None:
        L_up_all = np.array(L_up_all, dtype=np.int32)
        L_down_all = np.array(L_down_all, dtype=np.int32)
        K_all = np.array(K_all, dtype=np.int32)
        os.makedirs(os.path.dirname(args.out_all_k), exist_ok=True)
        np.savez(args.out_all_k, L_up=L_up_all, L_down=L_down_all, K=K_all)
        print(f"Wrote {args.out_all_k} ({len(L_up_all):,} trips, "
              f"E[L_up]={L_up_all.mean():.2f}, E[L_down]={L_down_all.mean():.2f})")


if __name__ == "__main__":
    main()
