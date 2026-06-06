"""Compute per-trip empirical L_up / L_down from a matched-routes JSON.

Methodology (results/route_distribution_study/REPORT.tex sec:partB):
    For each trip with K >= 3 unique links:
        peak_idx = argmax_i (freespeed[link_i] * permlanes[link_i])
        ties broken by minimum |i - (K-1)/2| (closeness to middle)
        L_up   = peak_idx
        L_down = K - 1 - peak_idx

Output npz mirrors the existing empirical_phase_split.npz schema:
    L_up, L_down, peak_pos, avg_speed (20 K-bins), avg_lanes (20 K-bins)

Usage:
    python analysis/compute_phase_split.py \\
        --matched data/matched_routes_osm.json \\
        --xml data/slc_network.xml \\
        --out results/route_distribution_study/empirical_phase_split_osm.npz
"""
import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict

import numpy as np


def load_network(xml_path):
    print(f"Loading network XML: {xml_path}", flush=True)
    root = ET.parse(xml_path).getroot()
    speed = {}
    lanes = {}
    for lk in root.find("links"):
        if "car" not in lk.get("modes", ""):
            continue
        lid = lk.get("id")
        speed[lid] = float(lk.get("freespeed"))
        lanes[lid] = float(lk.get("permlanes"))
    print(f"  car links: {len(speed):,}", flush=True)
    return speed, lanes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-all-k", default=None,
                    help="If given, also write a phase_split_all_K.npz that "
                    "includes K<3 trips with forced peak_idx=0.")
    args = ap.parse_args()

    speed, lanes = load_network(args.xml)

    print(f"Loading matched routes: {args.matched} "
          f"({os.path.getsize(args.matched)/1e6:.0f} MB)", flush=True)
    t0 = time.time()
    with open(args.matched) as f:
        data = json.load(f)
    print(f"  {len(data):,} routes loaded ({time.time()-t0:.0f}s)", flush=True)

    L_up, L_down, peak_pos = [], [], []
    K_arr = []
    avg_speed_per_trip = []
    avg_lanes_per_trip = []
    L_up_all, L_down_all, K_all = [], [], []
    n_skipped_short = n_skipped_zero = 0
    for item in data:
        if "route" not in item:
            continue
        seen = OrderedDict()
        for lid, _ in item["route"]["fixes"]:
            seen.setdefault(str(lid), True)
        links = list(seen.keys())
        K = len(links)
        if K < 1:
            continue
        sp = np.array([speed.get(l, 0.0) for l in links])
        ln = np.array([lanes.get(l, 0.0) for l in links])
        score = sp * ln
        if score.max() <= 0 or K < 3:
            forced_peak = 0
            L_up_all.append(forced_peak)
            L_down_all.append(K - 1 - forced_peak)
            K_all.append(K)
            if K < 3:
                n_skipped_short += 1
            else:
                n_skipped_zero += 1
            continue
        max_score = score.max()
        cands = np.where(score == max_score)[0]
        middle = (K - 1) / 2.0
        peak_idx = int(cands[np.argmin(np.abs(cands - middle))])
        L_up.append(peak_idx)
        L_down.append(K - 1 - peak_idx)
        peak_pos.append(peak_idx / (K - 1) if K > 1 else 0.5)
        K_arr.append(K)
        avg_speed_per_trip.append(sp.mean())
        avg_lanes_per_trip.append(ln.mean())
        L_up_all.append(peak_idx)
        L_down_all.append(K - 1 - peak_idx)
        K_all.append(K)

    L_up = np.array(L_up, dtype=np.int32)
    L_down = np.array(L_down, dtype=np.int32)
    peak_pos = np.array(peak_pos, dtype=np.float64)
    K_arr = np.array(K_arr, dtype=np.int32)
    avg_speed_per_trip = np.array(avg_speed_per_trip, dtype=np.float64)
    avg_lanes_per_trip = np.array(avg_lanes_per_trip, dtype=np.float64)

    print(f"Trips with K>=3 (kept):      {len(L_up):,}", flush=True)
    print(f"Skipped (K<3):               {n_skipped_short:,}", flush=True)
    print(f"Skipped (zero speed/lanes):  {n_skipped_zero:,}", flush=True)
    print(f"E[L_up]   = {L_up.mean():.4f}  -> beta = "
          f"{1/(1+L_up.mean()):.4f}", flush=True)
    print(f"E[L_down] = {L_down.mean():.4f}  -> rho  = "
          f"{1/(1+L_down.mean()):.4f}", flush=True)
    print(f"E[peak_pos] = {peak_pos.mean():.4f}", flush=True)

    K_BINS = 20
    K_min = 3
    K_max = K_arr.max() if len(K_arr) else 50
    edges = np.linspace(K_min, K_max, K_BINS + 1)
    avg_speed_bins = np.zeros(K_BINS)
    avg_lanes_bins = np.zeros(K_BINS)
    for i in range(K_BINS):
        mask = (K_arr >= edges[i]) & (K_arr < edges[i + 1] if i < K_BINS - 1
                                       else K_arr <= edges[i + 1])
        if mask.any():
            avg_speed_bins[i] = avg_speed_per_trip[mask].mean()
            avg_lanes_bins[i] = avg_lanes_per_trip[mask].mean()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        L_up=L_up,
        L_down=L_down,
        peak_pos=peak_pos,
        avg_speed=avg_speed_bins,
        avg_lanes=avg_lanes_bins,
    )
    print(f"Wrote {args.out}", flush=True)

    if args.out_all_k is not None:
        L_up_all = np.array(L_up_all, dtype=np.int32)
        L_down_all = np.array(L_down_all, dtype=np.int32)
        K_all = np.array(K_all, dtype=np.int32)
        os.makedirs(os.path.dirname(args.out_all_k), exist_ok=True)
        np.savez(args.out_all_k, L_up=L_up_all, L_down=L_down_all, K=K_all)
        print(
            f"Wrote {args.out_all_k} ({len(L_up_all):,} trips, "
            f"E[L_up]={L_up_all.mean():.4f}, "
            f"E[L_down]={L_down_all.mean():.4f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
