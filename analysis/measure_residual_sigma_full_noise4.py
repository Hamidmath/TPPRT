"""Full-corpus residual measurement for noise=4 only.

Goal: compute sigma_std / sigma_iqr / sigma_mad for ALL qualifying noise=4
routes (K >= 3), not a sample. Uses multiprocessing to scale across cores.

Recipe:
  1. Load network_osm.pkl -> matsim_lid -> sub-edge array.
  2. Stream the noise=4 combined JSON -> route_id -> list of (lid, t_gps).
  3. Stream the parquet by row group, filter to all those route_ids,
     project (lat, lon) to local metres.
  4. Split routes into chunks; multiprocessing.Pool computes per-fix
     residuals per chunk in parallel.
  5. Aggregate all residuals; compute std / IQR / MAD.
"""
import json
import math
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pickle
import pyarrow.parquet as pq

ROOT = "/home/hamid/Downloads/new/TwoPhase_PageRank_Project"
PKL = os.path.join(ROOT, "data", "network_osm.pkl")
NOISE4_JSON = os.path.join(ROOT, "data", "map-match", "noise4.json")
PARQUET = os.path.join(ROOT, "data", "compressed_routes.parquet")

SIGMA_DECLARED = 4.0
MIN_LINKS = 3
CENTER_LAT = 40.75
CENTER_LON = -111.90
R_EARTH = 6371000.0

N_WORKERS = max(1, os.cpu_count() - 2)  # leave 2 cores free


def project(lat, lon):
    x = R_EARTH * math.radians(lon - CENTER_LON) * math.cos(math.radians(CENTER_LAT))
    y = R_EARTH * math.radians(lat - CENTER_LAT)
    return x, y


def dist_point_to_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    safe = np.maximum(seg_len_sq, 1e-9)
    t = ((px - ax) * dx + (py - ay) * dy) / safe
    t = np.clip(t, 0.0, 1.0)
    qx = ax + t * dx
    qy = ay + t * dy
    return np.hypot(px - qx, py - qy)


def build_matsim_to_subedges(pkl):
    proj_nodes = pkl["proj_nodes"]
    edge_to_matsim = pkl["edge_to_matsim"]
    out = defaultdict(list)
    for (a, b), matsim_lid in edge_to_matsim.items():
        if a in proj_nodes and b in proj_nodes:
            xa, ya = proj_nodes[a]
            xb, yb = proj_nodes[b]
            out[str(matsim_lid)].append((xa, ya, xb, yb))
    matsim_seg = {}
    for lid, segs in out.items():
        matsim_seg[lid] = np.asarray(segs, dtype=np.float64)
    return matsim_seg


def load_routes(json_path, min_links):
    """Stream JSON; keep route_id -> fixes for routes with >= min_links."""
    print(f"[json] loading {json_path}", flush=True)
    t0 = time.time()
    out = {}
    seen = 0
    with open(json_path) as f:
        for line in f:
            s = line.strip().rstrip(",")
            if s in ("[", "]", ""):
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            seen += 1
            r = obj.get("route", {})
            rid = int(r.get("route_id"))
            fixes = r.get("fixes", [])
            if len({fix[0] for fix in fixes}) >= min_links:
                out[rid] = fixes
            if seen % 100000 == 0:
                print(f"  ... {seen:,} scanned, {len(out):,} kept",
                      flush=True)
    print(f"[json] kept {len(out):,} of {seen:,} routes "
          f"in {time.time() - t0:.1f}s", flush=True)
    return out


def load_parquet(route_ids):
    """Stream parquet row group by row group; return {rid: {t_int: (lat, lon)}}."""
    print(f"[parquet] loading rows for {len(route_ids):,} route_ids",
          flush=True)
    t0 = time.time()
    rid_set = set(int(r) for r in route_ids)
    f = pq.ParquetFile(PARQUET)
    out = defaultdict(dict)
    n_rg = f.num_row_groups
    kept = 0
    for i in range(n_rg):
        df = f.read_row_group(
            i, columns=["route_id", "time_sec", "lat", "lon"]
        ).to_pandas()
        mask = df["route_id"].isin(rid_set)
        if mask.any():
            sub = df.loc[mask]
            for rid, ts, la, lo in zip(sub["route_id"].values,
                                       sub["time_sec"].values,
                                       sub["lat"].values,
                                       sub["lon"].values):
                out[int(rid)][int(ts)] = (float(la), float(lo))
            kept += int(mask.sum())
        if (i + 1) % 50 == 0 or i == n_rg - 1:
            print(f"  row group {i+1}/{n_rg}, kept {kept:,} rows",
                  flush=True)
    print(f"[parquet] done in {time.time() - t0:.1f}s "
          f"({len(out):,} routes have GPS rows)", flush=True)
    return out


# Worker globals (populated by initializer)
_ROUTES = None
_PARQUET = None
_MATSIM_SEG = None


def _init_worker(routes, parquet_dict, matsim_seg):
    global _ROUTES, _PARQUET, _MATSIM_SEG
    _ROUTES = routes
    _PARQUET = parquet_dict
    _MATSIM_SEG = matsim_seg


def _process_chunk(rids):
    """Return numpy array of residuals (one per fix) for the given route_ids."""
    res = []
    skipped_no_geom = 0
    skipped_no_fix = 0
    for rid in rids:
        fixes = _ROUTES.get(rid)
        fix_lookup = _PARQUET.get(rid)
        if not fixes or not fix_lookup:
            continue
        for lid_str, t_gps in fixes:
            segs = _MATSIM_SEG.get(lid_str)
            if segs is None:
                skipped_no_geom += 1
                continue
            t_int = int(round(t_gps))
            ll = fix_lookup.get(t_int)
            if ll is None:
                skipped_no_fix += 1
                continue
            lat, lon = ll
            px, py = project(lat, lon)
            d = dist_point_to_segment(
                px, py,
                segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
            )
            res.append(float(d.min()))
    return (np.asarray(res), skipped_no_geom, skipped_no_fix)


def chunkify(rids, n_chunks):
    rids = list(rids)
    base = len(rids) // n_chunks
    rem = len(rids) % n_chunks
    chunks = []
    i = 0
    for k in range(n_chunks):
        sz = base + (1 if k < rem else 0)
        chunks.append(rids[i:i + sz])
        i += sz
    return chunks


def main():
    t_start = time.time()
    print(f"[pickle] loading {PKL} ...", flush=True)
    with open(PKL, "rb") as f:
        pkl = pickle.load(f)
    matsim_seg = build_matsim_to_subedges(pkl)
    print(f"  matsim_lids with high-res geometry: {len(matsim_seg):,}",
          flush=True)
    del pkl

    routes = load_routes(NOISE4_JSON, MIN_LINKS)
    parquet_dict = load_parquet(routes.keys())

    rids = list(routes.keys())
    chunks = chunkify(rids, N_WORKERS * 4)  # more chunks than workers for load balance
    print(f"[parallel] {N_WORKERS} workers, {len(chunks)} chunks "
          f"(avg {len(rids)//len(chunks):,} routes/chunk)",
          flush=True)

    t0 = time.time()
    with Pool(processes=N_WORKERS,
              initializer=_init_worker,
              initargs=(routes, parquet_dict, matsim_seg)) as pool:
        results = pool.map(_process_chunk, chunks)

    print(f"[parallel] computed in {time.time() - t0:.1f}s", flush=True)

    all_res = np.concatenate([r[0] for r in results])
    no_geom = sum(r[1] for r in results)
    no_fix = sum(r[2] for r in results)

    print()
    print(f"=========================================================")
    print(f"  noise = {SIGMA_DECLARED}  (full noise=4 corpus, K>=3)")
    print(f"=========================================================")
    print(f"  routes used         : {len(rids):,}")
    print(f"  fixes used          : {len(all_res):,}")
    print(f"  skipped (no geom)   : {no_geom:,}")
    print(f"  skipped (no fix)    : {no_fix:,}")
    print(f"  mean residual       : {all_res.mean():.4f} m")
    med = float(np.median(all_res))
    print(f"  median residual     : {med:.4f} m")
    sigma_std = float(all_res.std(ddof=1))
    print(f"  std                 : {sigma_std:.4f} m   (sensitive)")
    q25, q75 = np.percentile(all_res, [25, 75])
    iqr = float(q75 - q25)
    sigma_iqr = iqr / 1.349
    print(f"  IQR                 : {iqr:.4f} m  ->  sigma_iqr = {sigma_iqr:.4f} m")
    mad = float(np.median(np.abs(all_res - med)))
    sigma_mad = mad / 0.6745
    print(f"  MAD                 : {mad:.4f} m  ->  sigma_mad = {sigma_mad:.4f} m")
    print(f"  90th pct            : {np.percentile(all_res, 90):.4f} m")
    print(f"  95th pct            : {np.percentile(all_res, 95):.4f} m")
    print(f"  99th pct            : {np.percentile(all_res, 99):.4f} m")
    print(f"  max                 : {all_res.max():.4f} m")

    print()
    print(f"  declared sigma         = {SIGMA_DECLARED:.2f} m")
    print(f"  gap (std  - declared)  = {sigma_std - SIGMA_DECLARED:+.4f} m")
    print(f"  gap (iqr  - declared)  = {sigma_iqr - SIGMA_DECLARED:+.4f} m")
    print(f"  gap (mad  - declared)  = {sigma_mad - SIGMA_DECLARED:+.4f} m")

    print()
    print(f"  total wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
