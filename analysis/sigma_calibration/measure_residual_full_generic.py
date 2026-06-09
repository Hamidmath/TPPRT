"""Full-corpus residual measurement for any obs_noise level on CHPC.

Reads env vars:
  NOISE_TAG        e.g. "1", "1p75", "3", "3p75", "4", "5"
  SIGMA            declared sigma in metres (e.g. "4.0")
  JSON_PATH        absolute path to the matched-routes JSON
  OUTLIER_MAX      residuals > this many metres are filtered (e.g. "30")

Outputs JSON to ~/tppr/results/residual_full_noise{NOISE_TAG}.json with:
  - BEFORE/AFTER outlier filter stats
  - sigma_std / sigma_iqr / sigma_mad
  - histogram bins for further analysis
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

TPPR = os.environ.get("TPPR", os.path.expanduser("~/tppr"))
NOISE_TAG = os.environ["NOISE_TAG"]
SIGMA = float(os.environ["SIGMA"])
JSON_PATH = os.environ["JSON_PATH"]
OUTLIER_MAX = float(os.environ.get("OUTLIER_MAX", "30.0"))

PKL = os.path.join(TPPR, "inputs", "network_osm.pkl")
PARQUET = os.path.join(TPPR, "inputs", "compressed_routes.parquet")
OUT_JSON = os.path.join(TPPR, "results", f"residual_full_noise{NOISE_TAG}.json")

MIN_LINKS = 3
CENTER_LAT = 40.75
CENTER_LON = -111.90
R_EARTH = 6371000.0
BATCH_SIZE = 100000
N_WORKERS = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())) - 2)


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


def stream_routes(json_path):
    with open(json_path) as f:
        for line in f:
            s = line.strip().rstrip(",")
            if s in ("[", "]", ""):
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            r = obj.get("route", {})
            yield int(r.get("route_id")), r.get("fixes", [])


def load_routes(json_path, min_links):
    print(f"[json] loading {json_path}", flush=True)
    t0 = time.time()
    out = {}
    seen = 0
    for rid, fixes in stream_routes(json_path):
        seen += 1
        if len({fix[0] for fix in fixes}) >= min_links:
            out[rid] = fixes
        if seen % 100000 == 0:
            print(f"  ... {seen:,} scanned, {len(out):,} kept", flush=True)
    print(f"[json] kept {len(out):,} of {seen:,} routes "
          f"in {time.time() - t0:.1f}s", flush=True)
    return out


def load_parquet_for_batch(rid_set):
    f = pq.ParquetFile(PARQUET)
    out = defaultdict(dict)
    for i in range(f.num_row_groups):
        df = f.read_row_group(
            i, columns=["route_id", "time_sec", "lat", "lon"]).to_pandas()
        mask = df["route_id"].isin(rid_set)
        if mask.any():
            sub = df.loc[mask]
            for rid, ts, la, lo in zip(sub["route_id"].values,
                                       sub["time_sec"].values,
                                       sub["lat"].values,
                                       sub["lon"].values):
                out[int(rid)][int(ts)] = (float(la), float(lo))
    return out


_ROUTES = None
_PARQUET = None
_MATSIM_SEG = None


def _init_worker(routes, parquet, matsim_seg):
    global _ROUTES, _PARQUET, _MATSIM_SEG
    _ROUTES = routes
    _PARQUET = parquet
    _MATSIM_SEG = matsim_seg


def _process_chunk(rids):
    res = []
    sk_geom = sk_fix = 0
    for rid in rids:
        fixes = _ROUTES.get(rid)
        coords = _PARQUET.get(rid)
        if not fixes or not coords:
            continue
        for lid_str, t_gps in fixes:
            segs = _MATSIM_SEG.get(lid_str)
            if segs is None:
                sk_geom += 1
                continue
            t_int = int(round(t_gps))
            ll = coords.get(t_int)
            if ll is None:
                sk_fix += 1
                continue
            lat, lon = ll
            px, py = project(lat, lon)
            d = dist_point_to_segment(
                px, py,
                segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
            )
            res.append(float(d.min()))
    return np.asarray(res), sk_geom, sk_fix


def chunkify(rids, n):
    rids = list(rids)
    base, rem = divmod(len(rids), n)
    out = []
    i = 0
    for k in range(n):
        sz = base + (1 if k < rem else 0)
        out.append(rids[i:i+sz])
        i += sz
    return out


def stats_for(arr, declared_sigma):
    """Compute the suite of stats for a residual array."""
    if arr.size == 0:
        return {}
    med = float(np.median(arr))
    sigma_std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = float(q75 - q25)
    sigma_iqr = iqr / 1.349
    mad = float(np.median(np.abs(arr - med)))
    sigma_mad = mad / 0.6745
    pct = {p: float(np.percentile(arr, p)) for p in [10, 25, 50, 75, 90, 95, 99, 99.9]}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": med,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "sigma_std": sigma_std,
        "sigma_iqr": sigma_iqr,
        "sigma_mad": sigma_mad,
        "iqr": iqr,
        "mad": mad,
        "percentiles": pct,
        "gap_std": sigma_std - declared_sigma,
        "gap_iqr": sigma_iqr - declared_sigma,
        "gap_mad": sigma_mad - declared_sigma,
    }


def histogram(arr, max_edge=50.0, n_bins=200):
    """Return [(left_edge, right_edge, count)] for residuals."""
    edges = np.linspace(0.0, max_edge, n_bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    # also one bucket for residuals above max_edge
    n_above = int((arr > max_edge).sum())
    out = []
    for i in range(n_bins):
        out.append([float(edges[i]), float(edges[i+1]), int(counts[i])])
    out.append([float(max_edge), float("inf"), n_above])
    return out


def main():
    t_start = time.time()
    print(f"[start] NOISE_TAG={NOISE_TAG}, SIGMA={SIGMA}, "
          f"OUTLIER_MAX={OUTLIER_MAX}, N_WORKERS={N_WORKERS}", flush=True)

    print(f"[pickle] loading {PKL}", flush=True)
    with open(PKL, "rb") as f:
        pkl = pickle.load(f)
    matsim_seg = build_matsim_to_subedges(pkl)
    print(f"  matsim_lids w/ geom: {len(matsim_seg):,}", flush=True)
    del pkl

    routes = load_routes(JSON_PATH, MIN_LINKS)
    rids = list(routes.keys())
    print(f"[batch] total {len(rids):,} routes, "
          f"BATCH_SIZE={BATCH_SIZE:,}", flush=True)

    all_res = []
    total_no_geom = 0
    total_no_fix = 0
    n_batches = (len(rids) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi in range(n_batches):
        b_rids = rids[bi*BATCH_SIZE:(bi+1)*BATCH_SIZE]
        b_set = set(b_rids)
        print(f"\n=== batch {bi+1}/{n_batches}  ({len(b_rids):,} routes) ===",
              flush=True)
        t_pq = time.time()
        coords = load_parquet_for_batch(b_set)
        print(f"  parquet: {len(coords):,} routes with coords "
              f"in {time.time()-t_pq:.1f}s", flush=True)
        chunks = chunkify(b_rids, N_WORKERS * 4)
        t_p = time.time()
        with Pool(processes=N_WORKERS,
                  initializer=_init_worker,
                  initargs=(routes, coords, matsim_seg)) as pool:
            results = pool.map(_process_chunk, chunks)
        for r_arr, sg, sf in results:
            all_res.append(r_arr)
            total_no_geom += sg
            total_no_fix += sf
        print(f"  compute: {time.time()-t_p:.1f}s "
              f"(running total residuals: "
              f"{sum(r.size for r in all_res):,})", flush=True)
        del coords

    all_arr = np.concatenate(all_res)
    print(f"\n[total] {len(all_arr):,} residuals collected in "
          f"{time.time()-t_start:.1f}s", flush=True)

    # BEFORE filter
    stats_before = stats_for(all_arr, SIGMA)
    # AFTER filter
    mask = all_arr <= OUTLIER_MAX
    n_filtered = int((~mask).sum())
    all_filt = all_arr[mask]
    stats_after = stats_for(all_filt, SIGMA)

    print(f"\n[filter] dropped {n_filtered:,} residuals > {OUTLIER_MAX}m "
          f"({100.0 * n_filtered / all_arr.size:.3f}% of total)", flush=True)
    print(f"  before: n={stats_before['n']:,}  sigma_std={stats_before['sigma_std']:.3f}  "
          f"sigma_iqr={stats_before['sigma_iqr']:.3f}  sigma_mad={stats_before['sigma_mad']:.3f}",
          flush=True)
    print(f"  after : n={stats_after['n']:,}  sigma_std={stats_after['sigma_std']:.3f}  "
          f"sigma_iqr={stats_after['sigma_iqr']:.3f}  sigma_mad={stats_after['sigma_mad']:.3f}",
          flush=True)

    # Histogram of UNFILTERED residuals up to 50 m (most of the mass)
    hist = histogram(all_arr, max_edge=50.0, n_bins=200)

    out = {
        "noise_tag": NOISE_TAG,
        "declared_sigma": SIGMA,
        "outlier_max": OUTLIER_MAX,
        "n_routes": len(routes),
        "n_residuals_total": int(all_arr.size),
        "n_residuals_filtered": int(all_filt.size),
        "n_outliers_dropped": n_filtered,
        "outlier_pct": float(100.0 * n_filtered / all_arr.size),
        "skipped_no_geom": total_no_geom,
        "skipped_no_fix": total_no_fix,
        "stats_before_filter": stats_before,
        "stats_after_filter": stats_after,
        "histogram_residual_meters_0_to_50": hist,
        "wall_time_s": time.time() - t_start,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] saved {OUT_JSON} (wall {time.time()-t_start:.1f}s)",
          flush=True)


if __name__ == "__main__":
    main()
