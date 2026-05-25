"""Self-consistency check on the HMM emission scale.

For each chosen obs_noise level (declared sigma), match was already run
on CHPC. Here we *measure* the empirical residual standard deviation:

    sigma_measured = std{ d_perp(fix_i, matched_subedge_i) }

over a sample of routes. The matcher is self-consistent iff
sigma_measured ~ sigma_declared.

Setup:
  - Take routes that touched >= 3 distinct low-resolution links
    (so the trip is real, multi-link, not a 1-link blip).
  - Sample N_ROUTES of them.
  - For each fix in those routes, project (lat, lon) to the same
    equirectangular metres frame used by the matcher.
  - The matcher's true residual is distance(fix, matched sub-edge of
    parent low-res link).  We approximate this by the minimum
    perpendicular distance from fix to any sub-edge of the matched
    parent low-res link (the matcher would have chosen the closest
    sub-edge of that parent).
"""
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import pickle
import pyarrow.parquet as pq

ROOT = "/home/hamid/Downloads/new/TwoPhase_PageRank_Project"
PKL = os.path.join(ROOT, "data", "network_osm.pkl")
NOISE1_JSON = os.path.join(ROOT, "data", "map-match", "noise1.json")
NOISE3_JSON = os.path.join(ROOT, "data", "map-match", "noise3.json")
NOISE5_JSON = os.path.join(ROOT, "data", "map-match", "noise5.json")
NOISE4_JSON = os.path.join(ROOT, "data", "map-match", "noise4.json")
import glob as _glob
NOISE3p75_CHUNKS = sorted(
    p for p in _glob.glob(os.path.join(
        ROOT, "data", "map-match", "matched_routes_osm_noise3p75_chunk*.json"))
    if not p.endswith(".partial.json")
)
PARQUET = os.path.join(ROOT, "data", "compressed_routes.parquet")

CENTER_LAT = 40.75
CENTER_LON = -111.90
R_EARTH = 6371000.0

N_ROUTES = 40000  # 4x larger sample for stronger stability check
MIN_LINKS = 3
SEED = 7


def project(lat, lon):
    """Equirectangular projection -> metres, same as matcher pickle."""
    x = R_EARTH * math.radians(lon - CENTER_LON) * math.cos(math.radians(CENTER_LAT))
    y = R_EARTH * math.radians(lat - CENTER_LAT)
    return x, y


def project_vec(lat, lon):
    x = R_EARTH * np.deg2rad(lon - CENTER_LON) * np.cos(np.deg2rad(CENTER_LAT))
    y = R_EARTH * np.deg2rad(lat - CENTER_LAT)
    return x, y


def dist_point_to_segment(px, py, ax, ay, bx, by):
    """Perpendicular distance from point P to segment AB.  Vectorised
    over segments (so px, py are scalars and ax..by are arrays)."""
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    safe = np.maximum(seg_len_sq, 1e-9)
    t = ((px - ax) * dx + (py - ay) * dy) / safe
    t = np.clip(t, 0.0, 1.0)
    qx = ax + t * dx
    qy = ay + t * dy
    return np.hypot(px - qx, py - qy)


def build_matsim_to_subedges(pkl):
    """Invert edge_to_matsim: matsim_lid -> list of (xa, ya, xb, yb)."""
    proj_nodes = pkl["proj_nodes"]
    edge_to_matsim = pkl["edge_to_matsim"]
    out = defaultdict(list)
    for (a, b), matsim_lid in edge_to_matsim.items():
        if a in proj_nodes and b in proj_nodes:
            xa, ya = proj_nodes[a]
            xb, yb = proj_nodes[b]
            out[str(matsim_lid)].append((xa, ya, xb, yb))
    # convert to numpy arrays for fast distance computation
    matsim_seg = {}
    for lid, segs in out.items():
        arr = np.asarray(segs, dtype=np.float64)
        matsim_seg[lid] = arr  # shape (k, 4)
    return matsim_seg


def stream_routes(json_path):
    """Yield (route_id_int, fixes_list) from the matched JSON streamingly."""
    with open(json_path) as f:
        for line in f:
            line = line.strip().rstrip(",")
            if line in ("[", "]", ""):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            route = obj.get("route", {})
            rid = route.get("route_id")
            fixes = route.get("fixes", [])
            yield int(rid), fixes


def load_routes_with_min_links(json_path, min_links, keep_only=None):
    """Returns {route_id: fixes_list} for routes with >= min_links
    distinct lids. If keep_only is given (a set of route_ids), only
    those routes are kept in the dict (memory-light)."""
    print(f"[json] scanning {json_path}", flush=True)
    t0 = time.time()
    out = {}
    seen = 0
    for rid, fixes in stream_routes(json_path):
        seen += 1
        if len({fix[0] for fix in fixes}) >= min_links:
            if keep_only is None or rid in keep_only:
                out[rid] = fixes
        if seen % 200000 == 0:
            print(f"  ... {seen:,} routes scanned, "
                  f"{len(out):,} kept so far", flush=True)
    print(f"[json] {len(out):,} kept (of {seen:,} total) "
          f"in {time.time() - t0:.1f}s", flush=True)
    return out


def collect_qualifying_ids(json_path, min_links):
    """First-pass: only return the SET of route_ids with >= min_links."""
    print(f"[json] scanning ids only: {json_path}", flush=True)
    t0 = time.time()
    out = set()
    seen = 0
    for rid, fixes in stream_routes(json_path):
        seen += 1
        if len({fix[0] for fix in fixes}) >= min_links:
            out.add(rid)
    print(f"[json] {len(out):,} qualifying ids (of {seen:,}) "
          f"in {time.time() - t0:.1f}s", flush=True)
    return out


def measure_for(label, sigma_declared, routes, route_ids, parquet_rows, matsim_seg):
    print()
    print(f"== {label}  (sigma_declared = {sigma_declared} m) ==")
    residuals = []
    skipped_no_geom = 0
    skipped_no_fix = 0
    for rid in route_ids:
        fixes = routes.get(rid)
        if not fixes:
            continue
        fix_lookup = parquet_rows.get(rid)
        if not fix_lookup:
            continue
        for lid_str, t_gps in fixes:
            segs = matsim_seg.get(lid_str)
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
            # vectorised: compute distance from (px,py) to every segment of
            # the matched low-res parent link, take the minimum.
            d = dist_point_to_segment(
                px, py,
                segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
            )
            residuals.append(float(d.min()))
    r = np.asarray(residuals)
    med = float(np.median(r))
    q25, q75 = np.percentile(r, [25, 75])
    iqr = float(q75 - q25)
    sigma_iqr = iqr / 1.349
    mad = float(np.median(np.abs(r - med)))
    sigma_mad = mad / 0.6745
    sigma_std = float(r.std(ddof=1))
    print(f"  N fixes used   : {len(r):,d}")
    print(f"  skipped (no geom for matched lid): {skipped_no_geom:,d}")
    print(f"  skipped (no fix in parquet)      : {skipped_no_fix:,d}")
    print(f"  mean residual    : {r.mean():.3f} m")
    print(f"  median residual  : {med:.3f} m")
    print(f"  std  residual    : {sigma_std:.3f} m   (sensitive to tails)")
    print(f"  IQR              : {iqr:.3f} m  -> sigma_iqr = IQR/1.349 = "
          f"{sigma_iqr:.3f} m   <-- robust sigma_measured")
    print(f"  MAD              : {mad:.3f} m  -> sigma_mad = MAD/0.6745 = "
          f"{sigma_mad:.3f} m")
    print(f"  90th pct         : {np.percentile(r, 90):.3f} m")
    print(f"  95th pct         : {np.percentile(r, 95):.3f} m")
    print(f"  99th pct         : {np.percentile(r, 99):.3f} m")
    print(f"  gap (declared - sigma_iqr) = "
          f"{sigma_declared - sigma_iqr:+.3f} m   (robust gap)")
    return {"r": r, "sigma_std": sigma_std,
            "sigma_iqr": sigma_iqr, "sigma_mad": sigma_mad}


def load_parquet_for_route_ids(route_ids):
    """Stream parquet row group by row group; keep only rows for the
    requested route_ids."""
    print(f"[parquet] streaming rows for {len(route_ids):,} route_ids")
    t0 = time.time()
    rid_set = set(int(r) for r in route_ids)
    f = pq.ParquetFile(PARQUET)
    out = defaultdict(dict)
    n_rg = f.num_row_groups
    rows_total = 0
    for i in range(n_rg):
        df = f.read_row_group(i, columns=["route_id", "time_sec", "lat", "lon"]).to_pandas()
        mask = df["route_id"].isin(rid_set)
        if mask.any():
            sub = df.loc[mask]
            for rid, t, la, lo in zip(sub["route_id"].values,
                                      sub["time_sec"].values,
                                      sub["lat"].values,
                                      sub["lon"].values):
                out[int(rid)][int(t)] = (float(la), float(lo))
            rows_total += int(mask.sum())
        if (i + 1) % 10 == 0 or i == n_rg - 1:
            print(f"  row group {i+1}/{n_rg}, kept {rows_total:,} rows")
    print(f"  done in {time.time() - t0:.1f}s, "
          f"covered {len(out):,} of {len(rid_set):,} route_ids")
    return out


def main():
    print("[pickle] loading network_osm.pkl ...")
    t0 = time.time()
    with open(PKL, "rb") as f:
        pkl = pickle.load(f)
    print(f"  loaded in {time.time() - t0:.1f}s")
    matsim_seg = build_matsim_to_subedges(pkl)
    print(f"  matsim_lids with high-res geometry: {len(matsim_seg):,}")
    del pkl

    # Pass 1: collect qualifying ids from each dataset (no fixes -> tiny).
    ids1 = collect_qualifying_ids(NOISE1_JSON, MIN_LINKS)
    ids3 = collect_qualifying_ids(NOISE3_JSON, MIN_LINKS)
    ids5 = collect_qualifying_ids(NOISE5_JSON, MIN_LINKS)
    ids4 = collect_qualifying_ids(NOISE4_JSON, MIN_LINKS)
    ids3p75 = set()
    for path in NOISE3p75_CHUNKS:
        ids3p75 |= collect_qualifying_ids(path, MIN_LINKS)
    print(f"[ids] n1={len(ids1):,}  n3={len(ids3):,}  "
          f"n3p75={len(ids3p75):,}  n4={len(ids4):,}  n5={len(ids5):,}",
          flush=True)

    common = ids1 & ids3 & ids3p75 & ids4 & ids5
    print(f"\n[sample] {len(common):,} route_ids qualify in ALL FOUR datasets",
          flush=True)
    rng = random.Random(SEED)
    chosen = rng.sample(sorted(common), min(N_ROUTES, len(common)))
    print(f"[sample] picked {len(chosen):,} routes (seed={SEED})", flush=True)
    chosen_set = set(chosen)

    # Pass 2: now load fixes ONLY for the chosen 1500 routes, per dataset.
    parquet_rows = load_parquet_for_route_ids(chosen)
    n1 = load_routes_with_min_links(NOISE1_JSON, MIN_LINKS, keep_only=chosen_set)
    n3 = load_routes_with_min_links(NOISE3_JSON, MIN_LINKS, keep_only=chosen_set)
    n3p75 = {}
    for path in NOISE3p75_CHUNKS:
        n3p75.update(load_routes_with_min_links(path, MIN_LINKS, keep_only=chosen_set))
    n4 = load_routes_with_min_links(NOISE4_JSON, MIN_LINKS, keep_only=chosen_set)
    n5 = load_routes_with_min_links(NOISE5_JSON, MIN_LINKS, keep_only=chosen_set)

    r1 = measure_for("noise=1", 1.0, n1, chosen, parquet_rows, matsim_seg)
    r3 = measure_for("noise=3", 3.0, n3, chosen, parquet_rows, matsim_seg)
    r3p75 = measure_for("noise=3.75", 3.75, n3p75, chosen, parquet_rows, matsim_seg)
    r4 = measure_for("noise=4", 4.0, n4, chosen, parquet_rows, matsim_seg)
    r5 = measure_for("noise=5", 5.0, n5, chosen, parquet_rows, matsim_seg)

    print()
    print("=" * 90)
    print("  SELF-CONSISTENCY SUMMARY (robust sigma_iqr = IQR / 1.349)")
    print("=" * 90)
    print(f"  {'noise':<12s} {'declared':>10s} {'sig_iqr':>10s} {'gap_iqr':>10s}  "
          f"{'sig_mad':>10s} {'gap_mad':>10s}  {'sig_std':>10s}")
    for name, sigma, d in [("noise=1",    1.0,  r1),
                           ("noise=3",    3.0,  r3),
                           ("noise=3.75", 3.75, r3p75),
                           ("noise=4",    4.0,  r4),
                           ("noise=5",    5.0,  r5)]:
        gap_iqr = d["sigma_iqr"] - sigma
        gap_mad = d["sigma_mad"] - sigma
        tag = "too tight" if gap_iqr > 0 else "too loose"
        print(f"  {name:<12s} {sigma:>10.2f} {d['sigma_iqr']:>10.3f} "
              f"{gap_iqr:>+10.3f}  {d['sigma_mad']:>10.3f} {gap_mad:>+10.3f}  "
              f"{d['sigma_std']:>10.3f}   {tag}")


if __name__ == "__main__":
    main()
