"""CHPC parallel map-matcher built EXACTLY on Leuven library's OSM ingestion
pattern (shape-node-preserving graph; one InMemMap edge per consecutive pair
on each drivable highway way).

Network is loaded from the pre-built pickle written by build_network.py.

Matcher params (walkthrough2 §5):
    DistanceMatcher(map_con,
                    max_dist=400, max_dist_init=400,
                    min_prob_norm=1e-4, non_emitting_states=True)

Per-fix output (matsim_link_id, t_gps) using (osm_a, osm_b) -> matsim_link_id
mapping built into the pickle. Sub-edges with no matsim mapping are skipped
and counted in the log.

Parallelism: spawn multiprocessing.Pool. Each worker loads its own InMemMap
from the pickle (avoids fork+rtree COW issues).

Inputs (under $TPPR/inputs/):
    network_osm.pkl
    compressed_routes.parquet
Outputs (under $TPPR/results/):
    matched_routes_osm.json
    progress_osm.log
"""
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CENTER_LAT = 40.75
CENTER_LON = -111.90
R_EARTH = 6371000.0

MIN_GPS_FIXES = 5
CHUNK_INDEX = int(os.environ.get("CHUNK_INDEX", "0"))
N_CHUNKS = int(os.environ.get("N_CHUNKS", "1"))
MAX_DIST = int(os.environ.get("MAX_DIST", "400"))
MIN_PROB_NORM = 1e-4
OBS_NOISE = float(os.environ.get("OBS_NOISE", "1"))
RUN_TAG = os.environ.get("RUN_TAG", str(MAX_DIST))

TPPR = os.environ.get("TPPR", os.path.expanduser("~/tppr"))
INPUTS = os.path.join(TPPR, "inputs")
RESULTS = os.path.join(TPPR, "results")
NETWORK_PKL = os.path.join(INPUTS, "network_osm.pkl")
PARQUET = os.path.join(INPUTS, "compressed_routes.parquet")
OUTPUT_JSON = os.path.join(RESULTS, f"matched_routes_osm_{RUN_TAG}.json")
LOG_FILE = os.path.join(RESULTS, f"progress_osm_{RUN_TAG}.log")
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))

os.makedirs(RESULTS, exist_ok=True)


def project(lat, lon):
    cosphi = np.cos(np.radians(CENTER_LAT))
    x = (lon - CENTER_LON) * cosphi * R_EARTH * (np.pi / 180.0)
    y = (lat - CENTER_LAT) * R_EARTH * (np.pi / 180.0)
    return x, y


def log_line(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


_W = {}


def init_worker():
    from leuvenmapmatching.map.inmem import InMemMap
    from leuvenmapmatching.matcher.distance import DistanceMatcher

    with open(NETWORK_PKL, "rb") as f:
        data = pickle.load(f)
    proj_nodes = data["proj_nodes"]
    edges = data["edges"]
    edge_to_matsim = data["edge_to_matsim"]

    map_con = InMemMap(
        "slc_osm",
        use_latlon=False,
        use_rtree=True,
        index_edges=True,
    )
    for nid, (x, y) in proj_nodes.items():
        map_con.add_node(nid, (x, y))
    for (a, b) in edges:
        map_con.add_edge(a, b)

    matcher = DistanceMatcher(
        map_con,
        max_dist=MAX_DIST,
        max_dist_init=MAX_DIST,
        min_prob_norm=MIN_PROB_NORM,
        non_emitting_states=True,
        obs_noise=OBS_NOISE,
    )
    _W["matcher"] = matcher
    _W["edge_to_matsim"] = edge_to_matsim


def worker_match(payload):
    rid, lats, lons, times = payload
    matcher = _W["matcher"]
    e2m = _W["edge_to_matsim"]
    try:
        px, py = project(np.asarray(lats), np.asarray(lons))
        path = list(zip(px, py))
        res = matcher.match(path)
        states = res[0] if isinstance(res, tuple) else res
        if not states:
            return rid, None, 0, 0
        out = []
        n_unmapped = 0
        n_total_subedges = 0
        for m in matcher.lattice_best:
            if m.obs_ne != 0:
                continue
            e = m.edge_m
            if e.l2 is None:
                continue
            n_total_subedges += 1
            mtsm = e2m.get((e.l1, e.l2))
            if mtsm is None:
                n_unmapped += 1
                continue
            out.append((mtsm, float(times[m.obs])))
        if not out:
            return rid, None, n_unmapped, n_total_subedges
        return rid, out, n_unmapped, n_total_subedges
    except Exception as exc:
        return rid, ("ERR", str(exc)), 0, 0


def iter_payloads():
    pf = pq.ParquetFile(PARQUET)
    buf = pd.DataFrame()
    for batch in pf.iter_batches(batch_size=500_000):
        df = batch.to_pandas()
        if not buf.empty:
            df = pd.concat([buf, df])
        if df.empty:
            continue
        unique_ids = df["route_id"].unique()
        if len(unique_ids) <= 1:
            buf = df.copy()
            continue
        last = unique_ids[-1]
        process_ids = unique_ids[:-1]
        buf = df[df["route_id"] == last].copy()
        sub_dict = {rid: g for rid, g in df[df["route_id"] != last].groupby("route_id")}
        for rid in process_ids:
            if int(rid) % N_CHUNKS != CHUNK_INDEX:
                continue
            sub = sub_dict[rid]
            if len(sub) < MIN_GPS_FIXES:
                yield ("SKIP_SHORT", int(rid), None, None, None)
            else:
                yield (
                    "PROCESS",
                    int(rid),
                    sub["lat"].values.astype(np.float64),
                    sub["lon"].values.astype(np.float64),
                    sub["time_sec"].values.astype(np.float64),
                )
    if not buf.empty:
        for rid, sub in buf.groupby("route_id"):
            if int(rid) % N_CHUNKS != CHUNK_INDEX:
                continue
            if len(sub) < MIN_GPS_FIXES:
                yield ("SKIP_SHORT", int(rid), None, None, None)
            else:
                yield (
                    "PROCESS",
                    int(rid),
                    sub["lat"].values.astype(np.float64),
                    sub["lon"].values.astype(np.float64),
                    sub["time_sec"].values.astype(np.float64),
                )


def count_trips():
    log_line("Counting distinct route_ids (streaming pass)...")
    t0 = time.time()
    n_trips = 0
    last_id = None
    pf = pq.ParquetFile(PARQUET)
    for batch in pf.iter_batches(batch_size=1_000_000, columns=["route_id"]):
        ids = batch.column("route_id").to_numpy()
        if len(ids) == 0:
            continue
        unique_in_batch = np.unique(ids)
        n_in = len(unique_in_batch)
        if last_id is not None and last_id in set(unique_in_batch.tolist()):
            n_in -= 1
        n_trips += n_in
        last_id = int(ids[-1])
    log_line(f"  {n_trips:,} trips total in {time.time()-t0:.0f}s")
    return n_trips


def main():
    open(LOG_FILE, "w").close()
    log_line(f"Start. parquet={PARQUET}")
    log_line(f"network={NETWORK_PKL}")
    log_line(
        f"workers={N_WORKERS}, max_dist={MAX_DIST}, "
        f"min_prob_norm={MIN_PROB_NORM}, obs_noise={OBS_NOISE}, non_emitting_states=True"
    )

    if not os.path.exists(NETWORK_PKL):
        log_line(f"FATAL: {NETWORK_PKL} not found. Run build_network.py first.")
        return

    log_line(f"Pickle size: {os.path.getsize(NETWORK_PKL)/1e6:.1f} MB")

    n_total = count_trips()

    log_line(f"Spawning {N_WORKERS} workers (each loads pickle ~10-30s)...")
    t_start = time.time()
    last_log = t_start
    n_processed = n_matched = n_no = n_err = n_short = 0
    n_unmapped_total = 0
    n_subedges_total = 0

    with open(OUTPUT_JSON, "w") as f_out:
        f_out.write("[\n")
        first = True
        with Pool(processes=N_WORKERS, initializer=init_worker) as pool:

            def feed():
                nonlocal n_short
                for tag, rid, lats, lons, times in iter_payloads():
                    if tag == "SKIP_SHORT":
                        n_short += 1
                    else:
                        yield (rid, lats, lons, times)

            for rid, result, n_un, n_sub in pool.imap_unordered(
                worker_match, feed(), chunksize=8
            ):
                n_processed += 1
                n_unmapped_total += n_un
                n_subedges_total += n_sub
                if isinstance(result, tuple) and result and result[0] == "ERR":
                    n_err += 1
                elif result is None:
                    n_no += 1
                else:
                    item = {"route": {"route_id": int(rid), "fixes": result}}
                    if first:
                        first = False
                    else:
                        f_out.write(",\n")
                    json.dump(item, f_out)
                    n_matched += 1

                if n_processed % 5000 == 0 or (time.time() - last_log) > 30:
                    el = time.time() - t_start
                    rate = n_processed / max(el, 1e-3)
                    remain = max(n_total - n_processed - n_short, 0)
                    eta = remain / max(rate, 1e-3)
                    pct_unm = (
                        100.0 * n_unmapped_total / max(n_subedges_total, 1)
                    )
                    log_line(
                        f"processed={n_processed:,}/{n_total:,} matched={n_matched:,} "
                        f"no_match={n_no:,} err={n_err:,} short={n_short:,} | "
                        f"subedges={n_subedges_total:,} unmapped={n_unmapped_total:,} "
                        f"({pct_unm:.1f}%) | rate={rate:.0f} t/s "
                        f"el={el:.0f}s eta={eta:.0f}s"
                    )
                    last_log = time.time()
        f_out.write("\n]\n")

    el = time.time() - t_start
    pct_unm = 100.0 * n_unmapped_total / max(n_subedges_total, 1)
    log_line(
        f"Done in {el:.0f}s. processed={n_processed:,} matched={n_matched:,} "
        f"no_match={n_no:,} err={n_err:,} short={n_short:,} "
        f"subedges={n_subedges_total:,} unmapped={n_unmapped_total:,} ({pct_unm:.1f}%)"
    )
    log_line(f"Output: {OUTPUT_JSON}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
