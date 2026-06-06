import argparse
import glob
import multiprocessing
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DATE = datetime(2018, 1, 1)
MIN_SPEED = 2.0
MAX_GAP_S = 300
DUP_DIST_M = 1.0
BATCH_SIZE = 20
R_EARTH = 6_371_000

SCHEMA = pa.schema([
    ("route_id", pa.uint32()),
    ("time_sec", pa.uint32()),
    ("lat", pa.float64()),
    ("lon", pa.float64()),
])


def haversine(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R_EARTH * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def process_batch(file_paths):
    results = []
    route_counter = 0
    for path in file_paths:
        try:
            df = pd.read_csv(
                path, header=None, names=["did", "ts_str", "lat", "lon"],
                usecols=[0, 1, 2, 3],
                dtype={"did": str, "ts_str": str, "lat": float, "lon": float},
                engine="pyarrow",
            )
            ts = pd.to_datetime(df["ts_str"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
            valid = df["lat"].notna() & df["lon"].notna() & ts.notna()
            df = df[valid].copy()
            ts = ts[valid]
            if df.empty:
                continue
            df["ts"] = ts

            same_device = df["did"] == df["did"].shift(1)
            dist = haversine(df["lat"].shift(1), df["lon"].shift(1), df["lat"], df["lon"])
            dt = df["ts"].diff().dt.total_seconds()
            speed = pd.Series(np.inf, index=df.index)
            moving = same_device & (dt > 0)
            if moving.any():
                speed[moving] = dist[moving] / dt[moving]
            dup = same_device & (dt == 0)
            if dup.any():
                speed[dup & (dist < DUP_DIST_M)] = 0.0
            df = df[speed >= MIN_SPEED].copy()
            if df.empty:
                continue

            new_device = df["did"] != df["did"].shift(1)
            gap = (df["ts"].diff().dt.total_seconds() > MAX_GAP_S) & (df["did"] == df["did"].shift(1))
            groups = (new_device | gap).cumsum()
            df["route_id"] = groups + route_counter
            route_counter += int(groups.max())
            df["time_sec"] = (df["ts"] - BASE_DATE).dt.total_seconds().astype("uint32")
            results.append(df[["route_id", "time_sec", "lat", "lon"]])
        except Exception as exc:
            print(f"Error in {path}: {exc}")
    if not results:
        return None, 0
    return pd.concat(results), route_counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="Combined")
    ap.add_argument("--pattern", default="wp-snapped-*.csv")
    ap.add_argument("--output", default="compressed_routes.parquet")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No files matched {os.path.join(args.input_dir, args.pattern)}")
    batches = [files[i:i + BATCH_SIZE] for i in range(0, len(files), BATCH_SIZE)]
    print(f"Found {len(files)} files, {len(batches)} batches.")

    writer = pq.ParquetWriter(args.output, SCHEMA, compression="zstd")
    total = 0
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for n, (df, count) in enumerate(ex.map(process_batch, batches), 1):
                if df is not None:
                    df = df.copy()
                    df["route_id"] += total
                    total += count
                    writer.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
                if n % 5 == 0:
                    print(f"  batch {n}/{len(batches)}, routes: {total:,}")
    finally:
        writer.close()
    print(f"Done. {total:,} routes -> {args.output}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
