"""Build the trip-ORIGINS matrix from data/map-match/noise3p5.json.

Per-trip rule (paper §2.3 specialized to the first fix only):

    For each route r, take its FIRST map-matched fix (lid_first, t_first).
    Let bin = floor((t_abs - t_0) / 300) where
        t_0       = datetime(2018, 1, 1) in local time,
        t_abs     = t_0 + t_first (the timestamp field in the matched JSON
                    is seconds elapsed since 2018-01-01 00:00:00 local).
    Increment O[bin, lid_first] by 1.

Each trip contributes exactly one cell.

Streaming reader. The 1.3 GB file is one route per line inside the JSON
array, so we iterate lines, skip the [ and ] boundary lines, strip the
trailing comma, and parse each route on its own. RAM stays bounded by
the size of the running counter, not the file.

Output: data/origins_results.npz with the CSR keys read by
core.io.load_popularity_npz:
    matrix_data, matrix_indices, matrix_indptr, matrix_shape, times, link_ids
"""
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
MATCHED = ROOT / "data" / "map-match" / "noise3p5.json"
OUT     = ROOT / "data" / "origins_results.npz"
POP_NPZ = ROOT / "data" / "popularity_results_osm.npz"

BIN_SIZE = 300
BASE_TS  = datetime(2018, 1, 1).timestamp()


def main():
    print(f"[start] streaming {MATCHED}")
    print(f"  BIN_SIZE = {BIN_SIZE} s")
    print(f"  BASE_TS  = {BASE_TS} ({datetime.fromtimestamp(BASE_TS)})")

    counts = Counter()        # (bin_label, lid_str) -> int
    n_routes = 0              # routes with at least one fix
    n_empty  = 0              # routes with no fixes
    n_bad    = 0              # parse errors (should be 0)
    n_lines  = 0
    t0 = time.time()

    with open(MATCHED, "r") as f:
        for raw in f:
            n_lines += 1
            s = raw.strip()
            if s in ("", "[", "]"):
                continue
            if s.endswith(","):
                s = s[:-1]
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            route = obj.get("route", obj)
            fixes = route.get("fixes") or route.get("routelinks")
            if not fixes:
                n_empty += 1
                continue
            first = fixes[0]
            lid = str(first[0])
            t   = float(first[1])
            t_abs = BASE_TS + t
            b = int(t_abs // BIN_SIZE)
            bin_label = datetime.fromtimestamp(b * BIN_SIZE).strftime(
                "%Y-%m-%d %H:%M:%S")
            counts[(bin_label, lid)] += 1
            n_routes += 1
            if n_routes % 100_000 == 0:
                print(f"  routes counted: {n_routes:>9,d}   "
                      f"unique cells: {len(counts):>8,d}   "
                      f"elapsed: {time.time()-t0:6.1f}s")

    print(f"\n[stream done] {time.time()-t0:.1f}s, "
          f"{n_lines:,} lines read, "
          f"{n_routes:,} routes counted, "
          f"{n_empty:,} empty, "
          f"{n_bad:,} bad JSON")
    print(f"  unique (bin, lid) cells: {len(counts):,}")
    total = sum(counts.values())
    print(f"  total cell sum (= total trips counted): {total:,}")
    assert total == n_routes, f"sum mismatch {total} vs {n_routes}"

    # Build sorted axes (drop all-zero rows/cols implicitly: only seen ones).
    times    = sorted({k[0] for k in counts})
    link_ids = sorted({k[1] for k in counts})
    t2r = {t: i for i, t in enumerate(times)}
    l2c = {l: i for i, l in enumerate(link_ids)}
    print(f"  unique bins seen : {len(times):,}")
    print(f"  unique links seen: {len(link_ids):,}")

    rows, cols, vals = [], [], []
    for (tlabel, lid), n in counts.items():
        rows.append(t2r[tlabel])
        cols.append(l2c[lid])
        vals.append(n)
    M = coo_matrix((vals, (rows, cols)),
                    shape=(len(times), len(link_ids)),
                    dtype=np.int32).tocsr()
    print(f"\n[matrix] shape={M.shape}  nnz={M.nnz:,}  sum={int(M.sum()):,}")
    assert int(M.sum()) == n_routes

    # Cross-check bin labels against popularity matrix.
    sys.path.insert(0, str(ROOT))
    from core.io import load_popularity_npz
    pop = load_popularity_npz(str(POP_NPZ))
    pop_times = set(str(t) for t in pop["times"])
    pop_lids  = set(str(x) for x in pop["link_ids"])
    extra_times = set(times) - pop_times
    extra_lids  = set(link_ids) - pop_lids
    print(f"  bins in origins but not in popularity: {len(extra_times)}  "
          f"(expect 0)")
    print(f"  lids in origins but not in popularity: {len(extra_lids)}  "
          f"(expect 0; popularity is a superset of origins)")
    if extra_times:
        print(f"  example extra times: {sorted(extra_times)[:5]}")
    if extra_lids:
        print(f"  example extra lids:  {sorted(extra_lids)[:5]}")

    # Top 10 most-popular origin links across the corpus.
    col_sums = np.asarray(M.sum(axis=0)).ravel()
    top_idx = np.argsort(-col_sums)[:10]
    print(f"\n  top 10 origin links (over full 30-day corpus):")
    for c in top_idx:
        print(f"    lid {link_ids[c]:>8s}  total trip-starts {int(col_sums[c]):>6,d}")

    # Save.
    np.savez_compressed(
        OUT,
        matrix_data=M.data,
        matrix_indices=M.indices,
        matrix_indptr=M.indptr,
        matrix_shape=np.array(M.shape, dtype=np.int64),
        times=np.array(times),
        link_ids=np.array(link_ids),
    )
    print(f"\n[saved] {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
