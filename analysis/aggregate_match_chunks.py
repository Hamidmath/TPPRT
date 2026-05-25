"""Aggregate per-chunk matched-route JSON files into a single combined
file per obs_noise level.

The per-chunk format is the same one the matcher writes: a JSON array,
one route per line, like

    [
    {"route": {"route_id": 1, "fixes": [...]}},
    {"route": {"route_id": 9, "fixes": [...]}},
    ...
    ]

We concatenate the per-route lines from each chunk into one big array,
writing to data/matched_routes_osm_noise{N}_combined.json. The output is
still one route per line so the downstream streaming code can read it
with the same loop.

Usage:
    python analysis/aggregate_match_chunks.py
    python analysis/aggregate_match_chunks.py --noise 4
    python analysis/aggregate_match_chunks.py --noise 4 5
"""
import argparse
import glob
import os
import time

ROOT = "/home/hamid/Downloads/new/TwoPhase_PageRank_Project"
DATA = os.path.join(ROOT, "data", "map-match")


def chunk_paths_for_noise(noise):
    """Return the list of chunk JSON paths for a noise level, in deterministic
    order, skipping *.partial.json (which are preempted-run leftovers)."""
    pat = os.path.join(DATA, f"matched_routes_osm_noise{noise}_chunk*.json")
    paths = sorted(p for p in glob.glob(pat) if not p.endswith(".partial.json"))
    return paths


def aggregate(noise):
    chunks = chunk_paths_for_noise(noise)
    if not chunks:
        # Single-file noise level (already in canonical short-name form).
        candidate = os.path.join(DATA, f"noise{noise}.json")
        if not os.path.exists(candidate):
            print(f"[noise={noise}] no chunk files and no single file. skipping.")
            return
        print(f"[noise={noise}] already a single file at {candidate}; nothing to do.")
        return

    out_path = os.path.join(DATA, f"noise{noise}.json")
    print(f"[noise={noise}] {len(chunks)} chunk files -> {out_path}")
    t0 = time.time()

    total_routes = 0
    total_bytes = 0
    with open(out_path, "w") as out:
        out.write("[\n")
        first = True
        for cp in chunks:
            with open(cp) as f:
                kept = 0
                for line in f:
                    s = line.strip()
                    if s in ("[", "]", ""):
                        continue
                    if not first:
                        out.write(",\n")
                    out.write(s.rstrip(","))
                    first = False
                    kept += 1
            sz = os.path.getsize(cp)
            total_bytes += sz
            total_routes += kept
            print(f"  + {os.path.basename(cp):60s} {kept:>7,} routes "
                  f"({sz / 1e6:6.1f} MB)")
        out.write("\n]\n")

    print(f"[noise={noise}] wrote {total_routes:,} routes "
          f"({total_bytes / 1e6:.1f} MB read, "
          f"{os.path.getsize(out_path) / 1e6:.1f} MB written) "
          f"in {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", nargs="+", default=["1", "3", "4", "5"],
                    help="noise levels to aggregate (default: 1 3 4 5)")
    args = ap.parse_args()
    for n in args.noise:
        aggregate(n)


if __name__ == "__main__":
    main()
