"""Aggregate per-chunk matched-routes JSON files into one combined JSON
in the SAME line-by-line format as each chunk (one route per line,
comma-separated, wrapped in [ ]).

Env vars:
    NOISE_TAG     required, e.g. "3p5", "4", "1p75"
    RESULTS_DIR   optional, default $HOME/tppr/results
"""
import glob, json, os, sys, time

def main():
    noise = os.environ.get("NOISE_TAG")
    if not noise:
        print("ERROR: NOISE_TAG required", file=sys.stderr); sys.exit(1)
    results_dir = os.environ.get(
        "RESULTS_DIR", os.path.expanduser("~/tppr/results"))
    pat = os.path.join(
        results_dir, f"matched_routes_osm_noise{noise}_chunk*.json")
    chunks = sorted(p for p in glob.glob(pat) if not p.endswith(".partial"))
    if not chunks:
        print(f"[aggregate] no chunks matching {pat}", file=sys.stderr); sys.exit(1)
    out_path = os.path.join(
        results_dir, f"matched_routes_osm_noise{noise}_combined.json")
    print(f"[aggregate] noise={noise}: {len(chunks)} chunks -> {out_path}",
          flush=True)
    t0 = time.time()
    total = 0
    with open(out_path, "w") as out:
        out.write("[\n")
        first = True
        for i, p in enumerate(chunks):
            try:
                routes = json.load(open(p))
            except json.JSONDecodeError as e:
                print(f"CHUNK {p} BROKEN: {e}", file=sys.stderr); sys.exit(2)
            if not isinstance(routes, list):
                print(f"CHUNK {p} not a list", file=sys.stderr); sys.exit(2)
            for rec in routes:
                if not first:
                    out.write(",\n")
                out.write(json.dumps(rec))
                first = False
            total += len(routes)
            if (i + 1) % 8 == 0 or i == len(chunks) - 1:
                print(f"  {i+1:>3}/{len(chunks)} merged, total={total:,} "
                      f"({time.time()-t0:.1f}s)", flush=True)
        out.write("\n]\n")
    print(f"[aggregate] done: {total:,} routes in {time.time()-t0:.1f}s",
          flush=True)

if __name__ == "__main__":
    main()
