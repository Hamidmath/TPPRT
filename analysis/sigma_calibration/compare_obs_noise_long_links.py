"""Compare matched-route coverage at obs_noise=1 (baseline) vs obs_noise=3
on the SLC low-resolution MATSim XML network.

Per-link metric:
  trips(lid) = number of distinct route_ids that touched link `lid`.

Bucket every link by its `length` attribute (quintiles + an explicit
top-1% "very long" tier). Report, per bucket:
  - n_links: how many links live in this bucket
  - covered_n1: how many of them have trips >= 1 under noise=1
  - covered_n3: how many of them have trips >= 1 under noise=3
  - coverage_n1: covered_n1 / n_links
  - coverage_n3: covered_n3 / n_links
  - delta_coverage: coverage_n3 - coverage_n1
  - mean_trips_n1, mean_trips_n3: how many trips touch each link on
    average, in this bucket
  - mean_trips_ratio: mean_trips_n3 / mean_trips_n1

Headline question: does noise=3 cover long links better than noise=1?
"""
import json
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
XML = os.path.join(ROOT, "data", "slc_network.xml")
NOISE1 = os.path.join(ROOT, "data", "map-match", "noise1.json")
NOISE3 = os.path.join(ROOT, "data", "map-match", "noise3.json")


def load_link_lengths(xml_path):
    print(f"[xml] parsing {xml_path}")
    t0 = time.time()
    lens = {}
    for _, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag.endswith("link"):
            lid = elem.attrib.get("id")
            length = elem.attrib.get("length")
            if lid is not None and length is not None:
                lens[lid] = float(length)
            elem.clear()
    print(f"[xml] {len(lens):,} links in {time.time() - t0:.1f}s")
    return lens


def count_trips_per_link(json_path):
    """Streaming-ish loader: one JSON object per line after the leading '['.
    For each route, take the *set* of links it touched (distinct lids), and
    increment that link's counter by one. Returns {lid: int}."""
    print(f"[json] loading {json_path}")
    t0 = time.time()
    counts = defaultdict(int)
    with open(json_path) as f:
        # File looks like:
        #   [
        #   {"route": {...}},
        #   {"route": {...}},
        #   ...
        #   ]
        # We parse line by line, stripping trailing comma/whitespace.
        for i, line in enumerate(f):
            line = line.strip().rstrip(",")
            if line in ("[", "]", ""):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            route = obj.get("route", {})
            fixes = route.get("fixes", [])
            distinct_lids = {fix[0] for fix in fixes}
            for lid in distinct_lids:
                counts[lid] += 1
            if (i + 1) % 100000 == 0:
                print(f"  ... {i + 1:,} lines, {len(counts):,} distinct links so far")
    print(f"[json] {sum(counts.values()):,} route-link touches across "
          f"{len(counts):,} links in {time.time() - t0:.1f}s")
    return counts


def bucket(lengths, q):
    """Return a function lid -> bucket label, where buckets are quintiles
    of length plus a top-1% 'very long' bucket."""
    edges = np.quantile(list(lengths.values()), q)
    # edges[i] = quantile q[i]; bucket boundaries
    def labeller(lid):
        L = lengths[lid]
        for i, e in enumerate(edges):
            if L <= e:
                return i, q[i], e
        return len(edges), 1.0, max(lengths.values())
    return labeller, edges


def report(lengths, c1, c3):
    qs = [0.20, 0.40, 0.60, 0.80, 0.99, 1.0]
    bucket_names = [
        "shortest (0-20%)",
        "short    (20-40%)",
        "medium   (40-60%)",
        "long     (60-80%)",
        "longer   (80-99%)",
        "longest  (99-100%)",
    ]
    edges = np.quantile(list(lengths.values()), qs)

    print()
    print("=" * 108)
    print("  COVERAGE BY LINK LENGTH (low-resolution MATSim XML network, "
          "99,716 car links)")
    print("=" * 108)
    print(f"  {'bucket':<22s} {'len <=':>10s}  {'n_links':>8s}  "
          f"{'cov n=1':>9s}  {'cov n=3':>9s}  {'delta':>7s}  "
          f"{'tpl n=1':>9s}  {'tpl n=3':>9s}  {'tpl x':>7s}")
    print("  " + "-" * 104)

    by_bucket = defaultdict(list)
    for lid, L in lengths.items():
        for i, e in enumerate(edges):
            if L <= e:
                by_bucket[i].append(lid)
                break

    for i, name in enumerate(bucket_names):
        lids = by_bucket[i]
        n = len(lids)
        cov1 = sum(1 for lid in lids if c1.get(lid, 0) > 0)
        cov3 = sum(1 for lid in lids if c3.get(lid, 0) > 0)
        tpl1 = np.mean([c1.get(lid, 0) for lid in lids])
        tpl3 = np.mean([c3.get(lid, 0) for lid in lids])
        delta = (cov3 - cov1) / n if n else 0.0
        ratio = (tpl3 / tpl1) if tpl1 > 0 else float("nan")
        print(f"  {name:<22s} {edges[i]:>10.1f}  "
              f"{n:>8,d}  {cov1/n:>9.3f}  {cov3/n:>9.3f}  "
              f"{delta:>+7.3f}  {tpl1:>9.2f}  {tpl3:>9.2f}  "
              f"{ratio:>7.2f}")

    print()
    print("  bucket = quintile of `length` attribute in the MATSim XML "
          "(top tier = top 1%).")
    print("  cov    = fraction of links in the bucket with at least one "
          "matched trip.")
    print("  tpl    = mean trips per link in the bucket "
          "(0 counted; distinct route_ids per link).")
    print("  delta  = cov_n3 - cov_n1 (positive means noise=3 covers more "
          "links in the bucket).")
    print("  tpl x  = tpl_n3 / tpl_n1 (how many times more trips per link "
          "at noise=3 versus noise=1).")


def main():
    lengths = load_link_lengths(XML)
    c1 = count_trips_per_link(NOISE1)
    c3 = count_trips_per_link(NOISE3)
    report(lengths, c1, c3)

    # quick global headline
    total_links = len(lengths)
    cov_total_1 = sum(1 for lid in lengths if c1.get(lid, 0) > 0)
    cov_total_3 = sum(1 for lid in lengths if c3.get(lid, 0) > 0)
    print()
    print("  GLOBAL:")
    print(f"    coverage at noise=1: {cov_total_1:,} / {total_links:,} "
          f"links ({cov_total_1/total_links*100:.1f}%)")
    print(f"    coverage at noise=3: {cov_total_3:,} / {total_links:,} "
          f"links ({cov_total_3/total_links*100:.1f}%)")
    print(f"    extra links covered by noise=3 only: "
          f"{cov_total_3 - cov_total_1:,}")


if __name__ == "__main__":
    main()
