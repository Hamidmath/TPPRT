"""Find the connected anomaly region during the Utah State Fair and
plot its time-of-day profile vs reference weeks.

Pipeline:
  1. Use top 200 links by absolute lift from fair_link_anomaly.py.
  2. Restrict the road graph adjacency to that 200-link set.
  3. Find connected components (treating adjacency as undirected).
  4. Report all components with >= 10 links. The largest is the
     "anomaly region" for the Fair.
  5. For that region, sum trip counts per 5-min bin over the Fair days
     (Sep 11-13) and over the reference weeks (W1 Sep 4-6, W3 Sep
     18-20, W4 Sep 25-27). Aggregate by time-of-day (288 bins) and plot
     the Fair curve vs the reference mean +- 1 std band.

Outputs in documents/walkthrough2/results/anomalies/fair/:
  region_components.json   list of components, their link IDs, totals
  region_top_links.csv     link-level breakdown of the largest component
  tod_profile.png          time-of-day plot, Fair vs reference
  tod_profile.csv          underlying numbers
"""
import csv
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.io import load_popularity_npz

RAW_NPZ    = "data/popularity_results_osm.npz"
GRAPH_JSON = "data/city_graph_full.json"
TOP_LIFT_CSV = Path("documents/walkthrough2/results/anomalies/fair/top_lift.csv")
OUT_DIR    = Path("documents/walkthrough2/results/anomalies/fair")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_LINKS = 200
MIN_COMP    = 10

FAIR_DAYS = ["2018-09-11", "2018-09-12", "2018-09-13"]
REF_WEEKS = {
    "W1": ["2018-09-04", "2018-09-05", "2018-09-06"],
    "W3": ["2018-09-18", "2018-09-19", "2018-09-20"],
    "W4": ["2018-09-25", "2018-09-26", "2018-09-27"],
}


def load_top_lift(n):
    rows = list(csv.DictReader(open(TOP_LIFT_CSV)))
    rows = rows[:n]
    return rows


def build_undirected_adj(graph, restrict_ids):
    """Build undirected adjacency restricted to restrict_ids."""
    keep = set(str(x) for x in restrict_ids)
    adj = defaultdict(set)
    for src, succ_list in graph["adjacency"].items():
        if src not in keep:
            continue
        for dst in succ_list:
            if dst in keep:
                adj[src].add(dst)
                adj[dst].add(src)
    return adj, keep


def connected_components(adj, all_ids):
    seen = set()
    comps = []
    for v in all_ids:
        if v in seen:
            continue
        comp = []
        q = deque([v]); seen.add(v)
        while q:
            u = q.popleft()
            comp.append(u)
            for w in adj.get(u, ()):
                if w not in seen:
                    seen.add(w); q.append(w)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def sum_per_bin_for_links(M, times, link_idx_set, day_list):
    """Return ndarray (288,) of total counts per time-of-day bin over
    the given day_list, restricted to link_idx_set."""
    by_tod = np.zeros(288, dtype=np.float64)
    n_days = 0
    day_set = set(day_list)
    seen_days = set()
    cols = sorted(link_idx_set)
    Msub = M[:, cols]
    for i, t in enumerate(times):
        d = t[:10]
        if d not in day_set:
            continue
        seen_days.add(d)
        hour = int(t[11:13]); minute = int(t[14:16])
        tod = hour * 12 + minute // 5
        by_tod[tod] += float(Msub.getrow(i).sum())
    n_days = len(seen_days)
    return by_tod, n_days


def main():
    print("[start] fair region profile")

    top_rows = load_top_lift(TOP_K_LINKS)
    top_ids = [r["link_id"] for r in top_rows]
    print(f"  loaded top {len(top_ids)} links by absolute lift")

    graph = json.load(open(GRAPH_JSON))
    adj, keep = build_undirected_adj(graph, top_ids)
    comps = connected_components(adj, top_ids)
    big_comps = [c for c in comps if len(c) >= MIN_COMP]
    print(f"  connected components (>= {MIN_COMP} links): {len(big_comps)}")
    print(f"  sizes: {[len(c) for c in comps[:10]]} ...")

    # Per-link table for the LARGEST component
    if big_comps:
        largest = set(big_comps[0])
        rows = [r for r in top_rows if r["link_id"] in largest]
        rows.sort(key=lambda r: float(r["lift"]), reverse=True)
        with open(OUT_DIR / "region_top_links.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n  largest component: {len(largest)} links")
        print(f"  saved region_top_links.csv")
        print(f"  preview of largest component (top 10 by lift):")
        for r in rows[:10]:
            print(f"    link {r['link_id']:>6s}  lift={float(r['lift']):+.0f}  "
                  f"fair={r['fair']}  ref_med={float(r['ref_median']):.0f}  "
                  f"lanes={r['lanes']}  speed={r['speed']}")

    # Save all components summary
    comp_summary = []
    for ci, c in enumerate(comps):
        lifts = [float(r["lift"]) for r in top_rows if r["link_id"] in set(c)]
        fairs = [int(r["fair"])  for r in top_rows if r["link_id"] in set(c)]
        comp_summary.append(dict(
            component=ci, n_links=len(c),
            total_fair_trips=int(sum(fairs)),
            total_lift=float(sum(lifts)),
            mean_lift=float(np.mean(lifts)) if lifts else 0.0,
            link_ids=list(c),
        ))
    (OUT_DIR / "region_components.json").write_text(
        json.dumps(comp_summary, indent=2))
    print(f"  saved region_components.json ({len(comp_summary)} components)")

    # ---- Time-of-day profile for the largest component ----
    if not big_comps:
        print("  no component >=10 links, skipping TOD profile")
        return

    b = load_popularity_npz(RAW_NPZ)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    link_ids_all = [str(x) for x in b["link_ids"]]
    lid_to_idx = {lid: i for i, lid in enumerate(link_ids_all)}
    region_link_idxs = set(lid_to_idx[lid] for lid in big_comps[0]
                            if lid in lid_to_idx)
    print(f"\n  computing TOD profile over {len(region_link_idxs)} region links")

    fair_tod, _ = sum_per_bin_for_links(M, times, region_link_idxs, FAIR_DAYS)
    fair_per_day = fair_tod / 3.0
    ref_curves = {}
    for k, days in REF_WEEKS.items():
        tod, nd = sum_per_bin_for_links(M, times, region_link_idxs, days)
        ref_curves[k] = tod / max(nd, 1)
    ref_stack = np.stack([ref_curves["W1"], ref_curves["W3"], ref_curves["W4"]],
                          axis=0)
    ref_mean = ref_stack.mean(axis=0)
    ref_std = ref_stack.std(axis=0)

    x = np.arange(288) * 5 / 60.0  # hours
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.fill_between(x, ref_mean - ref_std, ref_mean + ref_std,
                     color="lightgrey", alpha=0.7,
                     label="reference (W1/W3/W4) mean +/- 1 std")
    ax.plot(x, ref_mean, color="dimgrey", lw=1.3,
             label="reference mean (per weekday)")
    ax.plot(x, fair_per_day, color="crimson", lw=1.6,
             label="Fair window (Sep 11-13) per weekday")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("time of day (h)")
    ax.set_ylabel(f"trips per 5-min bin, summed over {len(region_link_idxs)} region links")
    ax.set_title("Fair-window anomaly region: time-of-day profile")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tod_profile.png", dpi=140)
    plt.close(fig)
    print(f"  saved tod_profile.png")

    with open(OUT_DIR / "tod_profile.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "hour_dec", "fair_per_day",
                    "ref_W1", "ref_W3", "ref_W4",
                    "ref_mean", "ref_std", "lift"])
        for i in range(288):
            w.writerow([i, x[i],
                        fair_per_day[i], ref_curves["W1"][i],
                        ref_curves["W3"][i], ref_curves["W4"][i],
                        ref_mean[i], ref_std[i],
                        fair_per_day[i] - ref_mean[i]])
    print(f"  saved tod_profile.csv")

    # Peak-lift hours
    diff = fair_per_day - ref_mean
    top_bins = np.argsort(diff)[::-1][:12]
    print(f"\n  top time-of-day bins by Fair lift (per weekday):")
    for b_i in top_bins:
        hh = b_i // 12; mm = (b_i % 12) * 5
        print(f"    {hh:02d}:{mm:02d}  fair={fair_per_day[b_i]:7.1f}  "
              f"ref={ref_mean[b_i]:7.1f}  lift={diff[b_i]:+.1f}")

    print(f"\n[done] outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
