"""Link-level anomaly detection for Utah State Fair 2018.

For every OSM link we compare its trip count during the Fair window
(Tue-Thu Sep 11-13, the three weekdays with noon gate openings) against
the same Tue-Thu weekdays in three reference weeks (W1 Sep 4-6, W3 Sep
18-20, W4 Sep 25-27).

For each link i:
  fair_count(i)     = sum of c_t(i) over bins in Sep 11-13 24h
  ref_count_k(i)    = sum of c_t(i) over bins in reference week k, same Tue-Thu
  ref_median(i)     = median over k in {W1, W3, W4} of ref_count_k(i)
  lift(i)           = fair_count(i) - ref_median(i)
  fold(i)           = fair_count(i) / max(ref_median(i), 1)

We then filter on a minimum reference floor (so a 0 -> 5 link does not
dominate the ranking) and rank by (a) absolute lift and (b) fold change.

Outputs in documents/walkthrough2/results/anomalies/fair/:
  per_link.csv         every link with fair_count, ref counts, lift, fold
  top_lift.csv         top 200 by absolute lift, sorted desc
  top_fold.csv         top 200 by fold change (with ref_median >= 20)
  hist_lift.png        histogram of lift over all links with ref_median>=5
  hist_fold.png        histogram of fold (log x) over the same
  summary.json         counts, thresholds, totals
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.io import load_popularity_npz

RAW_NPZ    = "data/popularity_results_osm.npz"
GRAPH_JSON = "data/city_graph_full.json"
OUT_DIR    = Path("documents/walkthrough2/results/anomalies/fair")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FAIR_DAYS = ["2018-09-11", "2018-09-12", "2018-09-13"]
REF_WEEKS = {
    "W1": ["2018-09-04", "2018-09-05", "2018-09-06"],
    "W3": ["2018-09-18", "2018-09-19", "2018-09-20"],
    "W4": ["2018-09-25", "2018-09-26", "2018-09-27"],
}

REF_FLOOR_HIST = 5      # min ref_median to enter the histograms
REF_FLOOR_FOLD = 20     # min ref_median to be eligible for the fold ranking
TOP_N          = 200


def sum_over_days(M, times, day_list):
    """Return ndarray of shape (N_links,): sum of counts over all bins
    whose timestamp starts with one of the given calendar dates."""
    idx = [i for i, t in enumerate(times) if t[:10] in set(day_list)]
    if not idx:
        return np.zeros(M.shape[1], dtype=np.float64)
    sub = M[idx]
    return np.asarray(sub.sum(axis=0)).ravel().astype(np.float64)


def main():
    print("[start] fair-link anomaly detection")
    print(f"  fair window: {FAIR_DAYS}")
    print(f"  reference weeks: {list(REF_WEEKS.keys())}")
    b = load_popularity_npz(RAW_NPZ)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    link_ids = [str(x) for x in b["link_ids"]]
    N = M.shape[1]
    print(f"  matrix: {M.shape}, nnz={M.nnz:,}")

    fair = sum_over_days(M, times, FAIR_DAYS)
    refs = {k: sum_over_days(M, times, d) for k, d in REF_WEEKS.items()}
    ref_stack = np.stack([refs["W1"], refs["W3"], refs["W4"]], axis=1)
    ref_median = np.median(ref_stack, axis=1)
    ref_mean   = ref_stack.mean(axis=1)
    ref_min    = ref_stack.min(axis=1)
    ref_max    = ref_stack.max(axis=1)

    lift = fair - ref_median
    fold = fair / np.maximum(ref_median, 1.0)

    print(f"  total Fair  Tue-Thu trips: {int(fair.sum()):>9,}")
    for k, v in refs.items():
        print(f"  total {k} Tue-Thu trips: {int(v.sum()):>9,}")

    graph = json.load(open(GRAPH_JSON))
    g_links = graph["links"]

    def attrs(lid):
        d = g_links.get(lid, {})
        return d.get("speed"), d.get("lanes"), d.get("length")

    rows = []
    for i in range(N):
        lid = link_ids[i]
        sp, ln, le = attrs(lid)
        rows.append(dict(
            link_id=lid,
            link_idx=i,
            fair=int(fair[i]),
            ref_W1=int(refs["W1"][i]),
            ref_W3=int(refs["W3"][i]),
            ref_W4=int(refs["W4"][i]),
            ref_median=float(ref_median[i]),
            ref_mean=float(ref_mean[i]),
            ref_min=int(ref_min[i]),
            ref_max=int(ref_max[i]),
            lift=float(lift[i]),
            fold=float(fold[i]),
            speed=sp, lanes=ln, length=le,
        ))

    keys = list(rows[0].keys())
    with open(OUT_DIR / "per_link.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  saved per_link.csv ({len(rows):,} rows)")

    # --- histograms ---
    hist_mask = ref_median >= REF_FLOOR_HIST
    print(f"\n  links with ref_median >= {REF_FLOOR_HIST}: {hist_mask.sum():,}")
    lifts_h = lift[hist_mask]
    folds_h = fold[hist_mask]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(lifts_h, bins=80, color="steelblue", edgecolor="white")
    ax.axvline(0, color="black", lw=1.0)
    ax.set_xlabel("lift = fair_count - ref_median (trips per link, 3 days)")
    ax.set_ylabel("number of links (ref_median >= 5)")
    ax.set_title("Fair-window lift per link  (Sep 11-13 vs median of W1/W3/W4)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hist_lift.png", dpi=140)
    plt.close(fig)
    print(f"  saved hist_lift.png  median lift = {np.median(lifts_h):+.2f}  "
          f"p95 lift = {np.percentile(lifts_h, 95):+.2f}")

    fold_pos = np.clip(folds_h, 1e-2, None)
    bins = np.logspace(np.log10(0.05), np.log10(max(fold_pos.max(), 10)), 60)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(fold_pos, bins=bins, color="seagreen", edgecolor="white")
    ax.axvline(1.0, color="black", lw=1.0, label="fold = 1 (same as normal)")
    ax.set_xscale("log")
    ax.set_xlabel("fold = fair_count / max(ref_median, 1)  (log scale)")
    ax.set_ylabel("number of links (ref_median >= 5)")
    ax.set_title("Fair-window fold change per link")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hist_fold.png", dpi=140)
    plt.close(fig)
    print(f"  saved hist_fold.png  median fold = {np.median(folds_h):.3f}  "
          f"p95 fold = {np.percentile(folds_h, 95):.3f}  "
          f"p99 fold = {np.percentile(folds_h, 99):.3f}")

    # --- top by absolute lift ---
    rows_h = [r for r in rows if r["ref_median"] >= REF_FLOOR_HIST]
    by_lift = sorted(rows_h, key=lambda r: r["lift"], reverse=True)[:TOP_N]
    with open(OUT_DIR / "top_lift.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in by_lift:
            w.writerow(r)
    print(f"\n  top 10 by absolute lift (ref_median >= {REF_FLOOR_HIST}):")
    for r in by_lift[:10]:
        print(f"    link {r['link_id']:>6s}  fair={r['fair']:>4d}  "
              f"refs=({r['ref_W1']},{r['ref_W3']},{r['ref_W4']})  "
              f"ref_med={r['ref_median']:.0f}  lift={r['lift']:+.0f}  "
              f"lanes={r['lanes']}  speed={r['speed']:.1f}")

    # --- top by fold (require ref >= 20 so a 1 -> 30 doesn't dominate) ---
    rows_f = [r for r in rows if r["ref_median"] >= REF_FLOOR_FOLD]
    by_fold = sorted(rows_f, key=lambda r: r["fold"], reverse=True)[:TOP_N]
    with open(OUT_DIR / "top_fold.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in by_fold:
            w.writerow(r)
    print(f"\n  top 10 by fold change (ref_median >= {REF_FLOOR_FOLD}):")
    for r in by_fold[:10]:
        print(f"    link {r['link_id']:>6s}  fair={r['fair']:>4d}  "
              f"refs=({r['ref_W1']},{r['ref_W3']},{r['ref_W4']})  "
              f"ref_med={r['ref_median']:.0f}  fold={r['fold']:.2f}  "
              f"lanes={r['lanes']}  speed={r['speed']:.1f}  "
              f"length={r['length']:.0f}")

    summary = dict(
        fair_days=FAIR_DAYS,
        ref_weeks=REF_WEEKS,
        n_links_total=N,
        n_links_with_ref_floor_hist=int(hist_mask.sum()),
        n_links_with_ref_floor_fold=int((ref_median >= REF_FLOOR_FOLD).sum()),
        ref_floor_hist=REF_FLOOR_HIST,
        ref_floor_fold=REF_FLOOR_FOLD,
        top_n=TOP_N,
        totals=dict(fair_count=int(fair.sum()),
                     ref_W1=int(refs["W1"].sum()),
                     ref_W3=int(refs["W3"].sum()),
                     ref_W4=int(refs["W4"].sum())),
        lift_stats=dict(median=float(np.median(lifts_h)),
                         mean=float(lifts_h.mean()),
                         p95=float(np.percentile(lifts_h, 95)),
                         p99=float(np.percentile(lifts_h, 99))),
        fold_stats=dict(median=float(np.median(folds_h)),
                         p95=float(np.percentile(folds_h, 95)),
                         p99=float(np.percentile(folds_h, 99))),
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
