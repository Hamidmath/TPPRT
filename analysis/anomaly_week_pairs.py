"""Week-over-week anomaly detection on raw map-matched counts.

For each pair of consecutive weeks (1->2, 2->3, 3->4) compare every
5-min slot t with its twin t+7d. Per slot, compute

    MRE(t) = mean_{i in S(t)} |c_{t+7d}(i) - c_t(i)| / c_t(i),
    S(t)   = { links i : c_t(i) >= FLOOR }.

The floor c_t(i) >= 5 drops low-baseline noise (a side street going
1->3 trips would otherwise dominate the ranking).

Outputs in documents/walkthrough2/results/anomalies/:
  hist_w1w2.png, hist_w2w3.png, hist_w3w4.png
  top24_anomalies.json   (top 24 slot timestamps + MRE per comparison)
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.io import load_popularity_npz

RAW_NPZ = "data/popularity_results_osm.npz"
OUT_DIR = Path("documents/walkthrough2/results/anomalies")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLOOR = 5
MIN_LINKS_RANKED = 50
SLOTS_PER_WEEK = 7 * 24 * 12  # 2016
N_WEEKS = 4
TOP_K = 24


def compute_pair_mre(M, base_idx, tgt_idx):
    """Return MRE for a single slot pair, plus n_links used."""
    base = M.getrow(base_idx).toarray().ravel().astype(np.float64)
    tgt = M.getrow(tgt_idx).toarray().ravel().astype(np.float64)
    mask = base >= FLOOR
    n = int(mask.sum())
    if n == 0:
        return np.nan, 0
    rel = np.abs(tgt[mask] - base[mask]) / base[mask]
    return float(rel.mean()), n


def run_comparison(M, times, week_base, week_target):
    """Compare slots of week_base to week_target. Returns list of dicts."""
    base0 = week_base * SLOTS_PER_WEEK
    tgt0 = week_target * SLOTS_PER_WEEK
    rows = []
    for k in range(SLOTS_PER_WEEK):
        bi, ti = base0 + k, tgt0 + k
        if ti >= len(times):
            break
        mre, n = compute_pair_mre(M, bi, ti)
        rows.append({
            "k": k,
            "t_base": times[bi],
            "t_target": times[ti],
            "mre": mre,
            "n_links": n,
        })
    return rows


def plot_hist(rows, title, out_path):
    mres = np.array([r["mre"] for r in rows if not np.isnan(r["mre"])])
    if mres.size == 0:
        return
    med = float(np.median(mres))
    p95 = float(np.percentile(mres, 95))
    p99 = float(np.percentile(mres, 99))

    lo = max(1e-3, mres.min())
    hi = mres.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 60)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(mres, bins=bins, color="steelblue", edgecolor="white")
    ax.set_xscale("log")
    ax.axvline(med, color="black", lw=1.0, ls="-", label=f"median = {med:.3f}")
    ax.axvline(p95, color="darkorange", lw=1.0, ls="--", label=f"p95 = {p95:.3f}")
    ax.axvline(p99, color="crimson", lw=1.0, ls=":", label=f"p99 = {p99:.3f}")
    ax.set_xlabel("per-slot MRE (log scale)")
    ax.set_ylabel("number of 5-min slots")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    n_zero = sum(1 for r in rows if r["n_links"] == 0)
    print(f"  saved {out_path.name}   n={mres.size}  "
          f"median={med:.3f}  p95={p95:.3f}  p99={p99:.3f}  "
          f"zero-baseline slots dropped: {n_zero}")


def plot_hist_thick(rows, title, out_path, min_links, top_k):
    """Histogram restricted to slots with n_links >= min_links, marking
    the top-k cutoff."""
    thick = [r for r in rows
             if (not np.isnan(r["mre"])) and r["n_links"] >= min_links]
    if not thick:
        print(f"  skipped {out_path.name}: no slots pass n_links >= {min_links}")
        return
    mres = np.array([r["mre"] for r in thick])
    med = float(np.median(mres))
    p95 = float(np.percentile(mres, 95))
    p99 = float(np.percentile(mres, 99))

    sorted_desc = np.sort(mres)[::-1]
    cutoff = float(sorted_desc[min(top_k - 1, len(sorted_desc) - 1)])

    lo = max(1e-3, mres.min())
    hi = mres.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 50)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(mres, bins=bins, color="seagreen", edgecolor="white")
    ax.set_xscale("log")
    ax.axvline(med, color="black", lw=1.0, ls="-", label=f"median = {med:.3f}")
    ax.axvline(p95, color="darkorange", lw=1.0, ls="--", label=f"p95 = {p95:.3f}")
    ax.axvline(p99, color="crimson", lw=1.0, ls=":", label=f"p99 = {p99:.3f}")
    ax.axvline(cutoff, color="purple", lw=1.2, ls="-.",
                label=f"top-{top_k} cutoff = {cutoff:.3f}")
    ax.set_xlabel("per-slot MRE (log scale)")
    ax.set_ylabel(f"number of 5-min slots (n_links >= {min_links})")
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved {out_path.name}   n_thick={len(mres)}  "
          f"median={med:.3f}  p95={p95:.3f}  p99={p99:.3f}  cutoff={cutoff:.3f}")


def main():
    print(f"[start] anomaly week-pairs, floor >= {FLOOR}")
    b = load_popularity_npz(RAW_NPZ)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    print(f"  matrix: {M.shape}  nnz={M.nnz:,}")
    print(f"  range:  {times[0]}  ->  {times[-1]}")
    print(f"  slots per week: {SLOTS_PER_WEEK}")

    comparisons = [(0, 1, "w1w2"), (1, 2, "w2w3"), (2, 3, "w3w4")]
    out = {"floor": FLOOR, "min_links_ranked": MIN_LINKS_RANKED,
           "n_links_total": M.shape[1],
           "raw_data": RAW_NPZ, "comparisons": {}}

    for wb, wt, tag in comparisons:
        print(f"\n  comparison {tag}: week {wb + 1} (base) -> week {wt + 1} (target)")
        rows = run_comparison(M, times, wb, wt)
        title = (f"Week {wb + 1} vs Week {wt + 1}: per-slot MRE "
                 f"(raw counts, baseline >= {FLOOR})")
        plot_hist(rows, title, OUT_DIR / f"hist_{tag}.png")
        title_thick = (f"Week {wb + 1} vs Week {wt + 1}: per-slot MRE "
                       f"(baseline >= {FLOOR}, n_links >= {MIN_LINKS_RANKED})")
        plot_hist_thick(rows, title_thick,
                         OUT_DIR / f"hist_{tag}_thick.png",
                         MIN_LINKS_RANKED, TOP_K)

        valid = [r for r in rows if not np.isnan(r["mre"])]
        valid.sort(key=lambda r: r["mre"], reverse=True)
        top_unfiltered = valid[:TOP_K]

        thick = [r for r in valid if r["n_links"] >= MIN_LINKS_RANKED]
        top_thick = thick[:TOP_K]
        print(f"    top {TOP_K} (unfiltered):")
        for r in top_unfiltered[:5]:
            print(f"      {r['t_base']} -> {r['t_target']}  "
                  f"MRE={r['mre']:.3f}  n_links={r['n_links']}")
        print(f"    top {TOP_K} (n_links >= {MIN_LINKS_RANKED}):")
        for r in top_thick[:10]:
            print(f"      {r['t_base']} -> {r['t_target']}  "
                  f"MRE={r['mre']:.3f}  n_links={r['n_links']}")

        out["comparisons"][tag] = {
            "week_base": wb + 1, "week_target": wt + 1,
            "n_slots": len(valid),
            "n_slots_thick": len(thick),
            "median_mre": float(np.median([r["mre"] for r in valid])),
            "p95_mre": float(np.percentile([r["mre"] for r in valid], 95)),
            "p99_mre": float(np.percentile([r["mre"] for r in valid], 99)),
            "top_unfiltered": top_unfiltered,
            "top_thick": top_thick,
        }

    json_path = OUT_DIR / "top24_anomalies_thick.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\n[done] saved {json_path}")


if __name__ == "__main__":
    main()
