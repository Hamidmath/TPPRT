"""Climatology on DIFFUSED data, by bin-of-week.

For each of 840 bin-of-week slots (120 per weekday, seed=7):
  - find the (up to) 4 calendar dates in the corpus that hit the slot
  - sort chronologically; use the first 3 as "history", the 4th as target
  - prediction = mean of the 3 history E_b vectors
  - measure Top-100 and Overall MRE on the target

Uses the diffused prior (gamma=0.20).
Saves: results/climatology_diffused_840.json
"""
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz

SMOOTH_NPZ = str(config.DATA_DIR / "popularity_results_smoothed_osm_gamma020.npz")
RAW_NPZ    = str(config.DATA_DIR / "popularity_results_osm.npz")
GRAPH_JSON = str(config.GRAPH_FILE)
OUT = config.PROJECT_ROOT / "results" / "climatology_diffused_840.json"

EPS = 1e-6
TOP_K = 100
SEED = 7
BPW = 120  # bin-of-week slots per weekday


def load_prior(npz_path, N, lid_to_idx):
    b = load_popularity_npz(npz_path)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_ids = [str(lid) for lid in b["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_ids], dtype=np.int64)
    valid = proj >= 0
    rows = {}
    for ti, t in enumerate(times):
        row = M.getrow(ti).toarray().ravel().astype(np.float64)
        E = np.zeros(N, dtype=np.float64)
        np.add.at(E, proj[valid], row[valid])
        s = E.sum()
        if s > 0:
            E /= s
        rows[t] = E
    return rows


def main():
    print("loading graph + smoothed prior ...")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    rows = load_prior(SMOOTH_NPZ, N, lid_to_idx)
    print(f"  N={N:,}  bins={len(rows):,}")

    # Group times by bin-of-week (weekday, hour, minute)
    slot_to_times = defaultdict(list)
    for t in rows.keys():
        dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        slot_to_times[(dt.weekday(), dt.hour, dt.minute)].append(t)
    # Sort each slot chronologically
    for s in slot_to_times:
        slot_to_times[s].sort()

    # Keep slots with at least 4 calendar weeks of data.
    # For slots with 5 weeks (Sat/Sun in this 30-day corpus), we use
    # the chronologically-earliest 3 as history and the 4th as target;
    # the 5th observation is ignored.
    full_slots = {s: ts for s, ts in slot_to_times.items() if len(ts) >= 4}
    n4 = sum(1 for ts in full_slots.values() if len(ts) == 4)
    n5 = sum(1 for ts in full_slots.values() if len(ts) >= 5)
    print(f"  slots total: {len(slot_to_times)}  (>=4 weeks: {len(full_slots)}; "
          f"exact-4: {n4}, >=5: {n5})")

    # Sample 840 = 120 per weekday from full_slots
    by_wd = defaultdict(list)
    for s in full_slots.keys():
        by_wd[s[0]].append(s)
    rng = random.Random(SEED)
    sampled = []
    for wd in range(7):
        pool = by_wd[wd]
        rng.shuffle(pool)
        sampled.extend(pool[:BPW])
    print(f"  sampled slots: {len(sampled)}")

    # For each sampled slot: history = mean of weeks 0..2 (diffused),
    # target = diffused week 3 (ignore week 4 if present on Sat/Sun).
    top_list, ov_list = [], []
    for s in sampled:
        ts = full_slots[s][:4]    # take first 4 weeks; ignore week 5 if any
        hist = np.mean([rows[t] for t in ts[:3]], axis=0)
        target = rows[ts[3]]
        rel = np.abs(target - hist) / (target + EPS)
        top_idx = np.argsort(target)[::-1][:TOP_K]
        top_list.append(float(rel[top_idx].mean()))
        ov_list.append(float(rel.mean()))

    top_arr = np.array(top_list)
    ov_arr  = np.array(ov_list)

    def stats(a):
        return {
            "mean":   float(a.mean()),
            "median": float(np.median(a)),
            "std":    float(a.std()),
            "p05":    float(np.quantile(a, 0.05)),
            "p25":    float(np.quantile(a, 0.25)),
            "p75":    float(np.quantile(a, 0.75)),
            "p95":    float(np.quantile(a, 0.95)),
            "min":    float(a.min()),
            "max":    float(a.max()),
        }

    summary = {
        "n_slots_sampled": len(sampled),
        "seed": SEED, "bpw": BPW,
        "history": "mean of 3 chronologically-first diffused weeks",
        "target":  "diffused 4th-week E_b (week 5 dropped on Sat/Sun)",
        "data":   "popularity_results_smoothed_osm_gamma020.npz",
        "epsilon": EPS,
        "top_k":   TOP_K,
        "top100": stats(top_arr),
        "overall": stats(ov_arr),
        "top100_per_slot": top_list,
        "overall_per_slot": ov_list,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print()
    print("                Top-100        Overall")
    for k in ["mean", "median", "std", "p05", "p25", "p75", "p95", "min", "max"]:
        print(f"  {k:<8s}    {summary['top100'][k]:>9.4f}    {summary['overall'][k]:>9.4f}")
    print(f"\nsaved {OUT}")

    # Histograms
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG = OUT.parent / "climatology_diffused_840_hist.pdf"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, arr, label in [(axes[0], top_arr,  "Top-100 MRE per slot"),
                                (axes[1], ov_arr,   "Overall MRE per slot")]:
            ax.hist(arr, bins=40, color="#1f6fb4", alpha=0.85, edgecolor="white")
            ax.axvline(arr.mean(), color="red", ls="--", lw=1.4,
                       label=f"mean = {arr.mean():.4f}")
            ax.axvline(np.median(arr), color="orange", ls=":", lw=1.4,
                       label=f"median = {np.median(arr):.4f}")
            ax.set_xlabel(label); ax.set_ylabel("count")
            ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        fig.suptitle("Climatology on diffused E_b, 3 weeks $\\to$ 4th week, 840 slots")
        fig.tight_layout()
        fig.savefig(FIG, dpi=150)
        plt.close(fig)
        print(f"saved {FIG}")
    except Exception as e:
        print(f"histogram failed: {e}")


if __name__ == "__main__":
    main()
