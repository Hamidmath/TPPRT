"""Climatology baselines: variants of the "predict next week's E_b
from the historical average" idea.

Four variants, all evaluated on the standard 840-bin sample (seed=7):

  c1_mean3      -- mean of the three OTHER weeks at the same (wd,h,m)
                   (closest to what the professor suggested)
  c2_meanAll    -- mean of ALL other weeks at same (wd,h,m), excluding
                   the target week itself
  c3_medianAll  -- median across all weeks at same (wd,h,m)
  c4_prev_only  -- just use last week's bin (t - 7d), the natural noise
                   floor

Each compared against the Top-100 / Overall MRE on the SAME 840 bins
the ladder uses.

Saves: results/climatology_variants.json
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config
from core.io import load_popularity_npz

INPUTS = Path(os.environ.get("TPPR_INPUTS", config.DATA_DIR))
RESULTS = Path(os.environ.get("TPPR_RESULTS", config.PROJECT_ROOT / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)

RAW_NPZ    = str(INPUTS / "popularity_results_osm.npz")
SMOOTH_NPZ = str(INPUTS / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON = str(INPUTS / "city_graph_full.json")

EPS = 1e-6
TOP_K = 100
SEED = 7
BPW = 120


def load_prior(npz_path, links, lid_to_idx, N):
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
        if s > 0: E /= s
        rows[t] = E
    return rows


def sample_pairs(times, seed, bpw):
    rng = random.Random(seed)
    by_wd = {i: [] for i in range(7)}
    tset = set(times)
    for t in times:
        dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        sib = (dt + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        if sib in tset:
            by_wd[dt.weekday()].append(t)
    pairs = []
    for wd in range(7):
        pool = by_wd[wd]
        rng.shuffle(pool)
        for t in pool[:bpw]:
            dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
            sib = (dt + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
            pairs.append((t, sib))
    return pairs


def main():
    t0 = time.time()
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    print(f"  N = {N:,}")

    print("  loading raw prior ...")
    raw = load_prior(RAW_NPZ, links, lid_to_idx, N)
    print(f"  {len(raw)} bins")

    pairs = sample_pairs(sorted(raw.keys()), SEED, BPW)
    pairs = [(t, s) for t, s in pairs if t in raw and s in raw]
    print(f"  pairs: {len(pairs)}")

    # Index by (weekday, hour, minute, week_id)
    slot_to_times = defaultdict(list)
    for t in raw.keys():
        dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        slot_to_times[(dt.weekday(), dt.hour, dt.minute)].append(t)

    Es_target = [raw[s] for _, s in pairs]
    top_idx   = [np.argsort(E)[::-1][:TOP_K] for E in Es_target]

    def eval_pred(pred_fn, tag):
        ts = time.time()
        top, ov = [], []
        for i in range(len(pairs)):
            v = pred_fn(i)
            if v is None:
                continue
            rel = np.abs(Es_target[i] - v) / (Es_target[i] + EPS)
            top.append(float(rel[top_idx[i]].mean()))
            ov.append(float(rel.mean()))
        return {
            "tag": tag,
            "top100_mre_mean": float(np.mean(top)),
            "top100_mre_median": float(np.median(top)),
            "overall_mre_mean": float(np.mean(ov)),
            "overall_mre_median": float(np.median(ov)),
            "n_bins": len(top),
            "elapsed_s": time.time() - ts,
        }

    summary = {}

    # c1: mean of OTHER weeks at same (wd,h,m), excluding the test bin
    def c1(i):
        _, sib = pairs[i]
        dt = datetime.strptime(sib, "%Y-%m-%d %H:%M:%S")
        slot = (dt.weekday(), dt.hour, dt.minute)
        cands = [t for t in slot_to_times[slot] if t != sib]
        Es = [raw[t] for t in cands if t in raw]
        if not Es:
            return None
        return np.mean(Es, axis=0)
    summary["c1_mean_other"] = eval_pred(c1, "c1_mean_other")
    print(f"  c1_mean_other  top100={summary['c1_mean_other']['top100_mre_mean']:.4f}",
          flush=True)

    # c2: mean of all weeks at same (wd,h,m), including sib if it exists
    def c2(i):
        _, sib = pairs[i]
        dt = datetime.strptime(sib, "%Y-%m-%d %H:%M:%S")
        slot = (dt.weekday(), dt.hour, dt.minute)
        Es = [raw[t] for t in slot_to_times[slot] if t in raw]
        if not Es:
            return None
        return np.mean(Es, axis=0)
    summary["c2_mean_all"] = eval_pred(c2, "c2_mean_all")
    print(f"  c2_mean_all    top100={summary['c2_mean_all']['top100_mre_mean']:.4f}",
          flush=True)

    # c3: median across all weeks at same (wd,h,m)
    def c3(i):
        _, sib = pairs[i]
        dt = datetime.strptime(sib, "%Y-%m-%d %H:%M:%S")
        slot = (dt.weekday(), dt.hour, dt.minute)
        Es = [raw[t] for t in slot_to_times[slot] if t in raw]
        if not Es:
            return None
        return np.median(np.stack(Es, axis=0), axis=0)
    summary["c3_median_all"] = eval_pred(c3, "c3_median_all")
    print(f"  c3_median_all  top100={summary['c3_median_all']['top100_mre_mean']:.4f}",
          flush=True)

    # c4: just use last week's value
    def c4(i):
        t, _ = pairs[i]
        return raw[t] if t in raw else None
    summary["c4_prev_only"] = eval_pred(c4, "c4_prev_only")
    print(f"  c4_prev_only   top100={summary['c4_prev_only']['top100_mre_mean']:.4f}",
          flush=True)

    with open(RESULTS / "climatology_variants.json", "w") as f:
        json.dump({"seed": SEED, "bpw": BPW, "summary": summary,
                   "elapsed_total": time.time() - t0}, f, indent=2)
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
