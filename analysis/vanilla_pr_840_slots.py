"""Vanilla single-phase PR on the SAME 840 slots used by
climatology_diffused_840.py.

For each slot (120 random bin-of-week slots per weekday, seed=7,
ignoring week-5 occurrence on Sat/Sun):
  - "current week" = diffused E_b at the chronologically 3rd week
  - "next week"    = diffused E_b at the chronologically 4th week
  - Predict via single-phase PR with uniform out-edges:
        v_{n+1} = alpha * prior + (1 - alpha) * P_cs @ v_n
    Two priors:
      A. uniform 1/N
      B. popularity = E_b(week 3 of this slot)
  - MRE per slot: Top-100 over target's busiest 100 links AND Overall
    over all N links.

Sweeps alpha in {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}.

Saves: results/vanilla_pr_840_slots.json
       results/vanilla_pr_840_slots_hist.pdf
"""
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz

SMOOTH_NPZ = str(config.DATA_DIR / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON = str(config.GRAPH_FILE)
OUT = config.PROJECT_ROOT / "results" / "vanilla_pr_840_slots.json"

EPS = 1e-6
TOP_K = 100
SEED = 7
BPW = 120
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
TOL, MAX_ITER = 1e-6, 300


def build_P_cs(graph):
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    adj = graph.get("adjacency", {})
    rows, cols, data = [], [], []
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ:
            continue
        w = 1.0 / len(succ)
        for j in succ:
            rows.append(j); cols.append(i); data.append(w)
    return csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64), links, lid_to_idx


def sp_iter(P_cs, prior, alpha, tol=TOL, max_iter=MAX_ITER):
    v = prior.copy()
    for _ in range(max_iter):
        v_new = alpha * prior + (1.0 - alpha) * (P_cs @ v)
        s = float(v_new.sum())
        if s > 0:
            v_new /= s
        diff = float(np.abs(v_new - v).sum())
        v = v_new
        if diff < tol:
            return v
    return v


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


def stats(a):
    a = np.asarray(a)
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


def main():
    print("[start] vanilla PR on 840 slots, diffused, week3 -> week4")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    P_cs, links, lid_to_idx = build_P_cs(graph)
    N = len(links)
    print(f"  N={N:,}")

    rows = load_prior(SMOOTH_NPZ, N, lid_to_idx)
    print(f"  bins={len(rows):,}")

    # Same slot sampling as climatology_diffused_840.py
    slot_to_times = defaultdict(list)
    for t in rows.keys():
        dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        slot_to_times[(dt.weekday(), dt.hour, dt.minute)].append(t)
    for s in slot_to_times:
        slot_to_times[s].sort()
    full_slots = {s: ts for s, ts in slot_to_times.items() if len(ts) >= 4}
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

    # Precompute per-slot inputs (week 3) and targets (week 4)
    inputs  = [rows[full_slots[s][:4][2]] for s in sampled]   # week 3
    targets = [rows[full_slots[s][:4][3]] for s in sampled]   # week 4
    top_idxs = [np.argsort(t)[::-1][:TOP_K] for t in targets]

    results = {}

    # ------------------------------ uniform prior
    # Output is independent of slot: one fixed PR vector per alpha.
    uniform_prior = np.full(N, 1.0 / N, dtype=np.float64)
    for alpha in ALPHAS:
        v_fixed = sp_iter(P_cs, uniform_prior, alpha)
        top_list, ov_list = [], []
        for i in range(len(sampled)):
            rel = np.abs(targets[i] - v_fixed) / (targets[i] + EPS)
            top_list.append(float(rel[top_idxs[i]].mean()))
            ov_list.append(float(rel.mean()))
        tag = f"uniform_a{alpha:.2f}"
        results[tag] = {
            "alpha": alpha, "prior": "uniform",
            "top100": stats(top_list),
            "overall": stats(ov_list),
            "top100_per_slot": top_list,
            "overall_per_slot": ov_list,
        }
        print(f"  [{tag}]  top100 mean={results[tag]['top100']['mean']:.4f}  "
              f"overall mean={results[tag]['overall']['mean']:.4f}")

    # ------------------------------ popularity prior (= week-3 E_b)
    for alpha in ALPHAS:
        top_list, ov_list = [], []
        for i in range(len(sampled)):
            v = sp_iter(P_cs, inputs[i], alpha)
            rel = np.abs(targets[i] - v) / (targets[i] + EPS)
            top_list.append(float(rel[top_idxs[i]].mean()))
            ov_list.append(float(rel.mean()))
        tag = f"popularity_a{alpha:.2f}"
        results[tag] = {
            "alpha": alpha, "prior": "popularity",
            "top100": stats(top_list),
            "overall": stats(ov_list),
            "top100_per_slot": top_list,
            "overall_per_slot": ov_list,
        }
        print(f"  [{tag}]  top100 mean={results[tag]['top100']['mean']:.4f}  "
              f"overall mean={results[tag]['overall']['mean']:.4f}")

    summary = {
        "n_slots": len(sampled), "seed": SEED, "bpw": BPW,
        "input":  "diffused E_b at week 3 of the slot",
        "target": "diffused E_b at week 4 of the slot",
        "alphas": ALPHAS,
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {OUT}")

    # Headline table
    print("\n  config               Top-100      Overall")
    print("  -------              -------     -------")
    for tag, r in results.items():
        print(f"  {tag:<20s} {r['top100']['mean']:>7.4f}     {r['overall']['mean']:>7.4f}")

    # Histogram comparison: pick alpha=0.15 (Google) for both priors and alpha=0.30 popularity (best vanilla)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG = OUT.parent / "vanilla_pr_840_slots_hist.pdf"
        keys = ["uniform_a0.15", "uniform_a0.30",
                "popularity_a0.15", "popularity_a0.30"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for ax, k in zip(axes.flat, keys):
            r = results[k]
            top = np.array(r["top100_per_slot"])
            ov  = np.array(r["overall_per_slot"])
            ax.hist(top, bins=30, color="#1f6fb4", alpha=0.7, edgecolor="white",
                    label=f"Top-100 mean={top.mean():.3f}")
            ax.hist(ov,  bins=30, color="#c0392b", alpha=0.55, edgecolor="white",
                    label=f"Overall mean={ov.mean():.3f}")
            ax.set_title(k)
            ax.set_xlabel("MRE per slot"); ax.set_ylabel("count")
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.suptitle("Vanilla single-phase PR, 840 slots, diffused week3 $\\to$ week4")
        fig.tight_layout()
        fig.savefig(FIG, dpi=150)
        plt.close(fig)
        print(f"saved {FIG}")
    except Exception as e:
        print(f"histogram failed: {e}")


if __name__ == "__main__":
    main()
