"""Single-phase PageRank baseline for Table 6.

Two comparisons * three configurations on the 840-bin sampling protocol
(120 per weekday, seed=7) used by Table 10:

  Comparison 1: dataset->dataset (noise floor)
                MRE( target_prior(t+7d), source_prior(t), eps )
  Comparison 2: dataset->model (forecast)
                MRE( target_prior(t+7d), SP_chain(source_prior(t)), eps )

Configurations:
  (a) No diffusion, eps=1e-6 : raw popularity counts as prior
  (b) Diffused,    eps=1e-6 : gamma=0.20 diffused matrix as prior
  (c) Diffused,    eps=0    : gamma=0.20 diffused matrix as prior, no regularizer

Chain: single-phase PageRank with damping d = 1 - p, p = 0.0508 (the
single-geometric calibration on all-K trip lengths,
E[K_total] = 18.68 -> p = 1/(1+E[K]) = 0.0508).

Output: results/eval_table6_singlephase.json
"""
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz

INPUTS  = Path(os.environ.get("TPPR_INPUTS",  str(config.DATA_DIR)))
RESULTS = Path(os.environ.get("TPPR_RESULTS", str(config.PROJECT_ROOT / "results")))
RESULTS.mkdir(parents=True, exist_ok=True)

RAW_NPZ    = str(INPUTS / "popularity_results_osm.npz")
SMOOTH_NPZ = str(INPUTS / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON = str(INPUTS / "city_graph_full.json")
OUT = RESULTS / "eval_table6_singlephase.json"

SEED = 7
BPW  = 120
P_DAMP = 0.0508
D = 1.0 - P_DAMP  # 0.9492
TOL, MAX_ITER = 1e-7, 300


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
    P_cs = csr_matrix((data, (rows, cols)),
                       shape=(N, N), dtype=np.float64)
    return P_cs, links, lid_to_idx


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
    return rows, times


def sample_pairs(times, seed=SEED, bpw=BPW):
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


def sp_iterate(P_cs, prior, d=D, tol=TOL, max_iter=MAX_ITER):
    """v_{k+1} = (1-d) * E_b + d * P_cs @ v_k"""
    v = prior.copy()
    one_minus_d = 1.0 - d
    for _ in range(max_iter):
        v_new = one_minus_d * prior + d * (P_cs @ v)
        s = v_new.sum()
        if s > 0:
            v_new /= s
        if float(np.abs(v_new - v).sum()) < tol:
            return v_new
        v = v_new
    return v


def mre(target, pred, eps):
    """Mean relative error: mean_i |t_i - p_i| / (t_i + eps).
    For eps=0, requires every t_i > 0; we use the diffused matrix
    which enforces this.
    """
    denom = target + eps
    return float(np.mean(np.abs(target - pred) / denom))


def main():
    t0 = time.time()
    print(f"[start] Table 6 single-phase eval, p={P_DAMP} (d={D})")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    P_cs, links, lid_to_idx = build_P_cs(graph)
    N = len(links)
    print(f"  N = {N:,}, P_cs nnz = {P_cs.nnz:,}")

    print("  loading raw prior...")
    raw_rows, _ = load_prior(RAW_NPZ, N, lid_to_idx)
    print(f"    {len(raw_rows):,} bins")

    print("  loading diffused prior...")
    diff_rows, _ = load_prior(SMOOTH_NPZ, N, lid_to_idx)
    print(f"    {len(diff_rows):,} bins")

    pairs = sample_pairs(sorted(diff_rows.keys()))
    print(f"  pairs: {len(pairs)}")

    # 6 accumulators: (config, comparison)
    configs = [
        ("nodiff_eps1e-6", raw_rows,  1e-6),
        ("diff_eps1e-6",   diff_rows, 1e-6),
        ("diff_eps0",      diff_rows, 0.0),
    ]
    noise_floor = {cfg: [] for cfg, _, _ in configs}
    forecast    = {cfg: [] for cfg, _, _ in configs}

    # Cache SP outputs per (config, source) so we run each chain once
    sp_cache = {cfg: {} for cfg, _, _ in configs}

    ts = time.time()
    for k, (t, sib) in enumerate(pairs):
        for cfg, rows, eps in configs:
            src = rows[t]
            tgt = rows[sib]
            # Skip if either is all-zero (shouldn't happen for diffused; can
            # for raw if a bin has no fixes after projection).
            if src.sum() == 0 or tgt.sum() == 0:
                continue
            # Comparison 1: dataset->dataset
            noise_floor[cfg].append(mre(tgt, src, eps))
            # Comparison 2: dataset->model (SP forecast)
            if t not in sp_cache[cfg]:
                sp_cache[cfg][t] = sp_iterate(P_cs, src)
            v = sp_cache[cfg][t]
            forecast[cfg].append(mre(tgt, v, eps))
        if (k + 1) % 100 == 0:
            print(f"    pair {k+1}/{len(pairs)}  ({time.time()-ts:.0f}s)",
                  flush=True)

    summary = {
        "spec": dict(
            p=P_DAMP, d=D,
            seed=SEED, bpw=BPW, n_pairs=len(pairs),
            raw_npz="popularity_results_osm.npz",
            diff_npz="popularity_results_smoothed_osm_gamma020.npz",
            graph="city_graph_full.json",
        ),
        "results": {},
    }
    print()
    print(f"  {'config':18s}  {'comparison':18s}  {'count':>5s}  {'mean':>8s}")
    for cfg, _, _ in configs:
        for label, arr in [("noise_floor", noise_floor[cfg]),
                            ("forecast",    forecast[cfg])]:
            mean = float(np.mean(arr)) if arr else float("nan")
            print(f"  {cfg:18s}  {label:18s}  {len(arr):5d}  {mean:8.4f}")
            summary["results"][f"{cfg}/{label}"] = {
                "mean": mean,
                "median": float(np.median(arr)) if arr else float("nan"),
                "n_pairs_used": len(arr),
            }

    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {time.time()-t0:.0f}s saved {OUT}")


if __name__ == "__main__":
    main()
