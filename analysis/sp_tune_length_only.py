"""Single-phase tuning with LENGTH ONLY: tune (alpha, alpha_len) on
the SAME 840 (t, t+7d) pairs as Table 10. alpha_s = alpha_l = 0.

Same protocol as sp_tune_three_ov.py's "both" variant, but the edge
weight now includes length:

    weight(link) = (speed/mean)^alpha_s
                 * (lanes/mean)^alpha_l
                 * (length/mean)^alpha_len

Bounds: alpha in [0.01, 0.15]; alpha_s, alpha_l, alpha_len unbounded.
L-BFGS-B with 5 starts (run in parallel across L-BFGS-B starts, one
worker per CPU). Same eps=2e-3, ftol=1e-6, gtol=1e-4, maxiter=20.

Output: results/sp_tune_with_len.json
"""
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz

INPUTS  = Path(os.environ.get("TPPR_INPUTS",  str(config.DATA_DIR)))
RESULTS = Path(os.environ.get("TPPR_RESULTS", str(config.PROJECT_ROOT / "results")))
RESULTS.mkdir(parents=True, exist_ok=True)

SMOOTH_NPZ = str(INPUTS / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON = str(INPUTS / "city_graph_full.json")
OUT = RESULTS / "sp_tune_length_only.json"

EPS = 1e-6
TOP_K = 100
SEED = 7
BPW = 120
TOL, MAX_ITER = 1e-5, 100

ALPHA_LO, ALPHA_HI = 0.01, 0.15

STARTS = [
    # alpha, alpha_len   (alpha_s = alpha_l = 0)
    [0.10,  0.0],
    [0.15,  1.0],
    [0.05, -1.0],
    [0.12,  2.0],
    [0.08, -2.0],
]


def build_P_weighted(graph, links, lid_to_idx, weights):
    adj = graph.get("adjacency", {})
    rows, cols, data = [], [], []
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ:
            continue
        out_w = np.array([weights[j] for j in succ], dtype=np.float64)
        total = out_w.sum()
        if total <= 0:
            continue
        norm = out_w / total
        for k, j in enumerate(succ):
            rows.append(j); cols.append(i); data.append(float(norm[k]))
    return csr_matrix((data, (rows, cols)),
                       shape=(len(links), len(links)), dtype=np.float64)


def get_speed_lane_length(graph, links):
    s, l, le = [], [], []
    for lid in links:
        ed = graph["links"].get(lid, {})
        s.append(ed.get("speed", 11.17))
        l.append(ed.get("lanes", 1.0))
        le.append(max(ed.get("length", 100.0), 1.0))
    s = np.array(s, dtype=np.float64)
    l = np.array(l, dtype=np.float64)
    le = np.array(le, dtype=np.float64)
    if s.mean() > 0: s /= s.mean()
    if l.mean() > 0: l /= l.mean()
    if le.mean() > 0: le /= le.mean()
    return s, l, le


def sp_iter(P_cs, prior, alpha, tol=TOL, max_iter=MAX_ITER):
    v = prior.copy()
    one_a = 1.0 - alpha
    for _ in range(max_iter):
        v_new = alpha * prior + one_a * (P_cs @ v)
        s = v_new.sum()
        if s > 0: v_new /= s
        if float(np.abs(v_new - v).sum()) < tol:
            return v_new
        v = v_new
    return v


def load_diffused(npz_path, N, lid_to_idx, needed_times=None):
    b = load_popularity_npz(npz_path)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_ids = [str(lid) for lid in b["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_ids], dtype=np.int64)
    valid = proj >= 0
    needed = set(needed_times) if needed_times is not None else None
    rows = {}
    for ti, t in enumerate(times):
        if needed is not None and t not in needed: continue
        row = M.getrow(ti).toarray().ravel().astype(np.float64)
        E = np.zeros(N, dtype=np.float64)
        np.add.at(E, proj[valid], row[valid])
        s = E.sum()
        if s > 0: E /= s
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


# Worker globals (initialized per process)
WORKER = {}


def _worker_init():
    """Load graph + diffused data once per worker process."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes, lengths = get_speed_lane_length(graph, links)
    # Two-pass: first peek at the timestamp list to compute the pair
    # schedule, then load only the timestamps that the 840 pairs need.
    b = load_popularity_npz(SMOOTH_NPZ)
    all_times = [str(t) for t in b["times"]]
    del b
    pairs = sample_pairs(sorted(all_times))
    needed = set()
    for t, sib in pairs:
        needed.add(t); needed.add(sib)
    rows, _ = load_diffused(SMOOTH_NPZ, N, lid_to_idx, needed_times=needed)
    inputs  = [rows[t]   for t, _   in pairs]
    targets = [rows[sib] for _, sib in pairs]
    top_idx_list = [np.argsort(t)[::-1][:TOP_K] for t in targets]
    WORKER.update(dict(
        graph=graph, links=links, lid_to_idx=lid_to_idx, N=N,
        speeds=speeds, lanes=lanes, lengths=lengths,
        pairs=pairs, inputs=inputs, targets=targets,
        top_idx_list=top_idx_list, kernel_cache={},
    ))
    print(f"  worker pid {os.getpid()} ready  (N={N:,}, pairs={len(pairs)})",
          flush=True)


def _worker_get_kernel(a_s, a_l, a_len):
    key = (round(a_s, 4), round(a_l, 4), round(a_len, 4))
    cache = WORKER["kernel_cache"]
    if key in cache:
        return cache[key]
    speeds, lanes, lengths = WORKER["speeds"], WORKER["lanes"], WORKER["lengths"]
    N = WORKER["N"]
    w = np.ones(N, dtype=np.float64)
    if a_s   != 0: w = w * np.power(speeds,  a_s)
    if a_l   != 0: w = w * np.power(lanes,   a_l)
    if a_len != 0: w = w * np.power(lengths, a_len)
    P = build_P_weighted(WORKER["graph"], WORKER["links"],
                          WORKER["lid_to_idx"], w)
    cache[key] = P
    if len(cache) > 64:
        for k in list(cache.keys())[:32]:
            del cache[k]
    return P


def _worker_loss(x):
    alpha, a_len = float(x[0]), float(x[1])
    a_s, a_l = 0.0, 0.0
    P = _worker_get_kernel(a_s, a_l, a_len)
    pairs = WORKER["pairs"]; inputs = WORKER["inputs"]
    targets = WORKER["targets"]; top_idx = WORKER["top_idx_list"]
    top_list, ov_list = [], []
    for i in range(len(pairs)):
        v = sp_iter(P, inputs[i], alpha)
        rel = np.abs(targets[i] - v) / (targets[i] + EPS)
        top_list.append(float(rel[top_idx[i]].mean()))
        ov_list.append(float(rel.mean()))
    return float(np.mean(ov_list)), top_list, ov_list


def _worker_loss_scalar(x):
    return _worker_loss(x)[0]


def worker_run(payload):
    """Run one L-BFGS-B start; return result dict."""
    start_idx, x0 = payload
    ts = time.time()
    res = minimize(_worker_loss_scalar, np.array(x0, dtype=np.float64),
                    method="L-BFGS-B",
                    bounds=[(ALPHA_LO, ALPHA_HI), (None, None)],
                    options=dict(eps=2e-3, ftol=1e-6, gtol=1e-4, maxiter=20))
    f_opt, top_list, ov_list = _worker_loss(res.x)
    elapsed = time.time() - ts
    print(f"  start {start_idx} done in {elapsed:.0f}s  nfev={res.nfev}  "
          f"f={res.fun:.4f}  x={[round(float(v),3) for v in res.x]}",
          flush=True)
    return dict(start=start_idx, x0=list(map(float, x0)),
                 x_opt=list(map(float, res.x)), f_opt=float(res.fun),
                 nfev=int(res.nfev), elapsed=elapsed,
                 top_list=top_list, ov_list=ov_list)


def main():
    t0 = time.time()
    max_workers = int(os.environ.get("TPPR_PARALLEL", "4"))
    print(f"[start] SP tuning with alpha_len  (parallel starts, max_workers={max_workers})")
    print(f"  starts: {len(STARTS)}")
    for j, x0 in enumerate(STARTS):
        print(f"    [start {j}] x0 = {x0}")

    runs = []
    with ProcessPoolExecutor(max_workers=max_workers,
                              initializer=_worker_init) as ex:
        futs = {ex.submit(worker_run, (j, x0)): j
                 for j, x0 in enumerate(STARTS)}
        for fut in as_completed(futs):
            runs.append(fut.result())
    runs.sort(key=lambda r: r["start"])
    best = min(runs, key=lambda r: r["f_opt"])

    a_opt, alen_opt = best["x_opt"]
    summary = dict(
        model="single-phase + length only",
        scope="840 (t, t+7d) pairs, Table 10 protocol; alpha_s = alpha_l = 0",
        bounds=dict(alpha=[ALPHA_LO, ALPHA_HI], alpha_len=None),
        starts=[{k: v for k, v in r.items() if k not in ("top_list","ov_list")}
                 for r in runs],
        best=dict(start=best["start"], x_opt=best["x_opt"],
                   f_opt=best["f_opt"], nfev=best["nfev"]),
        best_params=dict(alpha=a_opt, alpha_s=0.0,
                          alpha_l=0.0, alpha_len=alen_opt),
        top100_mean=float(np.mean(best["top_list"])),
        top100_median=float(np.median(best["top_list"])),
        overall_mean=float(np.mean(best["ov_list"])),
        overall_median=float(np.median(best["ov_list"])),
    )
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {time.time()-t0:.0f}s   saved {OUT}")
    print(f"  best: alpha={a_opt:.4f}  alpha_len={alen_opt:+.3f}")
    print(f"        Overall MRE = {summary['overall_mean']:.4f}  "
          f"Top-100 = {summary['top100_mean']:.4f}")


if __name__ == "__main__":
    main()
