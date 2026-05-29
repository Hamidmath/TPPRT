"""Professor's single-phase ladder, alpha FIXED at the geometric-fit
value alpha = 0.0508 throughout. Five rows on the 840-pair eval:

  1. uniform prior        E_b = 1/N,             no road-type tuning
  2. diffuse prior        E_b = diffused E_b(t), no road-type tuning
  3. + lanes only         E_b = diffused, learn alpha_l
  4. + speed only         E_b = diffused, learn alpha_s
  5. + both               E_b = diffused, learn alpha_s, alpha_l

All five share the same 840 (t, t+7d) pairs as Table 10. Tuning rows
use L-BFGS-B with 5 parallel starts; alpha_s, alpha_l unbounded.

Output: results/sp_prof_ladder.json
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
OUT = RESULTS / "sp_prof_ladder.json"

EPS = 1e-6
TOP_K = 100
SEED = 7
BPW = 120
TOL, MAX_ITER = 1e-5, 100
ALPHA_FIXED = 0.0508

LANE_STARTS   = [[0.0], [+0.5], [-0.5], [+1.0], [-1.0]]
SPEED_STARTS  = [[0.0], [+1.0], [-1.0], [+2.65], [-2.0]]
LEN_STARTS    = [[0.0], [+0.5], [-0.5], [+1.0], [-1.0]]
BOTH_STARTS   = [[0.0, 0.0], [+1.0, +0.5], [-1.0, -1.0],
                  [+2.65, +0.17], [+3.0, +1.0]]
THREE_STARTS  = [[0.0, 0.0, 0.0], [+1.0, +0.5, +0.5],
                  [-1.0, -1.0, +1.0], [+2.65, +0.17, 0.0],
                  [+3.0, +1.0, -0.5]]


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
    s  = np.array(s,  dtype=np.float64)
    l  = np.array(l,  dtype=np.float64)
    le = np.array(le, dtype=np.float64)
    if s.mean()  > 0: s  /= s.mean()
    if l.mean()  > 0: l  /= l.mean()
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


WORKER = {}


def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes, lengths = get_speed_lane_length(graph, links)
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
    print(f"  worker pid {os.getpid()} ready  N={N:,}, pairs={len(pairs)}",
          flush=True)


def _kernel(mode, x):
    """Return the row-stochastic transition for a given mode + x vector."""
    a_s, a_l, a_len = 0.0, 0.0, 0.0
    if mode == "lanes_only":
        a_l = float(x[0])
    elif mode == "speed_only":
        a_s = float(x[0])
    elif mode == "length_only":
        a_len = float(x[0])
    elif mode == "both":
        a_s, a_l = float(x[0]), float(x[1])
    elif mode == "three":
        a_s, a_l, a_len = float(x[0]), float(x[1]), float(x[2])
    key = ("sp", round(a_s, 4), round(a_l, 4), round(a_len, 4))
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


def _eval_840(P_cs, inputs):
    """Return per-pair (top100, ov) lists."""
    top_list, ov_list = [], []
    targets  = WORKER["targets"]
    top_idx  = WORKER["top_idx_list"]
    for i in range(len(inputs)):
        v = sp_iter(P_cs, inputs[i], ALPHA_FIXED)
        rel = np.abs(targets[i] - v) / (targets[i] + EPS)
        top_list.append(float(rel[top_idx[i]].mean()))
        ov_list.append(float(rel.mean()))
    return top_list, ov_list


def _loss(mode, x):
    P = _kernel(mode, x)
    inputs = WORKER["inputs"]
    top_list, ov_list = _eval_840(P, inputs)
    return float(np.mean(ov_list)), top_list, ov_list


def worker_run(payload):
    mode, start_idx, x0 = payload
    ts = time.time()
    bounds = [(None, None)] * len(x0)
    res = minimize(lambda x: _loss(mode, x)[0], np.array(x0, dtype=np.float64),
                    method="L-BFGS-B", bounds=bounds,
                    options=dict(eps=2e-3, ftol=1e-6, gtol=1e-4, maxiter=20))
    f_opt, top_list, ov_list = _loss(mode, res.x)
    elapsed = time.time() - ts
    print(f"  [{mode}] start {start_idx} done {elapsed:.0f}s  "
          f"nfev={res.nfev}  f={res.fun:.4f}  x={[round(float(v),3) for v in res.x]}",
          flush=True)
    return dict(mode=mode, start=start_idx, x0=list(map(float, x0)),
                 x_opt=list(map(float, res.x)), f_opt=float(res.fun),
                 nfev=int(res.nfev), elapsed=elapsed,
                 top_list=top_list, ov_list=ov_list)


def baseline_eval(mode):
    """Sequential eval for uniform / diffuse priors (no tuning)."""
    P_uniform = _kernel("none", [0.0, 0.0])  # uniform out-edges
    if mode == "uniform_prior":
        # Single chain run with E_b = 1/N applied to all 840 pairs.
        N = WORKER["N"]
        u = np.full(N, 1.0 / N, dtype=np.float64)
        v_star = sp_iter(P_uniform, u, ALPHA_FIXED)
        per_pair = []
        for t in WORKER["targets"]:
            per_pair.append(float(np.mean(np.abs(t - v_star) / (t + EPS))))
        return dict(mode=mode, overall_mean=float(np.mean(per_pair)),
                     overall_median=float(np.median(per_pair)))
    else:  # diffuse_prior
        top_list, ov_list = _eval_840(P_uniform, WORKER["inputs"])
        return dict(mode=mode, overall_mean=float(np.mean(ov_list)),
                     overall_median=float(np.median(ov_list)),
                     top100_mean=float(np.mean(top_list)))


def main():
    t0 = time.time()
    max_workers = int(os.environ.get("TPPR_PARALLEL", "5"))
    print(f"[start] SP professor ladder, alpha={ALPHA_FIXED} fixed  "
          f"(parallel max_workers={max_workers})")

    out = dict(model="SP, alpha fixed at geometric-fit",
                alpha_fixed=ALPHA_FIXED, rows={})

    # ---- baselines (sequential, fast) ----
    print("\n== row 1: uniform prior ==")
    _worker_init()
    r = baseline_eval("uniform_prior")
    print(f"  Ov.MRE mean = {r['overall_mean']:.4f}, median = {r['overall_median']:.4f}")
    out["rows"]["uniform_prior"] = r

    print("\n== row 2: diffuse prior ==")
    r = baseline_eval("diffuse_prior")
    print(f"  Ov.MRE mean = {r['overall_mean']:.4f}, median = {r['overall_median']:.4f}")
    out["rows"]["diffuse_prior"] = r

    # ---- tuning modes (parallel starts) ----
    for mode, starts in [("lanes_only",   LANE_STARTS),
                          ("speed_only",  SPEED_STARTS),
                          ("length_only", LEN_STARTS),
                          ("both",        BOTH_STARTS),
                          ("three",       THREE_STARTS)]:
        print(f"\n== mode: + {mode} ==")
        runs = []
        with ProcessPoolExecutor(max_workers=max_workers,
                                  initializer=_worker_init) as ex:
            futs = {ex.submit(worker_run, (mode, j, x0)): j
                     for j, x0 in enumerate(starts)}
            for fut in as_completed(futs):
                runs.append(fut.result())
        runs.sort(key=lambda r: r["start"])
        best = min(runs, key=lambda r: r["f_opt"])
        a_s, a_l, a_len = 0.0, 0.0, 0.0
        if mode == "lanes_only":
            a_l = best["x_opt"][0]
        elif mode == "speed_only":
            a_s = best["x_opt"][0]
        elif mode == "length_only":
            a_len = best["x_opt"][0]
        elif mode == "both":
            a_s, a_l = best["x_opt"]
        else:  # three
            a_s, a_l, a_len = best["x_opt"]
        print(f"  best: a_s={a_s:+.3f}  a_l={a_l:+.3f}  a_len={a_len:+.3f}  "
              f"Ov.MRE={best['f_opt']:.4f}")
        out["rows"][mode] = dict(
            mode=mode, best_alpha_s=a_s, best_alpha_l=a_l, best_alpha_len=a_len,
            best_f=best["f_opt"],
            overall_mean=float(np.mean(best["ov_list"])),
            overall_median=float(np.median(best["ov_list"])),
            top100_mean=float(np.mean(best["top_list"])),
            starts=[{k: v for k, v in r.items()
                      if k not in ("top_list", "ov_list")} for r in runs],
        )

    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n[done] {time.time() - t0:.0f}s  saved {OUT}")
    for k, r in out["rows"].items():
        print(f"  {k:>14s}:  Ov.MRE mean = {r['overall_mean']:.4f}")


if __name__ == "__main__":
    main()
