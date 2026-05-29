"""Two-phase (new-form, mass-conserving) event-direction fit on the
WHOLE 99,716-link network.

Setup (locked):
  pairs   : (Sep 11 16:MM, Sep 18 16:MM) for MM in 00..55 step 5  (12)
  input   : diffused popularity at Sep 11 (week 2)
  target  : diffused popularity at Sep 18 (week 3)
  chain   : TP-new, beta = 0.102, rho = 0.101, BOTH FIXED
  loss    : Overall MRE on the entire network (no region restriction)
  search  : L-BFGS-B, 5 parallel starts; alpha_s, alpha_l unbounded

Mode is chosen via env var TPPR_MODE:
  speed_only   tune alpha_s
  lanes_only   tune alpha_l
  both         tune alpha_s, alpha_l

Output: results/tp_event_full_<MODE>.json
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

MODE = os.environ.get("TPPR_MODE", "both")
OUT = RESULTS / f"tp_event_full_{MODE}.json"

EPS = 1e-6
TOL, MAX_ITER = 1e-6, 200
BETA_FIXED = 0.102
RHO_FIXED  = 0.101
PAIRS = [(f"2018-09-11 16:{m:02d}:00", f"2018-09-18 16:{m:02d}:00")
          for m in range(0, 60, 5)]

LANE_STARTS  = [[0.0], [+0.5], [-0.5], [+1.0], [-1.0]]
SPEED_STARTS = [[0.0], [+1.0], [-1.0], [+2.65], [-2.0]]
BOTH_STARTS  = [[0.0, 0.0], [+1.0, +0.5], [-1.0, -1.0],
                 [+2.65, +0.17], [+3.0, +1.0]]
MODE_STARTS = {"speed_only": SPEED_STARTS, "lanes_only": LANE_STARTS,
                "both": BOTH_STARTS}


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


def get_speed_lane(graph, links):
    s, l = [], []
    for lid in links:
        ed = graph["links"].get(lid, {})
        s.append(ed.get("speed", 11.17))
        l.append(ed.get("lanes", 1.0))
    s = np.array(s, dtype=np.float64); l = np.array(l, dtype=np.float64)
    if s.mean() > 0: s /= s.mean()
    if l.mean() > 0: l /= l.mean()
    return s, l


def tp_new_iter(P_cs, u, beta, rho, tol=TOL, max_iter=MAX_ITER):
    N = u.shape[0]
    v_up = u.copy()
    v_down = np.zeros(N)
    for _ in range(max_iter):
        s_down = float(v_down.sum())
        v_up_new   = (1.0 - beta) * (P_cs @ v_up) + rho * u * s_down
        v_down_new = beta * v_up + (1.0 - rho) * (P_cs @ v_down)
        s = float(v_up_new.sum() + v_down_new.sum())
        if s > 0:
            v_up_new /= s; v_down_new /= s
        diff = float(np.abs(v_up_new - v_up).sum() + np.abs(v_down_new - v_down).sum())
        v_up, v_down = v_up_new, v_down_new
        if diff < tol:
            break
    return v_up + v_down


def load_diffused(npz_path, N, lid_to_idx, needed_times):
    b = load_popularity_npz(npz_path)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_ids = [str(lid) for lid in b["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_ids], dtype=np.int64)
    valid = proj >= 0
    needed = set(needed_times)
    rows = {}
    for ti, t in enumerate(times):
        if t not in needed: continue
        row = M.getrow(ti).toarray().ravel().astype(np.float64)
        E = np.zeros(N, dtype=np.float64)
        np.add.at(E, proj[valid], row[valid])
        s = E.sum()
        if s > 0: E /= s
        rows[t] = E
    return rows


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
    speeds, lanes = get_speed_lane(graph, links)
    needed = set()
    for t, sib in PAIRS:
        needed.add(t); needed.add(sib)
    rows = load_diffused(SMOOTH_NPZ, N, lid_to_idx, needed)
    inputs  = [rows[t]   for t, _   in PAIRS]
    targets = [rows[sib] for _, sib in PAIRS]
    WORKER.update(dict(
        graph=graph, links=links, lid_to_idx=lid_to_idx, N=N,
        speeds=speeds, lanes=lanes,
        inputs=inputs, targets=targets, kernel_cache={},
    ))
    print(f"  worker pid {os.getpid()} ready  N={N:,}, pairs={len(PAIRS)}",
          flush=True)


def _kernel(a_s, a_l):
    key = (round(a_s, 4), round(a_l, 4))
    cache = WORKER["kernel_cache"]
    if key in cache:
        return cache[key]
    speeds, lanes = WORKER["speeds"], WORKER["lanes"]
    N = WORKER["N"]
    w = np.ones(N, dtype=np.float64)
    if a_s != 0: w = w * np.power(speeds, a_s)
    if a_l != 0: w = w * np.power(lanes,  a_l)
    P = build_P_weighted(WORKER["graph"], WORKER["links"],
                          WORKER["lid_to_idx"], w)
    cache[key] = P
    if len(cache) > 32:
        for k in list(cache.keys())[:16]:
            del cache[k]
    return P


def _unpack(mode, x):
    if mode == "speed_only":  return float(x[0]), 0.0
    if mode == "lanes_only":  return 0.0, float(x[0])
    if mode == "both":        return float(x[0]), float(x[1])
    raise ValueError(mode)


def _loss(mode, x):
    a_s, a_l = _unpack(mode, x)
    P = _kernel(a_s, a_l)
    inputs  = WORKER["inputs"]
    targets = WORKER["targets"]
    ov_list = []
    for i in range(len(inputs)):
        v = tp_new_iter(P, inputs[i], BETA_FIXED, RHO_FIXED)
        rel = np.abs(targets[i] - v) / (targets[i] + EPS)
        ov_list.append(float(rel.mean()))
    return float(np.mean(ov_list)), ov_list


def worker_run(payload):
    mode, start_idx, x0 = payload
    ts = time.time()
    bounds = [(None, None)] * len(x0)
    res = minimize(lambda x: _loss(mode, x)[0], np.array(x0, dtype=np.float64),
                    method="L-BFGS-B", bounds=bounds,
                    options=dict(eps=2e-3, ftol=1e-6, gtol=1e-4, maxiter=20))
    f_opt, ov_list = _loss(mode, res.x)
    elapsed = time.time() - ts
    print(f"  [{mode}] start {start_idx} done {elapsed:.0f}s  "
          f"nfev={res.nfev}  f={res.fun:.4f}  "
          f"x={[round(float(v),3) for v in res.x]}", flush=True)
    return dict(mode=mode, start=start_idx, x0=list(map(float, x0)),
                 x_opt=list(map(float, res.x)), f_opt=float(res.fun),
                 nfev=int(res.nfev), elapsed=elapsed,
                 ov_list=ov_list)


def main():
    t0 = time.time()
    max_workers = int(os.environ.get("TPPR_PARALLEL", "5"))
    if MODE not in MODE_STARTS:
        raise ValueError(f"TPPR_MODE must be one of {list(MODE_STARTS)}; got {MODE!r}")
    starts = MODE_STARTS[MODE]
    print(f"[start] TP-new event-direction fit on whole network; mode={MODE}; "
          f"beta={BETA_FIXED}, rho={RHO_FIXED} fixed; max_workers={max_workers}")
    print(f"  starts: {len(starts)}  {starts}")

    runs = []
    with ProcessPoolExecutor(max_workers=max_workers,
                              initializer=_worker_init) as ex:
        futs = {ex.submit(worker_run, (MODE, j, x0)): j
                 for j, x0 in enumerate(starts)}
        for fut in as_completed(futs):
            runs.append(fut.result())
    runs.sort(key=lambda r: r["start"])
    best = min(runs, key=lambda r: r["f_opt"])
    a_s, a_l = _unpack(MODE, best["x_opt"])

    summary = dict(
        model="TP-new (mass-conserving) event-direction",
        mode=MODE,
        beta_fixed=BETA_FIXED, rho_fixed=RHO_FIXED,
        pairs=PAIRS,
        starts=[{k: v for k, v in r.items() if k != "ov_list"} for r in runs],
        best=dict(x_opt=best["x_opt"], f_opt=best["f_opt"],
                   alpha_s=a_s, alpha_l=a_l),
        best_overall_mre_mean=float(np.mean(best["ov_list"])),
        best_overall_mre_median=float(np.median(best["ov_list"])),
        per_pair_ov=best["ov_list"],
    )
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {time.time()-t0:.0f}s  saved {OUT}")
    print(f"  best: a_s={a_s:+.3f}  a_l={a_l:+.3f}  "
          f"Ov.MRE={summary['best_overall_mre_mean']:.4f}  "
          f"(median {summary['best_overall_mre_median']:.4f})")


if __name__ == "__main__":
    main()
