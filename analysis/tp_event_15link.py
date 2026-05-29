"""TP-new (mass-conserving two-phase) event-direction fit on the
15-link region inside the 2 km circle at (40.722348, -111.904691).

Setup (locked):
  pairs   : (Sep 11 16:MM, Sep 18 16:MM) for MM in 00, 05, ..., 55  (12)
  input   : diffused popularity at Sep 11 (week 2)
  target  : diffused popularity at Sep 18 (week 3)
  chain   : new-form TP, beta = 0.102, rho = 0.101, BOTH FIXED
  loss    : Overall MRE restricted to the 15 region links
  search  : L-BFGS-B, 5 parallel starts; alpha_s, alpha_l, alpha_len unbounded

Mode is chosen via environment variable TPPR_MODE in one of:
  lanes_only   tune alpha_l
  speed_only   tune alpha_s
  length_only  tune alpha_len
  both         tune alpha_s, alpha_l
  three        tune alpha_s, alpha_l, alpha_len

Output: results/tp_event_15link_<MODE>.json
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

MODE = os.environ.get("TPPR_MODE", "three")
OUT = RESULTS / f"tp_event_15link_{MODE}.json"

EPS = 1e-6
TOL, MAX_ITER = 1e-6, 200
BETA_FIXED = 0.102
RHO_FIXED  = 0.101
PAIRS = [(f"2018-09-11 16:{m:02d}:00", f"2018-09-18 16:{m:02d}:00")
          for m in range(0, 60, 5)]

REGION15 = ["58905","58889","58864","58870","58898","58888","58869","58862",
            "59159","59261","59071","59157","58874","58923","59260"]

LANE_STARTS   = [[0.0], [+0.5], [-0.5], [+1.0], [-1.0]]
SPEED_STARTS  = [[0.0], [+1.0], [-1.0], [+2.65], [-2.0]]
LEN_STARTS    = [[0.0], [+0.5], [-0.5], [+1.0], [-1.0]]
BOTH_STARTS   = [[0.0, 0.0], [+1.0, +0.5], [-1.0, -1.0],
                  [+2.65, +0.17], [+3.0, +1.0]]
THREE_STARTS  = [[0.0, 0.0, 0.0], [+1.0, +0.5, +0.5],
                  [-1.0, -1.0, +1.0], [+2.65, +0.17, 0.0],
                  [+3.0, +1.0, -0.5]]
MODE_STARTS = {"lanes_only": LANE_STARTS, "speed_only": SPEED_STARTS,
                "length_only": LEN_STARTS, "both": BOTH_STARTS,
                "three": THREE_STARTS}


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
    speeds, lanes, lengths = get_speed_lane_length(graph, links)
    needed = set()
    for t, sib in PAIRS:
        needed.add(t); needed.add(sib)
    rows = load_diffused(SMOOTH_NPZ, N, lid_to_idx, needed)
    inputs  = [rows[t]   for t, _   in PAIRS]
    targets = [rows[sib] for _, sib in PAIRS]
    region_idx = np.array([lid_to_idx[l] for l in REGION15
                            if l in lid_to_idx], dtype=np.int64)
    WORKER.update(dict(
        graph=graph, links=links, lid_to_idx=lid_to_idx, N=N,
        speeds=speeds, lanes=lanes, lengths=lengths,
        inputs=inputs, targets=targets, region_idx=region_idx,
        kernel_cache={},
    ))
    print(f"  worker pid {os.getpid()} ready  N={N:,}, pairs={len(PAIRS)}, "
          f"region_links={len(region_idx)}", flush=True)


def _kernel(a_s, a_l, a_len):
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
    if len(cache) > 32:
        for k in list(cache.keys())[:16]:
            del cache[k]
    return P


def _unpack(mode, x):
    if mode == "lanes_only":   return 0.0, float(x[0]), 0.0
    if mode == "speed_only":   return float(x[0]), 0.0, 0.0
    if mode == "length_only":  return 0.0, 0.0, float(x[0])
    if mode == "both":         return float(x[0]), float(x[1]), 0.0
    if mode == "three":        return float(x[0]), float(x[1]), float(x[2])
    raise ValueError(mode)


def _loss(mode, x):
    a_s, a_l, a_len = _unpack(mode, x)
    P = _kernel(a_s, a_l, a_len)
    region_idx = WORKER["region_idx"]
    inputs  = WORKER["inputs"]
    targets = WORKER["targets"]
    region_mre, full_ov = [], []
    for i in range(len(inputs)):
        v = tp_new_iter(P, inputs[i], BETA_FIXED, RHO_FIXED)
        rel = np.abs(targets[i] - v) / (targets[i] + EPS)
        region_mre.append(float(rel[region_idx].mean()))
        full_ov.append(float(rel.mean()))
    return float(np.mean(region_mre)), region_mre, full_ov


def worker_run(payload):
    mode, start_idx, x0 = payload
    ts = time.time()
    bounds = [(None, None)] * len(x0)
    res = minimize(lambda x: _loss(mode, x)[0], np.array(x0, dtype=np.float64),
                    method="L-BFGS-B", bounds=bounds,
                    options=dict(eps=2e-3, ftol=1e-6, gtol=1e-4, maxiter=20))
    f_opt, region_mre, full_ov = _loss(mode, res.x)
    elapsed = time.time() - ts
    print(f"  [{mode}] start {start_idx} done {elapsed:.0f}s  "
          f"nfev={res.nfev}  f={res.fun:.4f}  "
          f"x={[round(float(v),3) for v in res.x]}", flush=True)
    return dict(mode=mode, start=start_idx, x0=list(map(float, x0)),
                 x_opt=list(map(float, res.x)), f_opt=float(res.fun),
                 nfev=int(res.nfev), elapsed=elapsed,
                 region_mre=region_mre, full_ov=full_ov)


def main():
    t0 = time.time()
    max_workers = int(os.environ.get("TPPR_PARALLEL", "5"))
    if MODE not in MODE_STARTS:
        raise ValueError(f"TPPR_MODE must be one of {list(MODE_STARTS)}; got {MODE!r}")
    starts = MODE_STARTS[MODE]
    print(f"[start] TP-new event-direction fit on 15-link region; "
          f"mode={MODE}; beta={BETA_FIXED}, rho={RHO_FIXED} fixed; "
          f"max_workers={max_workers}")
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
    a_s, a_l, a_len = _unpack(MODE, best["x_opt"])

    summary = dict(
        model="TP-new (mass-conserving) event-direction",
        mode=MODE,
        beta_fixed=BETA_FIXED, rho_fixed=RHO_FIXED,
        pairs=PAIRS,
        region_link_ids=REGION15,
        starts=[{k: v for k, v in r.items()
                  if k not in ("region_mre", "full_ov")} for r in runs],
        best=dict(x_opt=best["x_opt"], f_opt=best["f_opt"],
                   alpha_s=a_s, alpha_l=a_l, alpha_len=a_len),
        best_region_mre_mean=float(np.mean(best["region_mre"])),
        best_region_mre_median=float(np.median(best["region_mre"])),
        best_full_ov_mean=float(np.mean(best["full_ov"])),
        per_pair_region_mre=best["region_mre"],
        per_pair_full_ov=best["full_ov"],
    )
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {time.time()-t0:.0f}s  saved {OUT}")
    print(f"  best:  a_s={a_s:+.3f}  a_l={a_l:+.3f}  a_len={a_len:+.3f}")
    print(f"         region MRE = {summary['best_region_mre_mean']:.4f}  "
          f"(median {summary['best_region_mre_median']:.4f}),  "
          f"full OV = {summary['best_full_ov_mean']:.4f}")


if __name__ == "__main__":
    main()
