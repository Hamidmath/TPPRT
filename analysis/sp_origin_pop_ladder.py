"""SP origin -> popularity ladder.

For each slot t in the 840-slot eval set (same sample_pairs() rule as
sp_prof_ladder.py, BPW=120, SEED=7), we:

  E_t  := raw trip-ORIGINS at slot t (from data/origins_results.npz),
          projected to the 99,716-link graph and DIFFUSED with the
          same Laplace alpha=0.01 + one-step graph-diffusion gamma=0.20
          as the smoothed popularity file.
  F_t  := diffused POPULARITY at slot t (from
          data/popularity_results_smoothed_osm_gamma020.npz).
  v_t  := SP chain output at parameters (alpha_s, alpha_l), alpha
          fixed at the geometric-fit value 0.0508, fed with E_t.

Per slot Overall MRE = mean over links of |F_t - v_t| / (F_t + EPS).
Loss = mean over 840 slots. Tune alpha_s and/or alpha_l.

The single-mode entry-point is set via TPPR_LADDER_MODE in:
    uniform_prior, diffuse_prior, lanes_only, speed_only, both

Mode 'three' / 'length_only' from sp_prof_ladder are intentionally not
shipped here; the professor said lanes/length add nothing.

Output: results/sp_origin_pop_ladder_<MODE>.json
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

ORIGINS_NPZ = str(INPUTS / "origins_results.npz")
SMOOTH_NPZ  = str(INPUTS / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON  = str(INPUTS / "city_graph_full.json")

EPS = 1e-6
SEED = 7
BPW = int(os.environ.get("TPPR_BPW", "120"))  # bins-per-weekday; default 120 -> 840 slots
TOL, MAX_ITER = 1e-5, 100
ALPHA_FIXED   = 0.0508          # SP restart, geometric fit
LAPLACE_ALPHA = 0.01            # diffusion Laplace pseudo-count
GAMMA         = 0.20            # diffusion mix

MODE = os.environ.get("TPPR_LADDER_MODE", "speed_only")
OUT = RESULTS / f"sp_origin_pop_ladder_{MODE}.json"

LANE_STARTS  = [[0.0], [+0.5], [-0.5], [+1.0], [-1.0]]
SPEED_STARTS = [[0.0], [+1.0], [-1.0], [+2.65], [-2.0]]
BOTH_STARTS  = [[0.0, 0.0], [+1.0, +0.5], [-1.0, -1.0],
                 [+2.65, +0.17], [+3.0, +1.0]]
MODE_STARTS = {"lanes_only": LANE_STARTS, "speed_only": SPEED_STARTS,
                "both": BOTH_STARTS}


def build_P_weighted(graph, links, lid_to_idx, weights):
    """Outgoing-weight-normalized transition (column-stochastic via P @ v)."""
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


def build_P_diffusion(graph, links, lid_to_idx):
    """Out-degree-row-stochastic adjacency P_diff for smoothing.
    Equivalent to pipeline/generate_smoothed.py: P[i, j] = 1/deg(i) for j
    in adj[i]. Used as (1-gamma) c + gamma * P_diff @ c per slot.
    """
    adj = graph.get("adjacency", {})
    rows, cols, data = [], [], []
    for lid, out_links in adj.items():
        if lid not in lid_to_idx:
            continue
        i = lid_to_idx[lid]
        valid = [lid_to_idx[ol] for ol in out_links if ol in lid_to_idx]
        if not valid:
            continue
        w = 1.0 / len(valid)
        for j in valid:
            rows.append(i); cols.append(j); data.append(w)
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


def sample_slots(times, seed=SEED, bpw=BPW):
    """Mirror sp_prof_ladder.sample_pairs but return only the source t."""
    rng = random.Random(seed)
    by_wd = {i: [] for i in range(7)}
    tset = set(times)
    for t in times:
        dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        sib = (dt + timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S")
        if sib in tset:
            by_wd[dt.weekday()].append(t)
    slots = []
    for wd in range(7):
        pool = by_wd[wd]
        rng.shuffle(pool)
        slots.extend(pool[:bpw])
    return slots


def load_diffused_pop(npz_path, N, lid_to_idx, needed_times):
    """Project per-slot diffused popularity to the (99716,) graph vector."""
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


def diffuse_origin_slot(raw_row, raw_proj, raw_valid, N, P_diff,
                         laplace_alpha=LAPLACE_ALPHA, gamma=GAMMA):
    c = np.full(N, laplace_alpha, dtype=np.float64)
    if raw_row is not None:
        c_add = np.zeros(N, dtype=np.float64)
        np.add.at(c_add, raw_proj[raw_valid], raw_row[raw_valid])
        c = c + c_add
    c_diff = (1.0 - gamma) * c + gamma * (P_diff @ c)
    s = c_diff.sum()
    if s > 0:
        c_diff /= s
    return c_diff


def load_diffused_origins(npz_path, N, lid_to_idx, needed_times, P_diff):
    """Read origins raw npz, project to graph universe per slot, apply
    Laplace + one-step diffusion to match generate_smoothed.py."""
    b = load_popularity_npz(npz_path)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    orig_ids = [str(lid) for lid in b["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in orig_ids], dtype=np.int64)
    valid = proj >= 0
    needed = set(needed_times)
    time_idx = {t: i for i, t in enumerate(times)}
    rows = {}
    for t in needed:
        ti = time_idx.get(t)
        if ti is None:
            # Slot has zero origins; diffuse from Laplace-only baseline.
            rows[t] = diffuse_origin_slot(None, proj, valid, N, P_diff)
            continue
        raw = M.getrow(ti).toarray().ravel().astype(np.float64)
        rows[t] = diffuse_origin_slot(raw, proj, valid, N, P_diff)
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
    P_diff = build_P_diffusion(graph, links, lid_to_idx)
    b = load_popularity_npz(SMOOTH_NPZ)
    all_times = [str(t) for t in b["times"]]
    del b
    slots = sample_slots(sorted(all_times))
    pop_rows = load_diffused_pop(SMOOTH_NPZ, N, lid_to_idx, slots)
    orig_rows = load_diffused_origins(ORIGINS_NPZ, N, lid_to_idx, slots, P_diff)
    inputs  = [orig_rows[t] for t in slots]
    targets = [pop_rows[t]  for t in slots]
    WORKER.update(dict(
        graph=graph, links=links, lid_to_idx=lid_to_idx, N=N,
        speeds=speeds, lanes=lanes, slots=slots,
        inputs=inputs, targets=targets, kernel_cache={},
    ))
    print(f"  worker pid {os.getpid()} ready  N={N:,}, slots={len(slots)}",
          flush=True)


def _kernel(mode, x):
    a_s = a_l = 0.0
    if mode == "lanes_only":
        a_l = float(x[0])
    elif mode == "speed_only":
        a_s = float(x[0])
    elif mode == "both":
        a_s, a_l = float(x[0]), float(x[1])
    key = ("sp", round(a_s, 4), round(a_l, 4))
    cache = WORKER["kernel_cache"]
    if key in cache:
        return cache[key]
    speeds, lanes = WORKER["speeds"], WORKER["lanes"]
    N = WORKER["N"]
    w = np.ones(N, dtype=np.float64)
    if a_s != 0: w = w * np.power(speeds, a_s)
    if a_l != 0: w = w * np.power(lanes, a_l)
    P = build_P_weighted(WORKER["graph"], WORKER["links"],
                          WORKER["lid_to_idx"], w)
    cache[key] = P
    if len(cache) > 64:
        for k in list(cache.keys())[:32]:
            del cache[k]
    return P


def _loss(mode, x):
    P = _kernel(mode, x)
    inputs  = WORKER["inputs"]
    targets = WORKER["targets"]
    ov_list = []
    for i in range(len(inputs)):
        v = sp_iter(P, inputs[i], ALPHA_FIXED)
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
          f"nfev={res.nfev}  f={res.fun:.4f}  x={[round(float(v),3) for v in res.x]}",
          flush=True)
    return dict(mode=mode, start=start_idx, x0=list(map(float, x0)),
                 x_opt=list(map(float, res.x)), f_opt=float(res.fun),
                 nfev=int(res.nfev), elapsed=elapsed, ov_list=ov_list)


def baseline_eval(mode):
    P_uniform = _kernel("none", [0.0, 0.0])
    if mode == "uniform_prior":
        N = WORKER["N"]
        u = np.full(N, 1.0 / N, dtype=np.float64)
        v_star = sp_iter(P_uniform, u, ALPHA_FIXED)
        per = [float(np.mean(np.abs(t - v_star) / (t + EPS)))
               for t in WORKER["targets"]]
        return dict(mode=mode, overall_mean=float(np.mean(per)),
                     overall_median=float(np.median(per)))
    inputs, targets = WORKER["inputs"], WORKER["targets"]
    ov_list = []
    for i in range(len(inputs)):
        v = sp_iter(P_uniform, inputs[i], ALPHA_FIXED)
        rel = np.abs(targets[i] - v) / (targets[i] + EPS)
        ov_list.append(float(rel.mean()))
    return dict(mode=mode, overall_mean=float(np.mean(ov_list)),
                 overall_median=float(np.median(ov_list)))


def main():
    t0 = time.time()
    max_workers = int(os.environ.get("TPPR_PARALLEL", "5"))
    print(f"[start] SP origin->pop ladder, alpha={ALPHA_FIXED} fixed, "
          f"mode={MODE}, BPW={BPW}, max_workers={max_workers}")

    if MODE in ("uniform_prior", "diffuse_prior"):
        _worker_init()
        r = baseline_eval(MODE)
        print(f"  {MODE}: Ov.MRE mean={r['overall_mean']:.4f} "
              f"median={r['overall_median']:.4f}")
        OUT.write_text(json.dumps(r, indent=2))
        print(f"[done] {time.time()-t0:.0f}s saved {OUT}")
        return

    if MODE not in MODE_STARTS:
        raise ValueError(f"TPPR_LADDER_MODE must be one of "
                          f"{list(MODE_STARTS) + ['uniform_prior','diffuse_prior']}; "
                          f"got {MODE!r}")
    starts = MODE_STARTS[MODE]
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
    a_s = a_l = 0.0
    if MODE == "lanes_only":
        a_l = best["x_opt"][0]
    elif MODE == "speed_only":
        a_s = best["x_opt"][0]
    else:
        a_s, a_l = best["x_opt"]

    summary = dict(
        model="SP origin->popularity",
        mode=MODE, alpha_fixed=ALPHA_FIXED,
        laplace_alpha=LAPLACE_ALPHA, gamma=GAMMA, bpw=BPW, seed=SEED,
        best=dict(x_opt=best["x_opt"], f_opt=best["f_opt"],
                   alpha_s=a_s, alpha_l=a_l),
        best_overall_mre_mean=float(np.mean(best["ov_list"])),
        best_overall_mre_median=float(np.median(best["ov_list"])),
        starts=[{k: v for k, v in r.items() if k != "ov_list"} for r in runs],
        per_slot_ov=best["ov_list"],
    )
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"[done] {time.time()-t0:.0f}s saved {OUT}")
    print(f"  best: a_s={a_s:+.3f}  a_l={a_l:+.3f}  "
          f"Ov.MRE={summary['best_overall_mre_mean']:.4f}  "
          f"(median {summary['best_overall_mre_median']:.4f})")


if __name__ == "__main__":
    main()
