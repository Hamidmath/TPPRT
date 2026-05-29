"""TP-new origin -> popularity FULL-CORPUS ladder WITH angle term alpha_ang.

Same setup as sp_origin_pop_full_ladder_ang.py but TP-new chain at
beta = 0.102, rho = 0.101 FIXED. Modes: lanes_ang, speed_ang, both_ang.

Output: results/tp_new_origin_pop_full_ladder_ang_<MODE>.json
"""
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
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

ORIGINS_NPZ = str(INPUTS / "origins_results.npz")
SMOOTH_NPZ  = str(INPUTS / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON  = str(INPUTS / "city_graph_full.json")
NETWORK_XML = str(INPUTS / "slc_network.xml")

EPS = 1e-6
EPS_ANG = 0.05
TOL, MAX_ITER = 1e-5, 200
BETA_FIXED    = 0.102
RHO_FIXED     = 0.101
LAPLACE_ALPHA = 0.01
GAMMA         = 0.20

TRAIN_START = "2018-08-31 19:00:00"
TRAIN_END   = "2018-09-23 18:55:00"
TEST_START  = "2018-09-23 19:00:00"
TEST_END    = "2018-09-30 18:55:00"

MODE = os.environ.get("TPPR_LADDER_MODE", "speed_ang")
OUT  = RESULTS / f"tp_new_origin_pop_full_ladder_ang_{MODE}.json"

LANES_ANG_STARTS = [[0.0, 0.0], [+0.5, +1.0], [-0.5, -1.0],
                     [+1.0, +0.5], [+0.5, -1.0]]
SPEED_ANG_STARTS = [[0.0, 0.0], [+1.0, +1.0], [-1.0, -1.0],
                     [+2.65, +0.5], [+3.0, +2.0]]
BOTH_ANG_STARTS  = [[0.0, 0.0, 0.0], [+1.0, +0.5, +1.0],
                     [-1.0, -1.0, -1.0], [+2.65, +0.17, +0.5],
                     [+3.0, +1.0, +2.0]]
MODE_STARTS = {
    "lanes_ang":  LANES_ANG_STARTS,
    "speed_ang":  SPEED_ANG_STARTS,
    "both_ang":   BOTH_ANG_STARTS,
}


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


def parse_link_dirs(xml_path):
    nodes = {}; link_dir = {}
    for ev, el in ET.iterparse(xml_path, events=("start",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("x")), float(el.get("y")))
        elif el.tag == "link":
            f, t = el.get("from"), el.get("to")
            if f in nodes and t in nodes:
                x1, y1 = nodes[f]; x2, y2 = nodes[t]
                dx, dy = x2 - x1, y2 - y1
                norm = (dx*dx + dy*dy) ** 0.5
                if norm > 0:
                    link_dir[el.get("id")] = (dx / norm, dy / norm)
            el.clear()
    return link_dir


def build_per_source_arrays(graph, links, lid_to_idx, link_dir):
    adj = graph.get("adjacency", {})
    succ_idx = [None] * len(links)
    cos_th   = [None] * len(links)
    for i, lid in enumerate(links):
        succ_lids = [ol for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ_lids:
            succ_idx[i] = np.empty(0, dtype=np.int64)
            cos_th[i]   = np.empty(0, dtype=np.float64)
            continue
        idxs = np.array([lid_to_idx[ol] for ol in succ_lids], dtype=np.int64)
        v_i = link_dir.get(lid)
        if v_i is None:
            cs = np.zeros(len(succ_lids), dtype=np.float64)
        else:
            cs = np.empty(len(succ_lids), dtype=np.float64)
            for k, ol in enumerate(succ_lids):
                v_j = link_dir.get(ol)
                if v_j is None:
                    cs[k] = 0.0
                else:
                    c = v_i[0]*v_j[0] + v_i[1]*v_j[1]
                    cs[k] = max(-1.0, min(1.0, c))
        succ_idx[i] = idxs
        cos_th[i]   = cs
    return succ_idx, cos_th


def build_P_weighted(N, succ_idx, cos_th, dest_w, a_ang, eps_ang=EPS_ANG):
    rows, cols, data = [], [], []
    for i in range(N):
        idxs = succ_idx[i]
        if idxs.size == 0:
            continue
        out_w = dest_w[idxs].copy()
        if a_ang != 0.0:
            out_w = out_w * np.power(1.0 + cos_th[i] + eps_ang, a_ang)
        total = out_w.sum()
        if total <= 0:
            continue
        norm = out_w / total
        for k, j in enumerate(idxs):
            rows.append(int(j)); cols.append(i); data.append(float(norm[k]))
    return csr_matrix((data, (rows, cols)),
                       shape=(N, N), dtype=np.float64)


def build_P_diffusion(graph, links, lid_to_idx):
    adj = graph.get("adjacency", {})
    rows, cols, data = [], [], []
    for lid, out_links in adj.items():
        if lid not in lid_to_idx: continue
        i = lid_to_idx[lid]
        valid = [lid_to_idx[ol] for ol in out_links if ol in lid_to_idx]
        if not valid: continue
        w = 1.0 / len(valid)
        for j in valid:
            rows.append(i); cols.append(j); data.append(w)
    return csr_matrix((data, (rows, cols)),
                       shape=(len(links), len(links)), dtype=np.float64)


def tp_new_iter(P_cs, u_in, beta, rho, tol=TOL, max_iter=MAX_ITER):
    u = u_in.astype(np.float64, copy=True)
    N = u.shape[0]
    v_up = u.copy(); v_down = np.zeros(N)
    for _ in range(max_iter):
        s_down = float(v_down.sum())
        v_up_new   = (1.0 - beta) * (P_cs @ v_up) + rho * u * s_down
        v_down_new = beta * v_up + (1.0 - rho) * (P_cs @ v_down)
        s = float(v_up_new.sum() + v_down_new.sum())
        if s > 0: v_up_new /= s; v_down_new /= s
        diff = float(np.abs(v_up_new - v_up).sum() + np.abs(v_down_new - v_down).sum())
        v_up, v_down = v_up_new, v_down_new
        if diff < tol: break
    return v_up + v_down


def load_pop_block(npz_path, N, lid_to_idx, t_start, t_end):
    b = load_popularity_npz(npz_path)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_ids = [str(lid) for lid in b["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_ids], dtype=np.int64)
    valid = proj >= 0
    sel_t = [(i, t) for i, t in enumerate(times) if t_start <= t <= t_end]
    sel_idx = [i for i, _ in sel_t]
    sel_lbl = [t for _, t in sel_t]
    out = np.zeros((len(sel_idx), N), dtype=np.float32)
    for k, ti in enumerate(sel_idx):
        row = M.getrow(ti).toarray().ravel().astype(np.float64)
        E = np.zeros(N, dtype=np.float64)
        np.add.at(E, proj[valid], row[valid])
        s = E.sum()
        if s > 0: E /= s
        out[k] = E.astype(np.float32)
    return sel_lbl, out


def diffuse_one(raw_row, raw_proj, raw_valid, N, P_diff):
    c = np.full(N, LAPLACE_ALPHA, dtype=np.float64)
    if raw_row is not None:
        c_add = np.zeros(N, dtype=np.float64)
        np.add.at(c_add, raw_proj[raw_valid], raw_row[raw_valid])
        c = c + c_add
    c_diff = (1.0 - GAMMA) * c + GAMMA * (P_diff @ c)
    s = c_diff.sum()
    if s > 0: c_diff /= s
    return c_diff


def load_diffused_origins_for_times(npz_path, N, lid_to_idx, time_labels, P_diff):
    b = load_popularity_npz(npz_path)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    orig_ids = [str(lid) for lid in b["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in orig_ids], dtype=np.int64)
    valid = proj >= 0
    time_idx = {t: i for i, t in enumerate(times)}
    out = np.zeros((len(time_labels), N), dtype=np.float32)
    for k, t in enumerate(time_labels):
        ti = time_idx.get(t)
        if ti is None:
            v = diffuse_one(None, proj, valid, N, P_diff)
        else:
            raw = M.getrow(ti).toarray().ravel().astype(np.float64)
            v = diffuse_one(raw, proj, valid, N, P_diff)
        out[k] = v.astype(np.float32)
    return out


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
    link_dir = parse_link_dirs(NETWORK_XML)
    succ_idx, cos_th = build_per_source_arrays(graph, links, lid_to_idx, link_dir)

    train_times, train_targets = load_pop_block(SMOOTH_NPZ, N, lid_to_idx,
                                                  TRAIN_START, TRAIN_END)
    test_times,  test_targets  = load_pop_block(SMOOTH_NPZ, N, lid_to_idx,
                                                  TEST_START, TEST_END)
    train_inputs = load_diffused_origins_for_times(ORIGINS_NPZ, N, lid_to_idx,
                                                      train_times, P_diff)
    test_inputs  = load_diffused_origins_for_times(ORIGINS_NPZ, N, lid_to_idx,
                                                      test_times, P_diff)

    WORKER.update(dict(
        graph=graph, links=links, lid_to_idx=lid_to_idx, N=N,
        speeds=speeds, lanes=lanes,
        succ_idx=succ_idx, cos_th=cos_th,
        train_inputs=train_inputs, train_targets=train_targets,
        test_inputs=test_inputs,   test_targets=test_targets,
        kernel_cache={},
    ))
    print(f"  worker pid {os.getpid()} ready  N={N:,}  "
          f"train={len(train_times)}  test={len(test_times)}  "
          f"link_dirs={len(link_dir):,}", flush=True)


def _kernel(mode, x):
    a_s = a_l = a_ang = 0.0
    if mode == "lanes_ang":
        a_l, a_ang = float(x[0]), float(x[1])
    elif mode == "speed_ang":
        a_s, a_ang = float(x[0]), float(x[1])
    elif mode == "both_ang":
        a_s, a_l, a_ang = float(x[0]), float(x[1]), float(x[2])
    key = ("tp_ang", round(a_s, 4), round(a_l, 4), round(a_ang, 4))
    cache = WORKER["kernel_cache"]
    if key in cache: return cache[key]
    speeds, lanes = WORKER["speeds"], WORKER["lanes"]
    N = WORKER["N"]
    dest_w = np.ones(N, dtype=np.float64)
    if a_s != 0: dest_w = dest_w * np.power(speeds, a_s)
    if a_l != 0: dest_w = dest_w * np.power(lanes, a_l)
    P = build_P_weighted(N, WORKER["succ_idx"], WORKER["cos_th"],
                          dest_w, a_ang)
    cache[key] = P
    if len(cache) > 32:
        for k in list(cache.keys())[:16]:
            del cache[k]
    return P


def _eval_block(P, inputs, targets):
    ov_list = []
    for i in range(len(inputs)):
        v = tp_new_iter(P, inputs[i], BETA_FIXED, RHO_FIXED)
        rel = np.abs(targets[i].astype(np.float64) - v) / (targets[i].astype(np.float64) + EPS)
        ov_list.append(float(rel.mean()))
    return ov_list


def _loss(mode, x):
    P = _kernel(mode, x)
    ov = _eval_block(P, WORKER["train_inputs"], WORKER["train_targets"])
    return float(np.mean(ov)), ov


def worker_run(payload):
    mode, start_idx, x0 = payload
    ts = time.time()
    bounds = [(None, None)] * len(x0)
    res = minimize(lambda x: _loss(mode, x)[0], np.array(x0, dtype=np.float64),
                    method="L-BFGS-B", bounds=bounds,
                    options=dict(eps=2e-3, ftol=1e-6, gtol=1e-4, maxiter=20))
    f_opt, ov_train = _loss(mode, res.x)
    P_best = _kernel(mode, res.x)
    ov_test = _eval_block(P_best, WORKER["test_inputs"], WORKER["test_targets"])
    elapsed = time.time() - ts
    print(f"  [{mode}] start {start_idx} done {elapsed:.0f}s  "
          f"train Ov.MRE={float(np.mean(ov_train)):.4f}  "
          f"test Ov.MRE={float(np.mean(ov_test)):.4f}  "
          f"x={[round(float(v),3) for v in res.x]}", flush=True)
    return dict(mode=mode, start=start_idx, x0=list(map(float, x0)),
                 x_opt=list(map(float, res.x)), f_opt=float(res.fun),
                 nfev=int(res.nfev), elapsed=elapsed,
                 ov_train=ov_train, ov_test=ov_test)


def main():
    t0 = time.time()
    max_workers = int(os.environ.get("TPPR_PARALLEL", "5"))
    print(f"[start] TP-new origin->pop FULL+ANGLE ladder, "
          f"beta={BETA_FIXED}, rho={RHO_FIXED} fixed, eps_ang={EPS_ANG}, "
          f"mode={MODE}, max_workers={max_workers}")

    if MODE not in MODE_STARTS:
        raise ValueError(f"TPPR_LADDER_MODE must be in {list(MODE_STARTS)}; "
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
    a_s = a_l = a_ang = 0.0
    if MODE == "lanes_ang":
        a_l, a_ang = best["x_opt"]
    elif MODE == "speed_ang":
        a_s, a_ang = best["x_opt"]
    else:
        a_s, a_l, a_ang = best["x_opt"]

    summary = dict(
        model="TP-new origin->popularity FULL + angle",
        mode=MODE, beta_fixed=BETA_FIXED, rho_fixed=RHO_FIXED,
        eps_ang=EPS_ANG, laplace_alpha=LAPLACE_ALPHA, gamma=GAMMA,
        train_range=[TRAIN_START, TRAIN_END],
        test_range=[TEST_START, TEST_END],
        best=dict(x_opt=best["x_opt"], f_opt=best["f_opt"],
                   alpha_s=a_s, alpha_l=a_l, alpha_ang=a_ang),
        train_overall_mre_mean=float(np.mean(best["ov_train"])),
        train_overall_mre_median=float(np.median(best["ov_train"])),
        test_overall_mre_mean=float(np.mean(best["ov_test"])),
        test_overall_mre_median=float(np.median(best["ov_test"])),
        starts=[{k: v for k, v in r.items()
                  if k not in ("ov_train", "ov_test")} for r in runs],
        per_slot_train=best["ov_train"],
        per_slot_test=best["ov_test"],
    )
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"[done] {time.time()-t0:.0f}s saved {OUT}")
    print(f"  best: a_s={a_s:+.3f}  a_l={a_l:+.3f}  a_ang={a_ang:+.3f}  "
          f"TRAIN MRE={summary['train_overall_mre_mean']:.4f}  "
          f"TEST MRE={summary['test_overall_mre_mean']:.4f}")


if __name__ == "__main__":
    main()
