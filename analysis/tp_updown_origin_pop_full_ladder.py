"""TP up/down origin -> popularity FULL-CORPUS ladder, paper-faithful.

Two-phase PageRank with SEPARATE up/down kernels, matching the paper's
block chain (eq:M-block):

    M_b = [ (1-beta) P^up      rho * Ebar_b 1^T  ]
          [ beta I             (1-rho) P^down    ]

P^up and P^down are built on the SAME forward road graph but with
INDEPENDENT speed/lane exponents, and a SINGLE SHARED angle exponent:

    w^up(j,i)   = sbar(j)^{a_s_up} * lbar(j)^{a_l_up} * a(j,i)^{a_ang}
    w^down(j,i) = sbar(j)^{a_s_dn} * lbar(j)^{a_l_dn} * a(j,i)^{a_ang}
    a(j,i)      = 1 + cos theta(j,i)            (delta = 0)
    P^*_{j,i}   = w^*(j,i) / sum_{j' in J_i} w^*(j',i)

Conventions per user spec:
  - delta = 0 (no U-turn floor); a true U-turn has a(j,i) = 0.
  - a_ang >= 0 (bounded in the optimiser).
  - 0^0 = 1 (when a_ang = 0 the angle factor is identically 1, so
    U-turn edges with base 0 are kept; numpy's power(0,0)=1 anyway).

beta = 0.102, rho = 0.101 FIXED. Five free parameters at most:
a_s_up, a_s_dn, a_l_up, a_l_dn, a_ang.

Mode via TPPR_LADDER_MODE in: sp, la, sp_la, la_ang, sp_ang, sp_la_ang.
Train = first 23 days (6,624 slots); test = last 7 days (2,016 slots).

Output: results/tp_updown_origin_pop_full_ladder_<MODE>.json
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
TOL, MAX_ITER = 1e-5, 200
BETA_FIXED    = 0.102
RHO_FIXED     = 0.101
LAPLACE_ALPHA = 0.01
GAMMA         = 0.20

TRAIN_START = "2018-08-31 19:00:00"
TRAIN_END   = "2018-09-23 18:55:00"
TEST_START  = "2018-09-23 19:00:00"
TEST_END    = "2018-09-30 18:55:00"

MODE = os.environ.get("TPPR_LADDER_MODE", "sp_la_ang")
# TPPR_ANORM=1 divides the angle base (1 + cos theta) by its network
# mean abar before the power, matching the paper's abar normalisation.
# It is a global constant that cancels in the per-source column
# normalisation, so results are identical; the toggle exists for a
# paper-exact run. Default off.
ANORM = os.environ.get("TPPR_ANORM", "0") == "1"
# delta added to the angle base (1 + cos theta + delta); default 0.
DELTA = float(os.environ.get("TPPR_DELTA", "0.0"))
# bound the shared angle exponent a_ang >= 0? default yes.
ANG_NONNEG = os.environ.get("TPPR_ANG_NONNEG", "1") == "1"
# optional filename tag to keep variant outputs distinct.
TAG = os.environ.get("TPPR_TAG", "")
_tag = f"_{TAG}" if TAG else ""
# TPPR_START_IDX >= 0 runs ONLY that single restart and writes a
# per-start file (for fully-parallelized one-search-per-job runs);
# -1 (default) runs all restarts in one job as before.
START_IDX = int(os.environ.get("TPPR_START_IDX", "-1"))
_sfx = f"_start{START_IDX}" if START_IDX >= 0 else ""
OUT  = RESULTS / (f"tp_updown_origin_pop_full_ladder"
                   f"{'_anorm' if ANORM else ''}{_tag}_{MODE}{_sfx}.json")

# Smoke-test knobs (0 = use everything). Do NOT set for production.
MAX_SLOTS = int(os.environ.get("TPPR_MAX_SLOTS", "0"))
NSTARTS   = int(os.environ.get("TPPR_NSTARTS", "0"))

# Random restarts per mode. Angle starts are all >= 0 (bounded).
# Layout matches _unpack(): up/down speed, up/down lane, shared angle.
MODE_STARTS = {
    "sp":        [[0.0, 0.0], [+2.0, -2.0], [+1.0, -1.0],
                   [+3.0, -1.0], [-1.0, +1.0]],
    "la":        [[0.0, 0.0], [+0.7, -0.7], [+1.0, -0.5],
                   [+0.5, +0.5], [-0.5, -0.5]],
    "sp_la":     [[0.0, 0.0, 0.0, 0.0], [+1.7, -1.7, +0.4, -0.4],
                   [+2.0, -1.0, +0.5, +0.5], [+1.0, +1.0, +0.4, +0.4],
                   [+2.2, -2.2, +0.7, -0.7]],
    "la_ang":    [[0.0, 0.0, 0.0], [+0.7, -0.7, +0.1],
                   [+1.0, -0.5, +0.5], [+1.7, +1.7, +0.0],
                   [+0.5, -1.0, +1.0]],
    "sp_ang":    [[0.0, 0.0, 0.0], [+2.0, -2.0, +0.1],
                   [+1.0, -1.0, +0.5], [+4.0, +0.0, +0.0],
                   [+3.0, -1.0, +1.0]],
    "sp_la_ang": [[0.0, 0.0, 0.0, 0.0, 0.0],
                   [+1.78, -1.78, +0.42, -0.42, +0.06],
                   [+2.0, -1.0, +0.5, +0.5, +0.1],
                   [+1.7, +0.4, +1.7, +0.4, +0.0],
                   [+3.0, -2.0, +1.0, -1.0, +1.0]],
    # angle only (all speed/lane exponents 0): tune just a_ang >= 0.
    # P_up == P_down here, so it is a uniform-walk two-phase chain
    # modulated only by the shared angle term.
    "ang_only":  [[0.0], [+0.1], [+0.5], [+1.0], [+2.0]],
}


def _unpack(mode, x):
    """Return (a_s_up, a_s_dn, a_l_up, a_l_dn, a_ang) from packed x."""
    a_s_up = a_s_dn = a_l_up = a_l_dn = a_ang = 0.0
    if mode == "sp":
        a_s_up, a_s_dn = float(x[0]), float(x[1])
    elif mode == "la":
        a_l_up, a_l_dn = float(x[0]), float(x[1])
    elif mode == "sp_la":
        a_s_up, a_s_dn, a_l_up, a_l_dn = (float(x[0]), float(x[1]),
                                            float(x[2]), float(x[3]))
    elif mode == "la_ang":
        a_l_up, a_l_dn, a_ang = float(x[0]), float(x[1]), float(x[2])
    elif mode == "sp_ang":
        a_s_up, a_s_dn, a_ang = float(x[0]), float(x[1]), float(x[2])
    elif mode == "sp_la_ang":
        a_s_up, a_s_dn, a_l_up, a_l_dn, a_ang = (float(x[0]), float(x[1]),
                                                   float(x[2]), float(x[3]),
                                                   float(x[4]))
    elif mode == "ang_only":
        a_ang = float(x[0])
    else:
        raise ValueError(f"bad mode {mode!r}")
    return a_s_up, a_s_dn, a_l_up, a_l_dn, a_ang


def _bounds(mode, n):
    """a_ang (shared angle, last slot when present) is bounded >= 0;
    all speed/lane exponents are unbounded."""
    b = [(None, None)] * n
    if ANG_NONNEG and mode in ("la_ang", "sp_ang", "sp_la_ang", "ang_only"):
        b[-1] = (0.0, None)
    return b


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


def build_P_weighted(N, succ_idx, cos_th, dest_w, a_ang, a_mean=1.0):
    """Column-stochastic kernel. Angle factor is (1 + cos theta)^a_ang
    with delta = 0. When a_ang == 0 the factor is identically 1 (the
    0^0 = 1 convention), so U-turn edges with base 0 are kept; when
    a_ang > 0 a true U-turn (base 0) gets weight 0. If a_mean != 1.0
    the base is divided by it (abar normalisation); a global constant
    that cancels in the column normalisation below."""
    rows, cols, data = [], [], []
    for i in range(N):
        idxs = succ_idx[i]
        if idxs.size == 0:
            continue
        out_w = dest_w[idxs].copy()
        if a_ang != 0.0:
            base = 1.0 + cos_th[i] + DELTA
            base = np.clip(base, 0.0, None)   # guard tiny fp negatives
            if a_mean != 1.0:
                base = base / a_mean
            out_w = out_w * np.power(base, a_ang)
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


def tp_iter(P_up, P_down, E_b, beta, rho, dangling_mask,
            tol=TOL, max_iter=MAX_ITER):
    """Two-phase chain with separate up/down kernels (eq:M-block). The
    mass on dangling links (zero out-degree) is routed into the
    teleport/prior E_b each step in both phases, so each kernel is
    column-stochastic and mass is conserved."""
    E_b = E_b.astype(np.float64, copy=True)
    N = E_b.shape[0]
    v_up = E_b.copy(); v_down = np.zeros(N)
    for _ in range(max_iter):
        s_down = float(v_down.sum())
        Pv_up   = P_up   @ v_up   + float(v_up[dangling_mask].sum())   * E_b
        Pv_down = P_down @ v_down + float(v_down[dangling_mask].sum()) * E_b
        v_up_n   = (1.0 - beta) * Pv_up   + rho * E_b * s_down
        v_down_n = beta * v_up + (1.0 - rho) * Pv_down
        s = float(v_up_n.sum() + v_down_n.sum())
        if s > 0: v_up_n /= s; v_down_n /= s
        diff = float(np.abs(v_up_n - v_up).sum()
                       + np.abs(v_down_n - v_down).sum())
        v_up, v_down = v_up_n, v_down_n
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

    if ANORM:
        vals = [np.clip(1.0 + c + DELTA, 0.0, None) for c in cos_th if c.size]
        a_mean = float(np.concatenate(vals).mean()) if vals else 1.0
    else:
        a_mean = 1.0

    dangling_mask = np.array([si.size == 0 for si in succ_idx], dtype=bool)

    train_times, train_targets = load_pop_block(SMOOTH_NPZ, N, lid_to_idx,
                                                  TRAIN_START, TRAIN_END)
    test_times,  test_targets  = load_pop_block(SMOOTH_NPZ, N, lid_to_idx,
                                                  TEST_START, TEST_END)
    train_inputs = load_diffused_origins_for_times(ORIGINS_NPZ, N, lid_to_idx,
                                                      train_times, P_diff)
    test_inputs  = load_diffused_origins_for_times(ORIGINS_NPZ, N, lid_to_idx,
                                                      test_times, P_diff)

    if MAX_SLOTS > 0:
        train_inputs = train_inputs[:MAX_SLOTS]
        train_targets = train_targets[:MAX_SLOTS]
        test_inputs = test_inputs[:MAX_SLOTS]
        test_targets = test_targets[:MAX_SLOTS]
        train_times = train_times[:MAX_SLOTS]
        test_times = test_times[:MAX_SLOTS]

    WORKER.update(dict(
        graph=graph, links=links, lid_to_idx=lid_to_idx, N=N,
        speeds=speeds, lanes=lanes,
        succ_idx=succ_idx, cos_th=cos_th, a_mean=a_mean,
        dangling_mask=dangling_mask,
        train_inputs=train_inputs, train_targets=train_targets,
        test_inputs=test_inputs,   test_targets=test_targets,
        kernel_cache={},
    ))
    print(f"  worker pid {os.getpid()} ready  N={N:,}  "
          f"train={len(train_times)}  test={len(test_times)}  "
          f"link_dirs={len(link_dir):,}  abar={a_mean:.6f}  "
          f"dangling={int(dangling_mask.sum())}", flush=True)


def _kernels(mode, x):
    a_s_up, a_s_dn, a_l_up, a_l_dn, a_ang = _unpack(mode, x)
    key = ("updown", round(a_s_up, 4), round(a_s_dn, 4),
           round(a_l_up, 4), round(a_l_dn, 4), round(a_ang, 4))
    cache = WORKER["kernel_cache"]
    if key in cache:
        return cache[key]
    speeds, lanes = WORKER["speeds"], WORKER["lanes"]
    N = WORKER["N"]
    w_up = np.ones(N, dtype=np.float64)
    w_dn = np.ones(N, dtype=np.float64)
    if a_s_up != 0: w_up = w_up * np.power(speeds, a_s_up)
    if a_l_up != 0: w_up = w_up * np.power(lanes,  a_l_up)
    if a_s_dn != 0: w_dn = w_dn * np.power(speeds, a_s_dn)
    if a_l_dn != 0: w_dn = w_dn * np.power(lanes,  a_l_dn)
    a_mean = WORKER["a_mean"]
    P_up = build_P_weighted(N, WORKER["succ_idx"], WORKER["cos_th"], w_up, a_ang, a_mean)
    P_dn = build_P_weighted(N, WORKER["succ_idx"], WORKER["cos_th"], w_dn, a_ang, a_mean)
    cache[key] = (P_up, P_dn)
    if len(cache) > 24:
        for k in list(cache.keys())[:12]:
            del cache[k]
    return P_up, P_dn


def _eval_block(P_up, P_dn, inputs, targets):
    dmask = WORKER["dangling_mask"]
    ov_list = []
    for i in range(len(inputs)):
        v = tp_iter(P_up, P_dn, inputs[i], BETA_FIXED, RHO_FIXED, dmask)
        tgt = targets[i].astype(np.float64)
        rel = np.abs(tgt - v) / (tgt + EPS)
        ov_list.append(float(rel.mean()))
    return ov_list


def _loss(mode, x):
    P_up, P_dn = _kernels(mode, x)
    ov = _eval_block(P_up, P_dn, WORKER["train_inputs"], WORKER["train_targets"])
    return float(np.mean(ov)), ov


def worker_run(payload):
    mode, start_idx, x0 = payload
    ts = time.time()
    bounds = _bounds(mode, len(x0))
    res = minimize(lambda x: _loss(mode, x)[0], np.array(x0, dtype=np.float64),
                    method="L-BFGS-B", bounds=bounds,
                    options=dict(eps=2e-3, ftol=1e-6, gtol=1e-4, maxiter=20))
    f_opt, ov_train = _loss(mode, res.x)
    P_up, P_dn = _kernels(mode, res.x)
    ov_test = _eval_block(P_up, P_dn, WORKER["test_inputs"], WORKER["test_targets"])
    elapsed = time.time() - ts
    a_s_up, a_s_dn, a_l_up, a_l_dn, a_ang = _unpack(mode, res.x)
    print(f"  [{mode}] start {start_idx} done {elapsed:.0f}s  "
          f"train Ov.MRE={float(np.mean(ov_train)):.4f}  "
          f"test Ov.MRE={float(np.mean(ov_test)):.4f}  "
          f"a_s(u/d)={a_s_up:+.3f}/{a_s_dn:+.3f}  "
          f"a_l(u/d)={a_l_up:+.3f}/{a_l_dn:+.3f}  "
          f"a_ang={a_ang:+.3f}", flush=True)
    return dict(mode=mode, start=start_idx, x0=list(map(float, x0)),
                 x_opt=list(map(float, res.x)), f_opt=float(res.fun),
                 nfev=int(res.nfev), elapsed=elapsed,
                 ov_train=ov_train, ov_test=ov_test)


def main():
    t0 = time.time()
    max_workers = int(os.environ.get("TPPR_PARALLEL", "5"))
    print(f"[start] TP up/down origin->pop FULL ladder, "
          f"beta={BETA_FIXED}, rho={RHO_FIXED} fixed, delta={DELTA}, "
          f"a_ang{'>=0' if ANG_NONNEG else ' free'}, anorm={ANORM}, "
          f"mode={MODE}, max_workers={max_workers}", flush=True)

    if MODE not in MODE_STARTS:
        raise ValueError(f"TPPR_LADDER_MODE must be in {list(MODE_STARTS)}; "
                          f"got {MODE!r}")
    starts = MODE_STARTS[MODE]
    if not ANG_NONNEG and MODE in ("la_ang", "sp_ang", "sp_la_ang", "ang_only"):
        # explore the negative-angle basin too (the old optimum was < 0)
        neg = [list(s[:-1]) + [-2.0] for s in starts]
        starts = starts + neg
    if NSTARTS > 0:
        starts = starts[:NSTARTS]
    if START_IDX >= 0:
        starts = [starts[START_IDX]]
    print(f"  starts: {len(starts)}  {starts}  (start_idx={START_IDX})",
          flush=True)

    runs = []
    with ProcessPoolExecutor(max_workers=max_workers,
                              initializer=_worker_init) as ex:
        futs = {ex.submit(worker_run, (MODE, j, x0)): j
                 for j, x0 in enumerate(starts)}
        for fut in as_completed(futs):
            runs.append(fut.result())
    runs.sort(key=lambda r: r["start"])
    best = min(runs, key=lambda r: r["f_opt"])
    a_s_up, a_s_dn, a_l_up, a_l_dn, a_ang = _unpack(MODE, best["x_opt"])

    summary = dict(
        model="TP up/down origin->popularity FULL ladder (paper-faithful)",
        mode=MODE, beta_fixed=BETA_FIXED, rho_fixed=RHO_FIXED,
        delta=DELTA, angle_nonneg=ANG_NONNEG, angle_anorm=ANORM,
        laplace_alpha=LAPLACE_ALPHA, gamma=GAMMA,
        train_range=[TRAIN_START, TRAIN_END],
        test_range=[TEST_START, TEST_END],
        best=dict(x_opt=best["x_opt"], f_opt=best["f_opt"],
                   alpha_s_up=a_s_up, alpha_s_down=a_s_dn,
                   alpha_l_up=a_l_up, alpha_l_down=a_l_dn,
                   alpha_ang=a_ang),
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
    print(f"  best: a_s(u/d)={a_s_up:+.3f}/{a_s_dn:+.3f}  "
          f"a_l(u/d)={a_l_up:+.3f}/{a_l_dn:+.3f}  a_ang={a_ang:+.3f}  "
          f"TRAIN MRE={summary['train_overall_mre_mean']:.4f}  "
          f"TEST MRE={summary['test_overall_mre_mean']:.4f}")


if __name__ == "__main__":
    main()
