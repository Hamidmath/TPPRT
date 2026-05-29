"""Old-form TP tuning with per-OSM-category road-type weights.

Setup (locked):
  pairs   : (t, t+7d) for t = 2018-09-11 16:00, 16:05, ..., 16:55  (12 pairs)
  input   : diffused popularity at t   (Sep 11)
  target  : diffused popularity at t+7d (Sep 18, same clock time)
  loss    : Overall MRE restricted to the 7-link tight Fair sub-region.
  model   : old-form (classical PageRank) two-phase chain
            v_{k+1} = (1-alpha) M_{2N} v_k + alpha E_{2N}, E_{2N} = [E_b, 0]
            edge weight = exp(w_cat[category(link)]) * lanes^alpha_l
  params  : tune alpha, beta, alpha_l, w_0, w_1, w_2, w_3, w_4 (8 unknowns)
  bounds  : alpha, beta in [0.01, 0.15]
            alpha_l, w_0..w_4 unbounded
  search  : L-BFGS-B, 8 starts, eps=2e-3, ftol=1e-6, gtol=1e-4, maxiter=30

Categories from the MATSim/OSM converter's freespeed table:
  cat 0 : motorway/freeway (speed >= 26.82 m/s, ~60+ mph)
  cat 1 : arterial/primary (17.88 <= speed < 26.82, 40-55 mph)
  cat 2 : secondary/collector (13.41 <= speed < 17.88, 30-35 mph)
  cat 3 : residential (11.18 <= speed < 13.41, ~25 mph)
  cat 4 : local/service (speed < 11.18, <= 20 mph)

Output: results/tp_old_fair_tight_cats.json
"""
import json
import os
import sys
import time
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
OUT = RESULTS / "tp_old_fair_tight_cats.json"

EPS = 1e-6
TOL, MAX_ITER = 1e-6, 200
PAIRS = [(f"2018-09-11 16:{m:02d}:00", f"2018-09-18 16:{m:02d}:00")
          for m in range(0, 60, 5)]

REGION_LINK_IDS = [
    "59063","77400","77395","85822","77405","59055","3614",
]

CAT_BREAKS = [11.18, 13.41, 17.88, 26.82]  # 5 categories: 0..4
N_CATS = 5


def speed_to_category(s):
    """Return category index 0..4 from a speed value (m/s)."""
    if s >= CAT_BREAKS[3]:
        return 0  # motorway
    if s >= CAT_BREAKS[2]:
        return 1  # arterial
    if s >= CAT_BREAKS[1]:
        return 2  # secondary/collector
    if s >= CAT_BREAKS[0]:
        return 3  # residential
    return 4  # local


def build_P_row_weighted(graph, links, lid_to_idx, weights):
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
            rows.append(i); cols.append(j); data.append(float(norm[k]))
    return csr_matrix((data, (rows, cols)),
                       shape=(len(links), len(links)), dtype=np.float64)


def get_speed_lane_cat(graph, links):
    s, l, c = [], [], []
    for lid in links:
        ed = graph["links"].get(lid, {})
        sp = ed.get("speed", 11.17)
        s.append(sp)
        l.append(ed.get("lanes", 1.0))
        c.append(speed_to_category(sp))
    s = np.array(s, dtype=np.float64)
    l = np.array(l, dtype=np.float64)
    c = np.array(c, dtype=np.int64)
    if l.mean() > 0: l /= l.mean()
    return s, l, c


def tp_old_iter(P, E_b, alpha, beta):
    N = E_b.shape[0]
    Pt = P.T
    v_up = E_b.copy()
    v_down = np.zeros(N)
    d = 1.0 - alpha
    for _ in range(MAX_ITER):
        v_up_n   = d * (1.0 - beta) * (Pt @ v_up)   + alpha * E_b
        v_down_n = d * beta * v_up + d * (Pt @ v_down)
        s = float(v_up_n.sum() + v_down_n.sum())
        if s < 1.0:
            v_up_n += (1.0 - s) * E_b
        if float(np.abs(v_up_n - v_up).sum() + np.abs(v_down_n - v_down).sum()) < TOL:
            v_up, v_down = v_up_n, v_down_n
            break
        v_up, v_down = v_up_n, v_down_n
    return v_up + v_down


def load_diffused(npz_path, N, lid_to_idx):
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


def main():
    t0 = time.time()
    print("[start] OLD-form TP with categorical road-type weights "
          "(7-link tight Fair region, Sep 11 16:MM)")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes, cats = get_speed_lane_cat(graph, links)
    cat_counts = [int((cats == k).sum()) for k in range(N_CATS)]
    print(f"  N = {N:,}, links per category: {cat_counts}")

    region_lids = REGION_LINK_IDS
    region_idx = np.array([lid_to_idx[lid] for lid in region_lids
                            if lid in lid_to_idx], dtype=np.int64)
    print(f"  region: {len(region_lids)} link IDs, "
          f"{len(region_idx)} mapped to graph indices")

    rows = load_diffused(SMOOTH_NPZ, N, lid_to_idx)
    inputs, targets = [], []
    for t, sib in PAIRS:
        if t not in rows or sib not in rows:
            raise RuntimeError(f"missing timestamp in diffused matrix: {t} or {sib}")
        inputs.append(rows[t]); targets.append(rows[sib])
    print(f"  pairs: {len(inputs)}")

    kernel_cache = {}
    n_calls = [0]

    def get_kernel(a_l, w_cats):
        key = (round(a_l, 4),
                tuple(round(float(x), 3) for x in w_cats))
        if key in kernel_cache:
            return kernel_cache[key]
        cat_mul = np.exp(np.asarray(w_cats, dtype=np.float64))[cats]
        w = cat_mul.copy()
        if a_l != 0:
            w = w * np.power(lanes, a_l)
        P = build_P_row_weighted(graph, links, lid_to_idx, w)
        kernel_cache[key] = P
        if len(kernel_cache) > 96:
            for k in list(kernel_cache.keys())[:48]:
                del kernel_cache[k]
        return P

    def loss_full(alpha, beta, a_l, w_cats):
        P = get_kernel(a_l, w_cats)
        region_mre, full_ov = [], []
        for i in range(len(PAIRS)):
            v = tp_old_iter(P, inputs[i], alpha, beta)
            rel = np.abs(targets[i] - v) / (targets[i] + EPS)
            region_mre.append(float(rel[region_idx].mean()))
            full_ov.append(float(rel.mean()))
        n_calls[0] += 1
        if n_calls[0] % 5 == 0:
            print(f"    eval {n_calls[0]:4d}  a={alpha:.3f} b={beta:.3f} "
                  f"a_l={a_l:+.3f} w_cats={[round(x,2) for x in w_cats]}  "
                  f"region_MRE={np.mean(region_mre):.4f}  "
                  f"full_OV={np.mean(full_ov):.4f}",
                  flush=True)
        return float(np.mean(region_mre)), region_mre, full_ov

    def loss(x):
        alpha, beta, a_l = float(x[0]), float(x[1]), float(x[2])
        w_cats = [float(v) for v in x[3:8]]
        return loss_full(alpha, beta, a_l, w_cats)[0]

    BOUNDS = [(0.01, 0.15), (0.01, 0.15)] + [(None, None)] * 6
    STARTS = [
        # alpha, beta, alpha_l, w_motor, w_artery, w_secondary, w_residential, w_local
        [0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.15, 0.15, 0.5, 0.0, 1.0, 1.0, 0.0, -1.0],
        [0.05, 0.10, 0.2, -2.0, 0.5, 1.5, 0.5, 0.0],
        [0.10, 0.10, 1.0, 1.0, 0.0, -0.5, 0.0, 0.0],
        [0.15, 0.05, 0.5, 2.0, 1.0, 0.0, -1.0, -1.0],
        [0.05, 0.05, -0.5, -1.0, -1.0, 0.0, 1.0, 1.0],
        [0.10, 0.05, 0.3, 0.0, 0.5, 0.0, 0.5, -0.5],
        [0.05, 0.15, 0.5, 1.0, 1.5, 0.5, 0.0, -1.5],
    ]

    runs = []; best = None
    for j, x0 in enumerate(STARTS):
        ts = time.time()
        print(f"\n  [start {j}] x0 = {x0}", flush=True)
        res = minimize(loss, np.array(x0, dtype=np.float64),
                        method="L-BFGS-B", bounds=BOUNDS,
                        options=dict(eps=2e-3, ftol=1e-6, gtol=1e-4, maxiter=30))
        print(f"    done in {time.time()-ts:.0f}s nfev={res.nfev} "
              f"f={res.fun:.4f} x={[round(float(v),3) for v in res.x]}", flush=True)
        runs.append({"start": j, "x0": list(map(float, x0)),
                      "x_opt": list(map(float, res.x)),
                      "f_opt": float(res.fun), "nfev": int(res.nfev)})
        if best is None or res.fun < best["f_opt"]:
            best = runs[-1]

    a_opt, b_opt, al_opt = best["x_opt"][0], best["x_opt"][1], best["x_opt"][2]
    w_opt = best["x_opt"][3:8]
    f_opt, region_mre, full_ov = loss_full(a_opt, b_opt, al_opt, w_opt)
    summary = dict(
        model="old-form TP, classical PageRank style, categorical road weights",
        scope="Fair tight 7-link region, Sep 11 16:00-17:00",
        region_link_ids=region_lids,
        region_n_mapped=int(len(region_idx)),
        pairs=PAIRS,
        category_breaks_mps=CAT_BREAKS,
        category_labels=["motorway","arterial","secondary","residential","local"],
        bounds=dict(alpha=[0.01, 0.15], beta=[0.01, 0.15],
                     alpha_l=None, w_cats=None),
        starts=runs,
        best=best,
        best_params=dict(alpha=a_opt, beta=b_opt, alpha_l=al_opt,
                          w_motorway=w_opt[0], w_arterial=w_opt[1],
                          w_secondary=w_opt[2], w_residential=w_opt[3],
                          w_local=w_opt[4]),
        best_region_mre_mean=float(np.mean(region_mre)),
        best_region_mre_median=float(np.median(region_mre)),
        best_full_ov_mean=float(np.mean(full_ov)),
        per_pair_region_mre=region_mre,
        per_pair_full_ov=full_ov,
    )
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {time.time()-t0:.0f}s   saved {OUT}")
    print(f"  best: alpha={a_opt:.4f} beta={b_opt:.4f} alpha_l={al_opt:+.3f}")
    print(f"        w_cats = {[round(float(v),3) for v in w_opt]}")
    print(f"        (motorway, arterial, secondary, residential, local)")
    print(f"        region_MRE={float(np.mean(region_mre)):.4f}  "
          f"full_OV={float(np.mean(full_ov)):.4f}")


if __name__ == "__main__":
    main()
