"""Old-form TP tuning focused on the Fair-anomaly hour and region.

Setup (locked):
  pairs  : (t, t+7d) for t = 2018-09-11 16:00, 16:05, ..., 16:55  (12 pairs)
  input  : diffused popularity at t   (Sep 11)
  target : diffused popularity at t+7d (Sep 18, same clock time)
  loss   : Overall MRE restricted to the 39 link IDs in component 0 of
           documents/walkthrough2/results/anomalies/fair/region_components.json
           (the Fairpark-interchange region).
  model  : old-form (classical PageRank) two-phase chain
           v_{k+1} = (1-alpha) M_{2N} v_k + alpha E_{2N}, E_{2N} = [E_b, 0]
  params : tune alpha, beta, alpha_s, alpha_l (4 unknowns)
  bounds : alpha, beta in [0.01, 0.15]
           alpha_s, alpha_l unbounded
  search : L-BFGS-B, 5 random starts, eps=2e-3, ftol=1e-6, gtol=1e-4, maxiter=20

Output: results/tp_old_fair_region_hour.json
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
OUT = RESULTS / "tp_old_fair_region_hour.json"

EPS = 1e-6
TOL, MAX_ITER = 1e-6, 200
PAIRS = [(f"2018-09-11 16:{m:02d}:00", f"2018-09-18 16:{m:02d}:00")
          for m in range(0, 60, 5)]

# 39 OSM link IDs that form the largest connected component of the
# top-200 Fair-window lift links (see documents/walkthrough2/results/
# anomalies/fair/region_components.json, component 0).
REGION_LINK_IDS = [
    "59022","59042","59053","59038","59054","59016","59055","59236",
    "77395","58932","59063","58934","77400","58931","85822","59146",
    "77405","78455","3614","59147","59149","92097","77709","58976",
    "58975","58979","58969","77717","77714","58911","63362","65472",
    "56863","99566","58824","59309","59310","59312","64541",
]


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
    print("[start] OLD-form TP, Fair-anomaly region + hour, "
          "tune (alpha, beta, alpha_s, alpha_l)")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes = get_speed_lane(graph, links)
    print(f"  N = {N:,}")

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
    print(f"  pairs: {len(inputs)}  (each is 1 slot, Sep 11 16:MM -> Sep 18 16:MM)")
    print(f"  PAIRS  first: {PAIRS[0]}")
    print(f"  PAIRS  last : {PAIRS[-1]}")

    kernel_cache = {}
    n_calls = [0]

    def get_kernel(a_s, a_l):
        key = (round(a_s, 4), round(a_l, 4))
        if key in kernel_cache:
            return kernel_cache[key]
        w = np.ones(N, dtype=np.float64)
        if a_s != 0: w = w * np.power(speeds, a_s)
        if a_l != 0: w = w * np.power(lanes, a_l)
        P = build_P_row_weighted(graph, links, lid_to_idx, w)
        kernel_cache[key] = P
        if len(kernel_cache) > 96:
            for k in list(kernel_cache.keys())[:48]:
                del kernel_cache[k]
        return P

    def loss_full(alpha, beta, a_s, a_l):
        P = get_kernel(a_s, a_l)
        region_mre, full_ov = [], []
        for i in range(len(PAIRS)):
            v = tp_old_iter(P, inputs[i], alpha, beta)
            rel = np.abs(targets[i] - v) / (targets[i] + EPS)
            region_mre.append(float(rel[region_idx].mean()))
            full_ov.append(float(rel.mean()))
        n_calls[0] += 1
        if n_calls[0] % 5 == 0:
            print(f"    eval {n_calls[0]:4d}  a={alpha:.3f} b={beta:.3f} "
                  f"a_s={a_s:+.3f} a_l={a_l:+.3f}  "
                  f"region_MRE={np.mean(region_mre):.4f}  "
                  f"full_OV={np.mean(full_ov):.4f}",
                  flush=True)
        return float(np.mean(region_mre)), region_mre, full_ov

    def loss(x):
        return loss_full(float(x[0]), float(x[1]), float(x[2]), float(x[3]))[0]

    BOUNDS = [(0.01, 0.15), (0.01, 0.15), (None, None), (None, None)]
    STARTS = [
        [0.05, 0.05, 0.0, 0.0],
        [0.15, 0.15, 1.0, 0.5],
        [0.05, 0.10, 2.65, 0.17],
        [0.10, 0.10, -1.0, -1.0],
        [0.15, 0.05, 3.0, 1.0],
    ]

    runs = []; best = None
    for j, x0 in enumerate(STARTS):
        ts = time.time()
        print(f"\n  [start {j}] x0 = {x0}", flush=True)
        res = minimize(loss, np.array(x0, dtype=np.float64),
                        method="L-BFGS-B", bounds=BOUNDS,
                        options=dict(eps=2e-3, ftol=1e-6, gtol=1e-4, maxiter=20))
        print(f"    done in {time.time()-ts:.0f}s nfev={res.nfev} "
              f"f={res.fun:.4f} x={list(map(float, res.x))}", flush=True)
        runs.append({"start": j, "x0": list(map(float, x0)),
                      "x_opt": list(map(float, res.x)),
                      "f_opt": float(res.fun), "nfev": int(res.nfev)})
        if best is None or res.fun < best["f_opt"]:
            best = runs[-1]

    a_opt, b_opt, as_opt, al_opt = best["x_opt"]
    f_opt, region_mre, full_ov = loss_full(a_opt, b_opt, as_opt, al_opt)
    summary = dict(
        model="old-form TP, classical PageRank style",
        scope="Fair-anomaly region + hour",
        region_link_ids=region_lids,
        region_n_mapped=int(len(region_idx)),
        pairs=PAIRS,
        bounds=dict(alpha=[0.01, 0.15], beta=[0.01, 0.15],
                     alpha_s=None, alpha_l=None),
        starts=runs,
        best=best,
        best_params=dict(alpha=a_opt, beta=b_opt,
                          alpha_s=as_opt, alpha_l=al_opt),
        best_region_mre_mean=float(np.mean(region_mre)),
        best_region_mre_median=float(np.median(region_mre)),
        best_full_ov_mean=float(np.mean(full_ov)),
        per_pair_region_mre=region_mre,
        per_pair_full_ov=full_ov,
    )
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {time.time()-t0:.0f}s   saved {OUT}")
    print(f"  best: alpha={a_opt:.4f}  beta={b_opt:.4f}  "
          f"alpha_s={as_opt:+.3f}  alpha_l={al_opt:+.3f}")
    print(f"        region_MRE={float(np.mean(region_mre)):.4f}  "
          f"full_OV={float(np.mean(full_ov)):.4f}")


if __name__ == "__main__":
    main()
