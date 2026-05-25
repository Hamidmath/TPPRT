"""6/6 train/test holdout split on the Sept 15 event window.

Train: first 6 bins (19:00-19:25).
Test:  last 6 bins  (19:30-19:55).

For both single-phase and two-phase chains, on top-10/20/30:
  - Optimize parameters on the train bins.
  - Report in-sample (train) and out-of-sample (test) d2m.
  - Also report default-parameter d2m on the test bins.
"""
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import config
from core.io import load_popularity_npz

ALPHA_LAPLACE = 0.01
GAMMA = 0.20
TOL, MAX_ITER = 1e-6, 200
EPS = 1e-6
WINDOW_START, WINDOW_END = "19:00:00", "19:55:00"
B_LO, B_HI = 0.01, 0.15

# defaults from main.tex
SP_DEFAULT = dict(p=0.0508, a_s=0.0, a_l=0.0)
TP_DEFAULT = dict(a_s=0.0, a_l=0.0, beta=0.102, rho=0.101)

K_LIST = [10, 20, 30]
N_TRAIN = 6   # first 6 bins are train; last 6 are test


def get_speed_lane_arrays(graph_data, links):
    speeds, lanes = [], []
    for lid in links:
        ed = graph_data["links"].get(lid, {})
        speeds.append(ed.get("speed", 11.17))
        lanes.append(ed.get("lanes", 1.0))
    s = np.array(speeds, dtype=np.float64)
    l = np.array(lanes, dtype=np.float64)
    if s.mean() > 0: s /= s.mean()
    if l.mean() > 0: l /= l.mean()
    return s, l


def build_sp_kernel(graph_data, links, lid_to_idx, weights):
    adj = graph_data.get("adjacency", {})
    rows, cols, data = [], [], []
    N = len(links)
    dangling = np.zeros(N, dtype=bool)
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ:
            dangling[i] = True
            continue
        out_w = np.array([weights[j] for j in succ], dtype=np.float64)
        total = out_w.sum()
        if total <= 0:
            dangling[i] = True
            continue
        for k, j in enumerate(succ):
            rows.append(j); cols.append(i); data.append(out_w[k] / total)
    return csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64), dangling


def build_tp_kernels(graph_data, links, lid_to_idx, speeds, lanes, a_s, a_l):
    w_up = np.ones_like(speeds); w_dn = np.ones_like(speeds)
    if a_s != 0.0:
        w_up = w_up * np.power(speeds, a_s); w_dn = w_dn * np.power(speeds, -a_s)
    if a_l != 0.0:
        w_up = w_up * np.power(lanes, a_l); w_dn = w_dn * np.power(lanes, -a_l)
    P_up, dangling = build_sp_kernel(graph_data, links, lid_to_idx, w_up)
    P_dn, _ = build_sp_kernel(graph_data, links, lid_to_idx, w_dn)
    return P_up, P_dn, dangling


def sp_iter_fixed(P, E_b, p, dangling, tol, max_iter):
    N = E_b.shape[0]
    v = E_b.copy()
    for k in range(max_iter):
        leak = float(v[dangling].sum())
        v_new = p * E_b + (1.0 - p) * (P @ v + leak * E_b)
        diff = float(np.abs(v_new - v).sum())
        v = v_new
        if diff < tol: return v, k + 1
    return v, max_iter


def tp_iter_fixed(P_up, P_dn, E_b, beta, rho, dangling, tol, max_iter):
    N = E_b.shape[0]
    v_up = E_b.copy()
    v_dn = np.zeros(N, dtype=np.float64)
    for k in range(max_iter):
        s_dn = float(v_dn.sum())
        leak_up = float(v_up[dangling].sum())
        leak_dn = float(v_dn[dangling].sum())
        v_up_new = (1.0 - beta) * (P_up @ v_up + leak_up * E_b) + rho * E_b * s_dn
        v_dn_new = beta * v_up + (1.0 - rho) * (P_dn @ v_dn + leak_dn * E_b)
        diff = float(np.abs(v_up_new - v_up).sum() + np.abs(v_dn_new - v_dn).sum())
        v_up, v_dn = v_up_new, v_dn_new
        if diff < tol: return v_up, v_dn, k + 1
    return v_up, v_dn, max_iter


def load_prior(npz_path, N, lid_to_idx):
    b = load_popularity_npz(npz_path)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    src_ids = [str(lid) for lid in b["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in src_ids], dtype=np.int64)
    valid = proj >= 0
    rows = {}
    if hasattr(M, "toarray") and not hasattr(M, "getrow"):
        # plain ndarray fallback (shouldn't trigger with _DenseCSR)
        for ti, t in enumerate(times):
            row = M[ti].astype(np.float64)
            E = np.zeros(N, dtype=np.float64)
            np.add.at(E, proj[valid], row[valid])
            s = E.sum()
            if s > 0: E /= s
            rows[t] = E
    else:
        for ti, t in enumerate(times):
            row = M.getrow(ti).toarray().ravel().astype(np.float64)
            E = np.zeros(N, dtype=np.float64)
            np.add.at(E, proj[valid], row[valid])
            s = E.sum()
            if s > 0: E /= s
            rows[t] = E
    return rows


def bins_window(day_str):
    s = datetime.strptime(f"{day_str} {WINDOW_START}", "%Y-%m-%d %H:%M:%S")
    e = datetime.strptime(f"{day_str} {WINDOW_END}", "%Y-%m-%d %H:%M:%S")
    out, cur = [], s
    while cur <= e:
        out.append(cur.strftime("%Y-%m-%d %H:%M:%S"))
        cur += timedelta(minutes=5)
    return out


def eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
            Es_in, Es_tg, mask, p, a_s, a_l):
    weights = np.ones_like(speeds)
    if a_s != 0.0:
        weights = weights * np.power(speeds, a_s)
    if a_l != 0.0:
        weights = weights * np.power(lanes, a_l)
    P, dangling = build_sp_kernel(graph_data, links, lid_to_idx, weights)
    per = []
    for E_in, E_tg in zip(Es_in, Es_tg):
        v, _ = sp_iter_fixed(P, E_in, p, dangling, TOL, MAX_ITER)
        s = float(v.sum())
        if s > 0: v /= s
        err = np.abs(E_tg - v) / (E_tg + EPS)
        per.append(float(err[mask].mean()))
    return float(np.mean(per)), per


def eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
            Es_in, Es_tg, mask, a_s, a_l, beta, rho):
    P_up, P_dn, dangling = build_tp_kernels(graph_data, links, lid_to_idx,
                                              speeds, lanes, a_s, a_l)
    per = []
    for E_in, E_tg in zip(Es_in, Es_tg):
        vu, vd, _ = tp_iter_fixed(P_up, P_dn, E_in, beta, rho, dangling,
                                   TOL, MAX_ITER)
        v = vu + vd
        s = float(v.sum())
        if s > 0: v /= s
        err = np.abs(E_tg - v) / (E_tg + EPS)
        per.append(float(err[mask].mean()))
    return float(np.mean(per)), per


SP_STARTS = [
    np.array([0.05, 0.0, 0.0]),
    np.array([0.01, -50.0, -10.0]),
    np.array([0.10, +20.0, -5.0]),
    np.array([0.01, -300.0, -15.0]),
    np.array([0.05, +5.0, +5.0]),
]

TP_STARTS = [
    np.array([0.0, 0.0, 0.10, 0.10]),
    np.array([-50.0, -10.0, 0.010, 0.115]),
    np.array([+17.0, -10.0, 0.150, 0.010]),
    np.array([-20.0, -2.0, 0.010, 0.090]),
    np.array([+3.0, -5.0, 0.090, 0.010]),
]


def optimize_sp(graph_data, links, lid_to_idx, speeds, lanes,
                Es_in_tr, Es_tg_tr, mask):
    cache = {}
    def f(x):
        p, a_s, a_l = float(x[0]), float(x[1]), float(x[2])
        key = (round(p, 6), round(a_s, 4), round(a_l, 4))
        if key in cache: return cache[key]
        val, _ = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                          Es_in_tr, Es_tg_tr, mask, p, a_s, a_l)
        cache[key] = val
        return val
    bounds = [(B_LO, B_HI), (None, None), (None, None)]
    starts_results = []
    for i, x0 in enumerate(SP_STARTS):
        ts = time.time()
        res = minimize(f, x0, method="L-BFGS-B", bounds=bounds,
                       options=dict(eps=1e-3, ftol=1e-7, gtol=1e-5, maxiter=80))
        starts_results.append(dict(x0=list(map(float, x0)),
                                    x_opt=list(map(float, res.x)),
                                    f_opt=float(res.fun),
                                    nfev=int(res.nfev),
                                    elapsed=time.time() - ts))
        print(f"    SP start {i}: f={res.fun:.4f} x={res.x} t={time.time()-ts:.0f}s",
              flush=True)
    best = min(starts_results, key=lambda r: r["f_opt"])
    return best, starts_results


def optimize_tp(graph_data, links, lid_to_idx, speeds, lanes,
                Es_in_tr, Es_tg_tr, mask):
    cache = {}
    def f(x):
        a_s, a_l, beta, rho = [float(z) for z in x]
        key = (round(a_s, 4), round(a_l, 4), round(beta, 6), round(rho, 6))
        if key in cache: return cache[key]
        val, _ = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                          Es_in_tr, Es_tg_tr, mask, a_s, a_l, beta, rho)
        cache[key] = val
        return val
    bounds = [(None, None), (None, None), (B_LO, B_HI), (B_LO, B_HI)]
    starts_results = []
    for i, x0 in enumerate(TP_STARTS):
        ts = time.time()
        res = minimize(f, x0, method="L-BFGS-B", bounds=bounds,
                       options=dict(eps=1e-3, ftol=1e-7, gtol=1e-5, maxiter=80))
        starts_results.append(dict(x0=list(map(float, x0)),
                                    x_opt=list(map(float, res.x)),
                                    f_opt=float(res.fun),
                                    nfev=int(res.nfev),
                                    elapsed=time.time() - ts))
        print(f"    TP start {i}: f={res.fun:.4f} x={res.x} t={time.time()-ts:.0f}s",
              flush=True)
    best = min(starts_results, key=lambda r: r["f_opt"])
    return best, starts_results


def main():
    t0 = time.time()
    print(f"[load] graph + diffused popularity")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links = list(graph_data["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes = get_speed_lane_arrays(graph_data, links)

    diff_path = config.DATA_DIR / "popularity_results_smoothed_osm_gamma020.npz"
    diff_rows = load_prior(str(diff_path), N, lid_to_idx)
    print(f"  {len(diff_rows)} bins loaded; N={N}")

    bins_p = bins_window("2018-09-08")
    bins_t = bins_window("2018-09-15")
    assert len(bins_p) == 12 and len(bins_t) == 12, (len(bins_p), len(bins_t))

    Es_p_all = [diff_rows[t] for t in bins_p]
    Es_t_all = [diff_rows[t] for t in bins_t]

    Es_p_tr, Es_p_te = Es_p_all[:N_TRAIN], Es_p_all[N_TRAIN:]
    Es_t_tr, Es_t_te = Es_t_all[:N_TRAIN], Es_t_all[N_TRAIN:]
    print(f"  train bins: {bins_t[0]}..{bins_t[N_TRAIN-1]}")
    print(f"  test  bins: {bins_t[N_TRAIN]}..{bins_t[-1]}")

    results = {"split": dict(train_bins=bins_t[:N_TRAIN],
                              test_bins=bins_t[N_TRAIN:]),
               "defaults": dict(sp=SP_DEFAULT, tp=TP_DEFAULT),
               "clusters": {}}

    mask_dir = HERE / "results" / "event_sept15"

    for K in K_LIST:
        print(f"\n=== K = {K} ===", flush=True)
        mask = np.load(mask_dir / f"top{K}_busiest_3day_with_deadends_mask.npy")
        print(f"  mask size: {int(mask.sum())} links")

        # Default-parameter d2m on test (no tuning at all)
        sp_def_te, _ = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                                 Es_p_te, Es_t_te, mask,
                                 SP_DEFAULT["p"], SP_DEFAULT["a_s"], SP_DEFAULT["a_l"])
        tp_def_te, _ = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                                 Es_p_te, Es_t_te, mask,
                                 TP_DEFAULT["a_s"], TP_DEFAULT["a_l"],
                                 TP_DEFAULT["beta"], TP_DEFAULT["rho"])
        sp_def_tr, _ = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                                 Es_p_tr, Es_t_tr, mask,
                                 SP_DEFAULT["p"], SP_DEFAULT["a_s"], SP_DEFAULT["a_l"])
        tp_def_tr, _ = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                                 Es_p_tr, Es_t_tr, mask,
                                 TP_DEFAULT["a_s"], TP_DEFAULT["a_l"],
                                 TP_DEFAULT["beta"], TP_DEFAULT["rho"])
        print(f"  default SP: train d2m={sp_def_tr:.4f}  test d2m={sp_def_te:.4f}")
        print(f"  default TP: train d2m={tp_def_tr:.4f}  test d2m={tp_def_te:.4f}")

        # Tune SP on train
        print(f"  --- tune SP on train ---", flush=True)
        sp_best, sp_starts = optimize_sp(graph_data, links, lid_to_idx,
                                            speeds, lanes,
                                            Es_p_tr, Es_t_tr, mask)
        p_o, as_o, al_o = sp_best["x_opt"]
        sp_tuned_te, _ = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                                   Es_p_te, Es_t_te, mask, p_o, as_o, al_o)
        print(f"  SP tuned: x={sp_best['x_opt']}  in-sample(train)="
              f"{sp_best['f_opt']:.4f}  held-out(test)={sp_tuned_te:.4f}",
              flush=True)

        # Tune TP on train
        print(f"  --- tune TP on train ---", flush=True)
        tp_best, tp_starts = optimize_tp(graph_data, links, lid_to_idx,
                                            speeds, lanes,
                                            Es_p_tr, Es_t_tr, mask)
        as_o, al_o, be_o, ro_o = tp_best["x_opt"]
        tp_tuned_te, _ = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                                   Es_p_te, Es_t_te, mask, as_o, al_o, be_o, ro_o)
        print(f"  TP tuned: x={tp_best['x_opt']}  in-sample(train)="
              f"{tp_best['f_opt']:.4f}  held-out(test)={tp_tuned_te:.4f}",
              flush=True)

        results["clusters"][str(K)] = dict(
            sp=dict(default_train=sp_def_tr, default_test=sp_def_te,
                    tuned_x=sp_best["x_opt"],
                    tuned_train_insample=sp_best["f_opt"],
                    tuned_test_holdout=sp_tuned_te,
                    starts=sp_starts),
            tp=dict(default_train=tp_def_tr, default_test=tp_def_te,
                    tuned_x=tp_best["x_opt"],
                    tuned_train_insample=tp_best["f_opt"],
                    tuned_test_holdout=tp_tuned_te,
                    starts=tp_starts),
        )

    out_path = HERE / "results" / "event_sept15" / "holdout_split.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out_path}  [done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
