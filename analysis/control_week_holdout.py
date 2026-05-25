"""Held-out forecast on Sept 15 with parameters trained
exclusively on non-event control week-pairs.

Training (parameters never see Sept 15):
  pair A: Sept 1 (Sat) -> Sept 8 (Sat), kickoff hour 19:00-19:55
  pair B: Sept 22 (Sat) -> Sept 29 (Sat), kickoff hour 19:00-19:55

Testing (frozen parameters):
  Sept 8 -> Sept 15, kickoff hour 19:00-19:55

For both single-phase (p, alpha_s, alpha_l) and two-phase
(alpha_s, alpha_l, beta, rho); top-10/20/30 clusters.
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
TOL, MAX_ITER = 1e-6, 200
EPS = 1e-6
WINDOW_START, WINDOW_END = "19:00:00", "19:55:00"
B_LO, B_HI = 0.01, 0.15

SP_DEFAULT = dict(p=0.0508, a_s=0.0, a_l=0.0)
TP_DEFAULT = dict(a_s=0.0, a_l=0.0, beta=0.102, rho=0.101)

K_LIST = [10, 20, 30]

TRAIN_PAIRS = [
    ("2018-09-01", "2018-09-08"),
    ("2018-09-22", "2018-09-29"),
]
TEST_PAIR = ("2018-09-08", "2018-09-15")


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


def collect_pair_bins(diff_rows, pairs):
    """Return list of (E_in, E_tg) pairs across all train pairs."""
    pairs_out = []
    for d_in, d_tg in pairs:
        bs_in = bins_window(d_in)
        bs_tg = bins_window(d_tg)
        for ti, tt in zip(bs_in, bs_tg):
            pairs_out.append((diff_rows[ti], diff_rows[tt]))
    return pairs_out


def d2d_avg(pairs_list, mask):
    per = []
    for E_in, E_tg in pairs_list:
        err = np.abs(E_tg - E_in) / (E_tg + EPS)
        per.append(float(err[mask].mean()))
    return float(np.mean(per))


def eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
            pairs_list, mask, p, a_s, a_l):
    weights = np.ones_like(speeds)
    if a_s != 0.0:
        weights = weights * np.power(speeds, a_s)
    if a_l != 0.0:
        weights = weights * np.power(lanes, a_l)
    P, dangling = build_sp_kernel(graph_data, links, lid_to_idx, weights)
    per = []
    for E_in, E_tg in pairs_list:
        v, _ = sp_iter_fixed(P, E_in, p, dangling, TOL, MAX_ITER)
        s = float(v.sum())
        if s > 0: v /= s
        err = np.abs(E_tg - v) / (E_tg + EPS)
        per.append(float(err[mask].mean()))
    return float(np.mean(per))


def eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
            pairs_list, mask, a_s, a_l, beta, rho):
    P_up, P_dn, dangling = build_tp_kernels(graph_data, links, lid_to_idx,
                                              speeds, lanes, a_s, a_l)
    per = []
    for E_in, E_tg in pairs_list:
        vu, vd, _ = tp_iter_fixed(P_up, P_dn, E_in, beta, rho, dangling,
                                   TOL, MAX_ITER)
        v = vu + vd
        s = float(v.sum())
        if s > 0: v /= s
        err = np.abs(E_tg - v) / (E_tg + EPS)
        per.append(float(err[mask].mean()))
    return float(np.mean(per))


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
                train_pairs, mask):
    cache = {}
    def f(x):
        p, a_s, a_l = float(x[0]), float(x[1]), float(x[2])
        key = (round(p, 6), round(a_s, 4), round(a_l, 4))
        if key in cache: return cache[key]
        val = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                       train_pairs, mask, p, a_s, a_l)
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
                train_pairs, mask):
    cache = {}
    def f(x):
        a_s, a_l, beta, rho = [float(z) for z in x]
        key = (round(a_s, 4), round(a_l, 4), round(beta, 6), round(rho, 6))
        if key in cache: return cache[key]
        val = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                       train_pairs, mask, a_s, a_l, beta, rho)
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

    train_pairs = collect_pair_bins(diff_rows, TRAIN_PAIRS)
    test_pairs = collect_pair_bins(diff_rows, [TEST_PAIR])
    print(f"  train pairs: {len(train_pairs)} bins "
          f"({TRAIN_PAIRS[0][0]}->{TRAIN_PAIRS[0][1]}, "
          f"{TRAIN_PAIRS[1][0]}->{TRAIN_PAIRS[1][1]})")
    print(f"  test pair:   {len(test_pairs)} bins "
          f"({TEST_PAIR[0]}->{TEST_PAIR[1]})")

    results = {"design": dict(
                    train_pairs=TRAIN_PAIRS,
                    test_pair=TEST_PAIR,
                    window=[WINDOW_START, WINDOW_END]),
               "defaults": dict(sp=SP_DEFAULT, tp=TP_DEFAULT),
               "clusters": {}}

    mask_dir = HERE / "results" / "event_sept15"
    for K in K_LIST:
        print(f"\n=== K = {K} ===", flush=True)
        mask = np.load(mask_dir / f"top{K}_busiest_3day_with_deadends_mask.npy")
        print(f"  mask size: {int(mask.sum())} links")

        d2d_train = d2d_avg(train_pairs, mask)
        d2d_test = d2d_avg(test_pairs, mask)
        sp_def_train = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                                 train_pairs, mask,
                                 SP_DEFAULT["p"], SP_DEFAULT["a_s"], SP_DEFAULT["a_l"])
        sp_def_test = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                                test_pairs, mask,
                                SP_DEFAULT["p"], SP_DEFAULT["a_s"], SP_DEFAULT["a_l"])
        tp_def_train = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                                 train_pairs, mask,
                                 TP_DEFAULT["a_s"], TP_DEFAULT["a_l"],
                                 TP_DEFAULT["beta"], TP_DEFAULT["rho"])
        tp_def_test = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                                test_pairs, mask,
                                TP_DEFAULT["a_s"], TP_DEFAULT["a_l"],
                                TP_DEFAULT["beta"], TP_DEFAULT["rho"])
        print(f"  d2d train={d2d_train:.4f}  d2d test={d2d_test:.4f}")
        print(f"  default SP train d2m={sp_def_train:.4f}  test d2m={sp_def_test:.4f}")
        print(f"  default TP train d2m={tp_def_train:.4f}  test d2m={tp_def_test:.4f}")

        print(f"  --- tune SP on control-week train pairs ---", flush=True)
        sp_best, sp_starts = optimize_sp(graph_data, links, lid_to_idx,
                                            speeds, lanes, train_pairs, mask)
        p_o, as_o, al_o = sp_best["x_opt"]
        sp_test = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                            test_pairs, mask, p_o, as_o, al_o)
        print(f"  SP frozen: x={sp_best['x_opt']}  train d2m="
              f"{sp_best['f_opt']:.4f}  TEST (Sept 15) d2m={sp_test:.4f}",
              flush=True)

        print(f"  --- tune TP on control-week train pairs ---", flush=True)
        tp_best, tp_starts = optimize_tp(graph_data, links, lid_to_idx,
                                            speeds, lanes, train_pairs, mask)
        as_o, al_o, be_o, ro_o = tp_best["x_opt"]
        tp_test = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                            test_pairs, mask, as_o, al_o, be_o, ro_o)
        print(f"  TP frozen: x={tp_best['x_opt']}  train d2m="
              f"{tp_best['f_opt']:.4f}  TEST (Sept 15) d2m={tp_test:.4f}",
              flush=True)

        results["clusters"][str(K)] = dict(
            d2d=dict(train=d2d_train, test=d2d_test),
            sp=dict(default_train=sp_def_train, default_test=sp_def_test,
                    tuned_x=sp_best["x_opt"],
                    tuned_train=sp_best["f_opt"],
                    tuned_test_sept15=sp_test,
                    starts=sp_starts),
            tp=dict(default_train=tp_def_train, default_test=tp_def_test,
                    tuned_x=tp_best["x_opt"],
                    tuned_train=tp_best["f_opt"],
                    tuned_test_sept15=tp_test,
                    starts=tp_starts),
        )

    out_path = HERE / "results" / "event_sept15" / "control_week_holdout.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out_path}  [done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
