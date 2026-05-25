"""Train on non-event Saturday-to-Saturday popularity pairs.
Test with Sept 15 trip-origins vector as input, frozen parameters,
score against Sept 15 popularity ground truth on top-K cluster.

Training pairs (kickoff hour 19:00-19:55, smoothed popularity):
  Sept 1  -> Sept 8
  Sept 22 -> Sept 29

Test (frozen parameters):
  input  = Sept 15 origins (diffused; raw also reported)
  target = Sept 15 popularity ground truth

For SP (p, alpha_s, alpha_l) and TP (alpha_s, alpha_l, beta, rho);
top-10/20/30 stadium clusters.
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

SP_DEFAULT = dict(p=0.0508, a_s=0.0, a_l=0.0)
TP_DEFAULT = dict(a_s=0.0, a_l=0.0, beta=0.102, rho=0.101)

K_LIST = [10, 20, 30]

TRAIN_PAIRS = [
    ("2018-09-01", "2018-09-08"),
    ("2018-09-22", "2018-09-29"),
]
TEST_PAIR = ("2018-09-15", "2018-09-15")  # input from origins, target from popularity


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


def build_diffusion_P(graph_data, links, lid_to_idx):
    """Row-stochastic uniform-over-successors kernel for the
    Laplace + one-step diffusion smoothing pipeline."""
    adj = graph_data.get("adjacency", {})
    N = len(links)
    row, col, data = [], [], []
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ: continue
        w = 1.0 / len(succ)
        for j in succ:
            row.append(i); col.append(j); data.append(w)
    return csr_matrix((data, (row, col)), shape=(N, N), dtype=np.float64)


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


def load_pop_rows(npz_path, N, lid_to_idx):
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


def diffuse_rows_inline(rows_dict, P_diff, N, gamma, alpha):
    """Apply Laplace alpha/N + one-step diffusion (1-gamma)c + gamma*Pc
    to a dict of per-bin probability rows; renormalize to sum 1."""
    out = {}
    for t, E in rows_dict.items():
        if E.sum() == 0:
            out[t] = E.copy()
            continue
        c = E + alpha / N
        c = (1.0 - gamma) * c + gamma * (P_diff @ c)
        s = c.sum()
        if s > 0: c = c / s
        out[t] = c
    return out


def bins_window(day_str):
    s = datetime.strptime(f"{day_str} {WINDOW_START}", "%Y-%m-%d %H:%M:%S")
    e = datetime.strptime(f"{day_str} {WINDOW_END}", "%Y-%m-%d %H:%M:%S")
    out, cur = [], s
    while cur <= e:
        out.append(cur.strftime("%Y-%m-%d %H:%M:%S"))
        cur += timedelta(minutes=5)
    return out


def collect_pairs(rows_in, rows_tg, day_pairs):
    """Return list of (E_in, E_tg) over the kickoff hour for each pair."""
    out = []
    for d_in, d_tg in day_pairs:
        bs_in = bins_window(d_in)
        bs_tg = bins_window(d_tg)
        for ti, tt in zip(bs_in, bs_tg):
            if ti in rows_in and tt in rows_tg:
                out.append((rows_in[ti], rows_tg[tt]))
    return out


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
    out = []
    for i, x0 in enumerate(SP_STARTS):
        ts = time.time()
        res = minimize(f, x0, method="L-BFGS-B", bounds=bounds,
                       options=dict(eps=1e-3, ftol=1e-7, gtol=1e-5, maxiter=80))
        out.append(dict(x0=list(map(float, x0)),
                          x_opt=list(map(float, res.x)),
                          f_opt=float(res.fun),
                          nfev=int(res.nfev),
                          elapsed=time.time() - ts))
        print(f"    SP start {i}: f={res.fun:.4f} x={res.x} t={time.time()-ts:.0f}s",
              flush=True)
    return min(out, key=lambda r: r["f_opt"]), out


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
    out = []
    for i, x0 in enumerate(TP_STARTS):
        ts = time.time()
        res = minimize(f, x0, method="L-BFGS-B", bounds=bounds,
                       options=dict(eps=1e-3, ftol=1e-7, gtol=1e-5, maxiter=80))
        out.append(dict(x0=list(map(float, x0)),
                          x_opt=list(map(float, res.x)),
                          f_opt=float(res.fun),
                          nfev=int(res.nfev),
                          elapsed=time.time() - ts))
        print(f"    TP start {i}: f={res.fun:.4f} x={res.x} t={time.time()-ts:.0f}s",
              flush=True)
    return min(out, key=lambda r: r["f_opt"]), out


def main():
    t0 = time.time()
    print("[load] graph")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links = list(graph_data["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes = get_speed_lane_arrays(graph_data, links)

    pop_path = config.DATA_DIR / "popularity_results_smoothed_osm_gamma020.npz"
    pop_rows = load_pop_rows(str(pop_path), N, lid_to_idx)
    print(f"  popularity rows: {len(pop_rows)}")

    orig_path = config.DATA_DIR / "origins_results.npz"
    orig_raw = load_pop_rows(str(orig_path), N, lid_to_idx)
    print(f"  origins rows (raw): {len(orig_raw)}")

    print("[diffuse] origins with Laplace alpha=0.01, gamma=0.20")
    P_diff = build_diffusion_P(graph_data, links, lid_to_idx)
    orig_diff = diffuse_rows_inline(orig_raw, P_diff, N, GAMMA, ALPHA_LAPLACE)

    # Training pairs: popularity in, popularity target.
    train_pairs = collect_pairs(pop_rows, pop_rows, TRAIN_PAIRS)
    bins_t = bins_window("2018-09-15")
    # Test pairs (event-aware): origins in for Sept 15, popularity target Sept 15.
    test_pairs_raw = [(orig_raw[t], pop_rows[t]) for t in bins_t if t in pop_rows]
    test_pairs_diff = [(orig_diff[t], pop_rows[t]) for t in bins_t if t in pop_rows]
    # Reference test pair (legacy d2m-tuned setup): pop Sept 8 in, pop Sept 15 target.
    bins_p = bins_window("2018-09-08")
    test_pairs_pop = [(pop_rows[ti], pop_rows[tt])
                       for ti, tt in zip(bins_p, bins_t)
                       if ti in pop_rows and tt in pop_rows]
    print(f"  train: {len(train_pairs)} pairs, "
          f"test(orig_raw): {len(test_pairs_raw)}, "
          f"test(orig_diff): {len(test_pairs_diff)}, "
          f"test(pop8): {len(test_pairs_pop)}")

    results = dict(design=dict(train_pairs=TRAIN_PAIRS,
                                  test_day="2018-09-15",
                                  window=[WINDOW_START, WINDOW_END],
                                  test_inputs=["pop_sept8", "orig_raw", "orig_diff"]),
                     defaults=dict(sp=SP_DEFAULT, tp=TP_DEFAULT),
                     clusters={})
    mask_dir = HERE / "results" / "event_sept15"

    for K in K_LIST:
        print(f"\n=== K = {K} ===", flush=True)
        mask = np.load(mask_dir / f"top{K}_busiest_3day_with_deadends_mask.npy")
        print(f"  mask size: {int(mask.sum())}")

        # Defaults on each test input
        sp_def_train = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                                 train_pairs, mask,
                                 SP_DEFAULT["p"], SP_DEFAULT["a_s"], SP_DEFAULT["a_l"])
        sp_def_pop = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                               test_pairs_pop, mask,
                               SP_DEFAULT["p"], SP_DEFAULT["a_s"], SP_DEFAULT["a_l"])
        sp_def_orig_raw = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                                    test_pairs_raw, mask,
                                    SP_DEFAULT["p"], SP_DEFAULT["a_s"], SP_DEFAULT["a_l"])
        sp_def_orig_diff = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                                     test_pairs_diff, mask,
                                     SP_DEFAULT["p"], SP_DEFAULT["a_s"], SP_DEFAULT["a_l"])
        tp_def_train = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                                 train_pairs, mask,
                                 TP_DEFAULT["a_s"], TP_DEFAULT["a_l"],
                                 TP_DEFAULT["beta"], TP_DEFAULT["rho"])
        tp_def_pop = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                               test_pairs_pop, mask,
                               TP_DEFAULT["a_s"], TP_DEFAULT["a_l"],
                               TP_DEFAULT["beta"], TP_DEFAULT["rho"])
        tp_def_orig_raw = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                                    test_pairs_raw, mask,
                                    TP_DEFAULT["a_s"], TP_DEFAULT["a_l"],
                                    TP_DEFAULT["beta"], TP_DEFAULT["rho"])
        tp_def_orig_diff = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                                     test_pairs_diff, mask,
                                     TP_DEFAULT["a_s"], TP_DEFAULT["a_l"],
                                     TP_DEFAULT["beta"], TP_DEFAULT["rho"])
        print(f"  default SP: train={sp_def_train:.4f}  test(pop8)={sp_def_pop:.4f}  "
              f"test(orig_raw)={sp_def_orig_raw:.4f}  test(orig_diff)={sp_def_orig_diff:.4f}")
        print(f"  default TP: train={tp_def_train:.4f}  test(pop8)={tp_def_pop:.4f}  "
              f"test(orig_raw)={tp_def_orig_raw:.4f}  test(orig_diff)={tp_def_orig_diff:.4f}")

        # Tune on train, evaluate on each test input.
        print(f"  --- tune SP on train (non-event) ---", flush=True)
        sp_best, sp_starts = optimize_sp(graph_data, links, lid_to_idx,
                                            speeds, lanes, train_pairs, mask)
        p_o, as_o, al_o = sp_best["x_opt"]
        sp_test_pop = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                                test_pairs_pop, mask, p_o, as_o, al_o)
        sp_test_raw = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                                test_pairs_raw, mask, p_o, as_o, al_o)
        sp_test_diff = eval_sp(graph_data, links, lid_to_idx, speeds, lanes,
                                 test_pairs_diff, mask, p_o, as_o, al_o)
        print(f"  SP frozen x={sp_best['x_opt']}  train={sp_best['f_opt']:.4f}  "
              f"test(pop8)={sp_test_pop:.4f}  test(orig_raw)={sp_test_raw:.4f}  "
              f"test(orig_diff)={sp_test_diff:.4f}", flush=True)

        print(f"  --- tune TP on train (non-event) ---", flush=True)
        tp_best, tp_starts = optimize_tp(graph_data, links, lid_to_idx,
                                            speeds, lanes, train_pairs, mask)
        as_o, al_o, be_o, ro_o = tp_best["x_opt"]
        tp_test_pop = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                                test_pairs_pop, mask, as_o, al_o, be_o, ro_o)
        tp_test_raw = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                                test_pairs_raw, mask, as_o, al_o, be_o, ro_o)
        tp_test_diff = eval_tp(graph_data, links, lid_to_idx, speeds, lanes,
                                 test_pairs_diff, mask, as_o, al_o, be_o, ro_o)
        print(f"  TP frozen x={tp_best['x_opt']}  train={tp_best['f_opt']:.4f}  "
              f"test(pop8)={tp_test_pop:.4f}  test(orig_raw)={tp_test_raw:.4f}  "
              f"test(orig_diff)={tp_test_diff:.4f}", flush=True)

        results["clusters"][str(K)] = dict(
            sp=dict(default_train=sp_def_train,
                    default_test_pop8=sp_def_pop,
                    default_test_orig_raw=sp_def_orig_raw,
                    default_test_orig_diff=sp_def_orig_diff,
                    tuned_x=sp_best["x_opt"],
                    tuned_train=sp_best["f_opt"],
                    tuned_test_pop8=sp_test_pop,
                    tuned_test_orig_raw=sp_test_raw,
                    tuned_test_orig_diff=sp_test_diff,
                    starts=sp_starts),
            tp=dict(default_train=tp_def_train,
                    default_test_pop8=tp_def_pop,
                    default_test_orig_raw=tp_def_orig_raw,
                    default_test_orig_diff=tp_def_orig_diff,
                    tuned_x=tp_best["x_opt"],
                    tuned_train=tp_best["f_opt"],
                    tuned_test_pop8=tp_test_pop,
                    tuned_test_orig_raw=tp_test_raw,
                    tuned_test_orig_diff=tp_test_diff,
                    starts=tp_starts),
        )

    out = HERE / "results" / "event_sept15" / "event_aware_test.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out}  [done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
