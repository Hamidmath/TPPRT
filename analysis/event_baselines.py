"""Held-out event-forecast baselines on Sept 15.

Compared against the frozen two-phase / single-phase chain
results from event_aware_test, this script computes simpler
predictors (no learning, or no chain at all) on the same
Sept 15 19:00-19:55 window and the same top-K stadium cluster.

Predictors evaluated, all scored against Sept 15 popularity:
  1. Identity using Sept 8 popularity ........... d2d (no chain)
  2. Identity using Sept 15 raw origins .......... origins-only
  3. Identity using Sept 15 diffused origins ..... diffused origins
  4. Uniform 1/|cluster| .......................... uniform
  5. Training-Saturday-mean popularity at the
     kickoff hour ................................ climatology
  6. One-step random walk on diffused origins .... P @ origins
  7. Two-phase chain WITHOUT dangling fix
     (legacy "v1" formulation), default params,
     diffused origins as input ................... v1 (no dangling)
  8. Two-phase chain WITHOUT diffusion smoothing
     of the input (raw counts + Laplace only),
     dangling fix on, default params ............. no-diffusion TP
"""
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import config
from core.io import load_popularity_npz

ALPHA_LAPLACE = 0.01
GAMMA = 0.20
TOL, MAX_ITER = 1e-6, 200
EPS = 1e-6
WINDOW_START, WINDOW_END = "19:00:00", "19:55:00"

SP_DEFAULT = dict(p=0.0508, a_s=0.0, a_l=0.0)
TP_DEFAULT = dict(a_s=0.0, a_l=0.0, beta=0.102, rho=0.101)
K_LIST = [10, 20, 30]

TRAIN_DAYS = ["2018-09-01", "2018-09-08", "2018-09-22", "2018-09-29"]


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


def tp_iter_fixed(P_up, P_dn, E_b, beta, rho, dangling, tol, max_iter,
                   use_dangling=True):
    """Two-phase chain. If use_dangling=False, this reproduces the
    legacy 'v1' chain that ignored dangling links (the version we
    used much earlier that did not seem to work well)."""
    N = E_b.shape[0]
    v_up = E_b.copy()
    v_dn = np.zeros(N, dtype=np.float64)
    for k in range(max_iter):
        s_dn = float(v_dn.sum())
        if use_dangling:
            leak_up = float(v_up[dangling].sum())
            leak_dn = float(v_dn[dangling].sum())
            v_up_new = (1.0 - beta) * (P_up @ v_up + leak_up * E_b) + rho * E_b * s_dn
            v_dn_new = beta * v_up + (1.0 - rho) * (P_dn @ v_dn + leak_dn * E_b)
        else:
            # v1: no dangling correction. Mass quietly leaks out.
            v_up_new = (1.0 - beta) * (P_up @ v_up) + rho * E_b * s_dn
            v_dn_new = beta * v_up + (1.0 - rho) * (P_dn @ v_dn)
        diff = float(np.abs(v_up_new - v_up).sum() + np.abs(v_dn_new - v_dn).sum())
        v_up, v_dn = v_up_new, v_dn_new
        if diff < tol: return v_up, v_dn, k + 1
    return v_up, v_dn, max_iter


def load_pop_rows(npz_path, N, lid_to_idx, keep_times=None):
    """Load only the bins listed in keep_times (set of timestamp strings)
    if given; otherwise load everything."""
    b = load_popularity_npz(npz_path)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    src_ids = [str(lid) for lid in b["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in src_ids], dtype=np.int64)
    valid = proj >= 0
    rows = {}
    if keep_times is not None:
        keep_times = set(keep_times)
    for ti, t in enumerate(times):
        if keep_times is not None and t not in keep_times:
            continue
        row = M.getrow(ti).toarray().ravel().astype(np.float64)
        E = np.zeros(N, dtype=np.float64)
        np.add.at(E, proj[valid], row[valid])
        s = E.sum()
        if s > 0: E /= s
        rows[t] = E
    return rows


def diffuse_rows(rows_dict, P_diff, N, gamma, alpha):
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


def laplace_only_rows(rows_dict, N, alpha):
    """Laplace smoothing without the graph-diffusion step."""
    out = {}
    for t, E in rows_dict.items():
        if E.sum() == 0:
            out[t] = E.copy()
            continue
        c = E + alpha / N
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


def mre_on_mask(truth, pred, mask):
    err = np.abs(truth - pred) / (truth + EPS)
    return float(err[mask].mean())


def main():
    t0 = time.time()
    print("[load] graph")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links = list(graph_data["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes = get_speed_lane_arrays(graph_data, links)

    # only keep the bins we actually use: Sept 1, 8, 15, 22, 29 kickoff hour
    needed_times = set()
    for d in TRAIN_DAYS + ["2018-09-15"]:
        needed_times.update(bins_window(d))

    pop_path = config.DATA_DIR / "popularity_results_smoothed_osm_gamma020.npz"
    pop_rows = load_pop_rows(str(pop_path), N, lid_to_idx, keep_times=needed_times)

    # raw (non-diffused) popularity for the "no-diffusion" baseline
    raw_path = config.DATA_DIR / "popularity_results_osm.npz"
    if not raw_path.exists():
        raw_path = config.DATA_DIR / "popularity_results.npz"
    raw_pop_rows = load_pop_rows(str(raw_path), N, lid_to_idx, keep_times=needed_times)
    raw_pop_lap = laplace_only_rows(raw_pop_rows, N, ALPHA_LAPLACE)

    orig_path = config.DATA_DIR / "origins_results.npz"
    orig_raw = load_pop_rows(str(orig_path), N, lid_to_idx, keep_times=needed_times)

    print("[diffuse] origins")
    P_diff = build_diffusion_P(graph_data, links, lid_to_idx)
    orig_diff = diffuse_rows(orig_raw, P_diff, N, GAMMA, ALPHA_LAPLACE)

    bins_t = bins_window("2018-09-15")
    bins_p = bins_window("2018-09-08")

    # Training-Saturday-mean popularity at the kickoff hour
    # (mean over Sept 1, 8, 22, 29 -- non-event Saturdays
    #  including Sept 8 since the d2d uses Sept 8 directly, this
    #  climatology is a separate estimate).
    train_bins_per_day = [bins_window(d) for d in TRAIN_DAYS]
    climatology = []
    for b_idx in range(12):
        accum = np.zeros(N, dtype=np.float64)
        count = 0
        for tb in train_bins_per_day:
            t = tb[b_idx]
            if t in pop_rows:
                accum += pop_rows[t]
                count += 1
        if count > 0: accum /= count
        s = accum.sum()
        if s > 0: accum /= s
        climatology.append(accum)

    truth_rows = [pop_rows[t] for t in bins_t]
    sept8_rows = [pop_rows[t] for t in bins_p if t in pop_rows]
    origin_raw_rows = [orig_raw[t] for t in bins_t]
    origin_diff_rows = [orig_diff[t] for t in bins_t]
    raw_pop_sept15 = [raw_pop_lap[t] for t in bins_t]

    # one-step diffusion of the origins prior (P @ orig_diff)
    one_step_rows = []
    for E in origin_diff_rows:
        v = P_diff @ E
        s = v.sum()
        if s > 0: v /= s
        one_step_rows.append(v)

    # v1 chain (no dangling fix) with TP defaults, diffused origins
    print("[chain] v1 (no dangling) TP-default on diffused origins")
    P_up, P_dn, dangling = build_tp_kernels(graph_data, links, lid_to_idx,
                                              speeds, lanes,
                                              TP_DEFAULT["a_s"], TP_DEFAULT["a_l"])
    v1_rows = []
    for E in origin_diff_rows:
        vu, vd, _ = tp_iter_fixed(P_up, P_dn, E,
                                   TP_DEFAULT["beta"], TP_DEFAULT["rho"],
                                   dangling, TOL, MAX_ITER, use_dangling=False)
        v = vu + vd
        s = v.sum()
        if s > 0: v /= s
        v1_rows.append(v)

    # TP-default chain WITHOUT diffusion smoothing on the prior
    # (Laplace-only origins).
    orig_lap_only = laplace_only_rows(orig_raw, N, ALPHA_LAPLACE)
    orig_lap_only_rows = [orig_lap_only[t] for t in bins_t]
    print("[chain] TP-default (with dangling) on Laplace-only origins")
    no_diff_rows = []
    for E in orig_lap_only_rows:
        vu, vd, _ = tp_iter_fixed(P_up, P_dn, E,
                                   TP_DEFAULT["beta"], TP_DEFAULT["rho"],
                                   dangling, TOL, MAX_ITER, use_dangling=True)
        v = vu + vd
        s = v.sum()
        if s > 0: v /= s
        no_diff_rows.append(v)

    results = {"design": dict(test_day="2018-09-15",
                               window=[WINDOW_START, WINDOW_END]),
                "clusters": {}}

    mask_dir = HERE / "results" / "event_sept15"
    for K in K_LIST:
        mask = np.load(mask_dir / f"top{K}_busiest_3day_with_deadends_mask.npy")
        nm = int(mask.sum())
        print(f"\n=== K = {K}  (cluster size {nm}) ===", flush=True)

        def score(pred_rows):
            per = [mre_on_mask(t, p, mask) for t, p in zip(truth_rows, pred_rows)]
            return float(np.mean(per)), per

        uniform_pred = np.ones(N, dtype=np.float64) / N
        uniform_rows = [uniform_pred for _ in bins_t]

        rows_dict = {
            "d2d_sept8":         sept8_rows,
            "origins_raw":       origin_raw_rows,
            "origins_diffused":  origin_diff_rows,
            "uniform":           uniform_rows,
            "climatology_train": climatology,
            "P_one_step_origins": one_step_rows,
            "v1_tp_no_dangling": v1_rows,
            "tp_no_diffusion":   no_diff_rows,
            "raw_sept15_pop":    raw_pop_sept15,
        }
        cluster_out = {}
        for name, pred_rows in rows_dict.items():
            mean_mre, per_bin = score(pred_rows)
            print(f"  {name:25s} d2m={mean_mre:.4f}")
            cluster_out[name] = dict(d2m=mean_mre, per_bin=per_bin)
        results["clusters"][str(K)] = cluster_out

    out_path = HERE / "results" / "event_sept15" / "event_baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out_path}  [done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
