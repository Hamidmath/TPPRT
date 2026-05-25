"""E3: detailed slot-by-slot dump for four target days.

Days picked:
  2018-09-03 (Labor Day, Monday) — federal holiday
  2018-09-05 (Wed) — normal weekday
  2018-09-15 (Sat) — Utah vs Washington football game
  2018-09-22 (Sat) — normal Saturday

For each day, every 30-min slot (48 of them), report:
  chain output v_t (top 50 link IDs + scores)
  next-slot E_{t+1} (top 50)
  d2m_chain / d2m_d2d (top-100 and all-active)
  Pearson r between v_t and E_{t+1}

Output goes to a JSON for the experiments log.
"""
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import config
from core.io import load_popularity_npz

ALPHA_LAPLACE = 0.01
GAMMA = 0.20
TOL, MAX_ITER = 1e-6, 200
EPS = 1e-6
BETA = 0.102
RHO = 0.101
BIN_MIN = 30
BINS_PER_SLOT = BIN_MIN // 5
TOP_K = 100
TARGET_DAYS = ["2018-09-03", "2018-09-05", "2018-09-15", "2018-09-22"]
LABELS = {
    "2018-09-03": "Labor Day (Mon)",
    "2018-09-05": "Normal Wed",
    "2018-09-15": "Game Sat",
    "2018-09-22": "Normal Sat",
}


def build_sp_kernel(graph_data, links, lid_to_idx, weights):
    adj = graph_data.get("adjacency", {})
    rows, cols, data = [], [], []
    N = len(links)
    dangling = np.zeros(N, dtype=bool)
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ:
            dangling[i] = True; continue
        out_w = np.array([weights[j] for j in succ], dtype=np.float64)
        total = out_w.sum()
        if total <= 0:
            dangling[i] = True; continue
        for k, j in enumerate(succ):
            rows.append(j); cols.append(i); data.append(out_w[k] / total)
    return csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64), dangling


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


def tp_iter(P_up, P_dn, E_b, beta, rho, dangling, tol, max_iter):
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
        if diff < tol:
            return v_up + v_dn, k + 1
    return v_up + v_dn, max_iter


def main():
    t0 = time.time()
    print("[load] graph + raw popularity")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links = list(graph_data["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    # uniform speed/lane (defaults)
    w_up = np.ones(N); w_dn = np.ones(N)
    P_up, dangling = build_sp_kernel(graph_data, links, lid_to_idx, w_up)
    P_dn, _ = build_sp_kernel(graph_data, links, lid_to_idx, w_dn)
    P_diff = build_diffusion_P(graph_data, links, lid_to_idx)

    pop = load_popularity_npz(str(config.DATA_DIR / "popularity_results_osm.npz"))
    M = pop["matrix"]
    times = [str(t) for t in pop["times"]]
    src_ids = [str(lid) for lid in pop["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in src_ids], dtype=np.int64)
    valid = proj >= 0
    time_idx = {t: i for i, t in enumerate(times)}

    def get5(t_str):
        ti = time_idx.get(t_str)
        if ti is None: return None
        row = M.getrow(ti).toarray().ravel().astype(np.float64)
        C = np.zeros(N, dtype=np.float64)
        np.add.at(C, proj[valid], row[valid])
        return C

    def smooth(C):
        if C.sum() == 0: return None
        c = C + ALPHA_LAPLACE / N
        c = (1.0 - GAMMA) * c + GAMMA * (P_diff @ c)
        return c / c.sum()

    out = dict(meta=dict(days=TARGET_DAYS, labels=LABELS,
                          bin_min=BIN_MIN, gamma=GAMMA,
                          beta=BETA, rho=RHO),
                results={})
    for d in TARGET_DAYS:
        print(f"\n[day] {d} ({LABELS[d]})")
        rec = []
        # slots are 00:00, 00:30, 01:00, ..., 23:30, plus 00:00 next day
        start = datetime.strptime(d, "%Y-%m-%d")
        for slot_idx in range(48):
            slot_dt = start + timedelta(minutes=BIN_MIN * slot_idx)
            tgt_dt = slot_dt + timedelta(minutes=BIN_MIN)
            # aggregate 5-min counts for the 30-min slot
            C = np.zeros(N)
            for k in range(BINS_PER_SLOT):
                t5 = (slot_dt + timedelta(minutes=5*k)).strftime("%Y-%m-%d %H:%M:%S")
                c = get5(t5)
                if c is not None: C += c
            C_tgt = np.zeros(N)
            for k in range(BINS_PER_SLOT):
                t5 = (tgt_dt + timedelta(minutes=5*k)).strftime("%Y-%m-%d %H:%M:%S")
                c = get5(t5)
                if c is not None: C_tgt += c
            E_t = smooth(C)
            E_tn = smooth(C_tgt)
            if E_t is None or E_tn is None: continue
            v_t, niter = tp_iter(P_up, P_dn, E_t, BETA, RHO, dangling,
                                  TOL, MAX_ITER)
            sv = v_t.sum()
            if sv > 0: v_t = v_t / sv

            any_pos = (E_tn > 0) | (E_t > 0)
            top_idx = np.argsort(E_tn)[::-1][:TOP_K]
            mask_top = np.zeros(N, dtype=bool); mask_top[top_idx] = True
            d2m_chain_all = float(np.mean(np.abs(E_tn[any_pos] - v_t[any_pos]) / (E_tn[any_pos] + EPS)))
            d2m_d2d_all = float(np.mean(np.abs(E_tn[any_pos] - E_t[any_pos]) / (E_tn[any_pos] + EPS)))
            d2m_chain_top = float(np.mean(np.abs(E_tn[mask_top] - v_t[mask_top]) / (E_tn[mask_top] + EPS)))
            d2m_d2d_top = float(np.mean(np.abs(E_tn[mask_top] - E_t[mask_top]) / (E_tn[mask_top] + EPS)))
            try: r_chain = float(pearsonr(E_tn[any_pos], v_t[any_pos]).statistic)
            except Exception: r_chain = float("nan")
            try: r_d2d = float(pearsonr(E_tn[any_pos], E_t[any_pos]).statistic)
            except Exception: r_d2d = float("nan")
            rec.append(dict(
                slot=slot_dt.strftime("%H:%M"),
                hour=slot_dt.hour,
                d2m_chain_all=d2m_chain_all,
                d2m_chain_top100=d2m_chain_top,
                d2m_d2d_all=d2m_d2d_all,
                d2m_d2d_top100=d2m_d2d_top,
                r_chain=r_chain, r_d2d=r_d2d,
                n_active=int(any_pos.sum()),
                niter=niter,
            ))
        out["results"][d] = rec
        print(f"  recorded {len(rec)} slots")

    out_path = HERE / "results" / "event_sept15" / "wang_day_detail.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nsaved {out_path}  [done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
