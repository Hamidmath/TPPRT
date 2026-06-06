"""Quantiles of D_{b,i}(gamma) / sigma^NF_global for ZERO cells
(those structurally empty across all 4 weeks).

Same grid as the existing per-cell table; output to a JSON.
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
SIGMA_NF_GLOBAL = 2.65e-4
GRID_GAMMA = [0.0, 0.05, 0.10, 0.12, 0.13, 0.14, 0.15, 0.20,
              0.25, 0.30, 0.40, 0.50, 0.70]
WEEKS = 4
BINS_PER_WEEK = 7 * 24 * 12


def build_diffusion_P(graph_data, links):
    id_to_idx = {lid: i for i, lid in enumerate(links)}
    adj = graph_data.get("adjacency", {})
    N = len(links)
    row, col, data = [], [], []
    for i, lid in enumerate(links):
        succ = [id_to_idx[ol] for ol in adj.get(lid, []) if ol in id_to_idx]
        if not succ: continue
        w = 1.0 / len(succ)
        for j in succ:
            row.append(i); col.append(j); data.append(w)
    return csr_matrix((data, (row, col)), shape=(N, N), dtype=np.float64)


def main():
    t0 = time.time()
    print("[load] graph + raw popularity")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links = list(graph_data["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    P_diff = build_diffusion_P(graph_data, links)

    pop = load_popularity_npz(str(config.DATA_DIR / "popularity_results_osm.npz"))
    matrix = pop["matrix"]
    times = [str(t) for t in pop["times"]]
    pop_link_ids = pop["link_ids"]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_link_ids],
                    dtype=np.int64)
    valid = proj >= 0
    time_index = {t: i for i, t in enumerate(times)}

    week_starts = [datetime(2018, 9, 1) + timedelta(weeks=w) for w in range(WEEKS)]

    def get_count_row(time_str):
        ti = time_index.get(time_str)
        if ti is None: return None
        r = matrix.getrow(ti).toarray().ravel().astype(np.float64)
        C = np.zeros(N, dtype=np.float64)
        np.add.at(C, proj[valid], r[valid])
        return C

    print(f"[gather] {WEEKS} weeks x {BINS_PER_WEEK} bins")
    E_rows = np.zeros((WEEKS, BINS_PER_WEEK, N), dtype=np.float32)
    for w in range(WEEKS):
        ws = week_starts[w]
        for b in range(BINS_PER_WEEK):
            t = (ws + timedelta(minutes=5*b)).strftime("%Y-%m-%d %H:%M:%S")
            C = get_count_row(t)
            if C is not None:
                s = C.sum()
                if s > 0: E_rows[w, b] = (C / s).astype(np.float32)
        print(f"  week {w} done", flush=True)

    raw_any_pos = (E_rows > 0).any(axis=0)
    raw_all_zero = ~raw_any_pos
    n_zero = int(raw_all_zero.sum())
    print(f"  active cells: {int(raw_any_pos.sum())}")
    print(f"  zero cells (structurally empty in all 4 weeks): {n_zero:,}")

    results = []
    for gamma in GRID_GAMMA:
        t_g = time.time()
        # D_avg = mean over weeks of |tilde_E - E|; for zero cells E=0 so D_avg = mean of tilde_E
        D_avg = np.zeros((BINS_PER_WEEK, N), dtype=np.float64)
        for w in range(WEEKS):
            for b in range(BINS_PER_WEEK):
                E = E_rows[w, b].astype(np.float64)
                if E.sum() == 0:
                    # week w has no observations at this bin-of-week; tilde_E still has Laplace+diffusion
                    # We need to compute tilde_E from c = E + alpha/N (E=0) -> c = alpha/N broadcast.
                    # diffusing a constant by P (row-stochastic) leaves a constant since P*const=const.
                    # so tilde_E = const = (alpha/N) / N if we renormalize? actually each row has alpha/N for all i; sum=alpha; normalized: (alpha/N)/alpha = 1/N
                    # |tilde_E - E| = 1/N - 0 = 1/N
                    D_avg[b] += 1.0 / N
                    continue
                c = E + ALPHA_LAPLACE / N
                if gamma > 0:
                    c_diff = (1.0 - gamma) * c + gamma * (P_diff @ c)
                else:
                    c_diff = c
                s = c_diff.sum()
                if s > 0: c_diff = c_diff / s
                D_avg[b] += np.abs(c_diff - E)
        D_avg /= WEEKS

        D_zero = D_avg[raw_all_zero]
        ratio = D_zero / SIGMA_NF_GLOBAL
        r = dict(gamma=float(gamma),
                  median=float(np.median(ratio)),
                  p95=float(np.percentile(ratio, 95)),
                  p99=float(np.percentile(ratio, 99)),
                  mean=float(ratio.mean()),
                  frac_above_half=float((ratio > 0.5).mean()),
                  elapsed_s=time.time() - t_g)
        results.append(r)
        print(f"  gamma={r['gamma']:.2f}  median={r['median']:.3f}  "
              f"p95={r['p95']:.3f}  mean={r['mean']:.3f}  "
              f"frac>0.5={r['frac_above_half']*100:.2f}%  "
              f"t={time.time()-t_g:.0f}s", flush=True)

    out = HERE / "results" / "event_sept15" / "gamma_zero_cells.json"
    with open(out, "w") as f:
        json.dump({"sigma_nf_global": SIGMA_NF_GLOBAL,
                   "n_zero_cells": n_zero,
                   "grid_gamma": GRID_GAMMA,
                   "results": results,
                   "elapsed_s": time.time() - t0}, f, indent=2)
    print(f"saved {out}  [done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
