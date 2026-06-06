"""Extra γ rows for the merged Table 3 of the paper.

Computes both per-cell and global criterion quantiles at
γ ∈ {0.12, 0.13, 0.14}. Output is appended to a JSON file alongside
the existing gamma_noise_floor_global.json. The per-cell criterion
needs σ^NF_{b,i} per active cell; we compute σ^NF from the raw
popularity matrix across the four full weeks of the corpus.
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
GRID_GAMMA = [0.12, 0.13, 0.14]
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
    n_active = int(raw_any_pos.sum())
    print(f"  active cells: {n_active}")

    # per-cell sigma^NF: cross-week std on active cells
    print("[sigma^NF] computing per-cell cross-week std")
    sigma_per_cell = np.std(E_rows.astype(np.float64), axis=0, ddof=0)
    sigma_active = sigma_per_cell[raw_any_pos]
    print(f"  per-cell median = {np.median(sigma_active):.4e}")
    print(f"  per-cell global = {SIGMA_NF_GLOBAL:.4e}")

    results = []
    for gamma in GRID_GAMMA:
        t_g = time.time()
        D_avg = np.zeros((BINS_PER_WEEK, N), dtype=np.float64)
        for w in range(WEEKS):
            for b in range(BINS_PER_WEEK):
                E = E_rows[w, b].astype(np.float64)
                if E.sum() == 0: continue
                c = E + ALPHA_LAPLACE / N
                c_diff = (1.0 - gamma) * c + gamma * (P_diff @ c)
                s = c_diff.sum()
                if s > 0: c_diff = c_diff / s
                D_avg[b] += np.abs(c_diff - E)
        D_avg /= WEEKS

        D_active = D_avg[raw_any_pos]
        # safeguard against sigma_active == 0 (cells with zero std but at least one observation: shouldn't happen for raw_any_pos, but guard anyway)
        sigma_pc = np.where(sigma_active > 0, sigma_active, np.inf)
        ratio_per_cell = D_active / sigma_pc
        ratio_global = D_active / SIGMA_NF_GLOBAL

        r = dict(
            gamma=float(gamma),
            per_cell=dict(
                median=float(np.median(ratio_per_cell)),
                p95=float(np.percentile(ratio_per_cell, 95)),
                frac_above_half=float((ratio_per_cell > 0.5).mean()),
            ),
            glob=dict(
                median=float(np.median(ratio_global)),
                p95=float(np.percentile(ratio_global, 95)),
                frac_above_half=float((ratio_global > 0.5).mean()),
            ),
            elapsed_s=time.time() - t_g,
        )
        results.append(r)
        print(f"  g={gamma:.2f}  per-cell  med={r['per_cell']['median']:.3f}  "
              f"p95={r['per_cell']['p95']:.3f}  "
              f"frac>0.5={r['per_cell']['frac_above_half']*100:.2f}%  ", flush=True)
        print(f"            global    med={r['glob']['median']:.3f}  "
              f"p95={r['glob']['p95']:.3f}  "
              f"frac>0.5={r['glob']['frac_above_half']*100:.2f}%  "
              f"t={time.time()-t_g:.0f}s", flush=True)

    out = HERE / "results" / "event_sept15" / "gamma_extra_rows.json"
    with open(out, "w") as f:
        json.dump(dict(n_active=n_active,
                       sigma_per_cell_median=float(np.median(sigma_active)),
                       sigma_nf_global=SIGMA_NF_GLOBAL,
                       grid_gamma=GRID_GAMMA,
                       results=results,
                       elapsed_s=time.time() - t0), f, indent=2)
    print(f"\nsaved {out}  [done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
