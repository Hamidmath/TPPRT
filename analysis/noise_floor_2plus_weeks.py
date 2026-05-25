"""Alternative cross-week noise floor histogram filtered to cells
observed in at least 2 of the 4 weeks.

The standard figure mixes cells with one week of data (whose
"std" is dominated by the three zeros, producing a tight spike
near the leftmost bin) together with cells observed in 2 to 4
weeks. To make the distribution of "real" jitter visible, this
script keeps only cells whose number of weeks with at least one
positive E_b(i) value is >= 2.

Output: figures/noise_floor_2plus_weeks.pdf
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, diags

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz

N_BINS_WEEK = 7 * 288
N_WEEKS = 4
N_FULL = N_WEEKS * N_BINS_WEEK
MIN_WEEKS = 2

OUT_FIG = Path(
    "/home/hamid/Downloads/new/TwoPhase_PageRank_Project/documents/walkthrough2/figures"
) / "noise_floor_2plus_weeks.pdf"


def load_M_norm():
    print("[load] graph + popularity ...")
    with open(config.GRAPH_FILE) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    b = load_popularity_npz(str(config.DATA_DIR / "popularity_results_osm.npz"))
    matrix = b["matrix"]
    pop_link_ids = b["link_ids"]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_link_ids], dtype=np.int64)
    valid_mask = proj >= 0
    coo = matrix.tocoo()
    sel = valid_mask[coo.col]
    rows_sel = coo.row[sel]
    cols_sel = proj[coo.col[sel]]
    data_sel = coo.data[sel].astype(np.float32)
    T = len(b["times"])
    M_raw = csr_matrix((data_sel, (rows_sel, cols_sel)),
                       shape=(T, N), dtype=np.float32)
    rs = np.asarray(M_raw.sum(axis=1)).ravel()
    inv = np.where(rs > 0, 1.0 / rs, 0.0).astype(np.float32)
    M_norm = diags(inv) @ M_raw
    print(f"  N = {N:,}, T = {T}")
    return M_norm, T, N


def compute_stds_and_week_counts(M_norm, T, N):
    """Per (link, bow): cross-week std, plus the number of weeks
    in which E_b(i) > 0."""
    full = min(N_FULL, T)
    n_per_week = N_BINS_WEEK
    print(f"[stds] {N_WEEKS} weeks x {n_per_week} bows x {N} links ...")
    t0 = time.time()
    stds = np.zeros((n_per_week, N), dtype=np.float32)
    n_weeks_pos = np.zeros((n_per_week, N), dtype=np.uint8)
    for bow in range(n_per_week):
        rows = []
        for w in range(N_WEEKS):
            t = w * n_per_week + bow
            if t >= full:
                break
            r = M_norm.getrow(t).toarray().ravel()
            rows.append(r)
        if not rows:
            continue
        rows_arr = np.stack(rows, axis=0)  # (W, N)
        stds[bow] = rows_arr.std(axis=0, ddof=1) if len(rows) > 1 else 0.0
        n_weeks_pos[bow] = (rows_arr > 0).sum(axis=0)
        if bow % 400 == 0:
            print(f"  bow {bow}/{n_per_week} ({time.time()-t0:.0f}s)")
    print(f"[stds] done in {time.time()-t0:.0f}s")
    return stds, n_weeks_pos


def main():
    t0 = time.time()
    M_norm, T, N = load_M_norm()
    stds, n_weeks_pos = compute_stds_and_week_counts(M_norm, T, N)

    n_weeks_pos_flat = n_weeks_pos.ravel()
    stds_flat = stds.ravel().astype(np.float64)

    # Filter: cells observed in >= MIN_WEEKS of the 4 weeks.
    keep = n_weeks_pos_flat >= MIN_WEEKS
    stds_kept = stds_flat[keep]
    n_total = stds_flat.size
    n_kept = int(keep.sum())
    print(f"  cells observed in >= {MIN_WEEKS} weeks: {n_kept:,} "
          f"({100*n_kept/n_total:.2f}%)")
    print(f"  cells dropped (0 or 1 week): {n_total - n_kept:,}")

    med = float(np.median(stds_kept))
    mean = float(stds_kept.mean())
    print(f"  median = {med:.3e}  mean = {mean:.3e}")

    EDGES = np.logspace(-9, -2, 80)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.hist(np.clip(stds_kept, 1e-9, 1e-2),
            bins=EDGES, color="#1f6fb4", alpha=0.85,
            rasterized=True)
    ax.axvline(med, color="red", ls="--", lw=1.4,
               label=f"median = {med:.2e}")
    ax.axvline(mean, color="orange", ls=":", lw=1.4,
               label=f"mean = {mean:.2e}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"cross-week std $\tau_{b,i}$ (probability units)")
    ax.set_ylabel("count")
    ax.set_title(f"Cells observed in $\\geq$ {MIN_WEEKS} of the 4 weeks "
                 f"({n_kept:,} pairs)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=180)
    plt.close(fig)
    print(f"saved {OUT_FIG}")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
