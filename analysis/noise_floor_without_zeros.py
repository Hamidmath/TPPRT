"""Single-panel version of noise_floor_with_zeros.py: keep only the
"active cells only" view (the left panel). Output saved to
documents/walkthrough2/figures/noise_floor_without_zeros.pdf (+ .png).
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

OUT_DIR = Path(
    "/home/hamid/Downloads/new/TwoPhase_PageRank_Project/documents/walkthrough2/figures"
)
OUT_PDF = OUT_DIR / "noise_floor_without_zeros.pdf"
OUT_PNG = OUT_DIR / "noise_floor_without_zeros.png"


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
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_link_ids],
                     dtype=np.int64)
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


def compute_stds(M_norm, T, N):
    full = min(N_FULL, T)
    n_per_week = N_BINS_WEEK
    print(f"[stds] {N_WEEKS} weeks x {n_per_week} bows x {N} links ...")
    t0 = time.time()
    stds = np.zeros((n_per_week, N), dtype=np.float32)
    active = np.zeros((n_per_week, N), dtype=bool)
    for bow in range(n_per_week):
        rows = []
        any_pos = np.zeros(N, dtype=bool)
        for w in range(N_WEEKS):
            t = w * n_per_week + bow
            if t >= full:
                break
            r = M_norm.getrow(t).toarray().ravel()
            rows.append(r)
            any_pos |= r > 0
        if not rows:
            continue
        rows_arr = np.stack(rows, axis=0)
        stds[bow] = rows_arr.std(axis=0, ddof=1) if len(rows) > 1 else 0.0
        active[bow] = any_pos
        if bow % 400 == 0:
            print(f"  bow {bow}/{n_per_week} ({time.time()-t0:.0f}s)")
    print(f"[stds] done in {time.time()-t0:.0f}s")
    return stds, active


def main():
    t0 = time.time()
    M_norm, T, N = load_M_norm()
    stds, active = compute_stds(M_norm, T, N)

    # Active cells only (float64 upcast so the float32 zeros don't get
    # silently dropped at the leftmost log-bin edge).
    stds_active = stds[active].astype(np.float64)
    n_active = int(active.sum())
    med_a = float(np.median(stds_active))
    mean_a = float(stds_active.mean())
    print(f"  active cells   = {n_active:,}")
    print(f"  median = {med_a:.3e}  mean = {mean_a:.3e}")

    EDGES = np.logspace(-9, -2, 80)
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.4))
    ax.hist(np.clip(stds_active, 1e-9, 1e-2),
            bins=EDGES, color="#1f6fb4", alpha=0.85)
    ax.axvline(med_a, color="red", ls="--", lw=1.4,
                label=f"median = {med_a:.2e}")
    ax.axvline(mean_a, color="orange", ls=":", lw=1.4,
                label=f"mean = {mean_a:.2e}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("cross-week std (probability units)")
    ax.set_ylabel("count")
    ax.set_title(f"active cells only ({n_active:,} pairs)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)
    print(f"saved {OUT_PDF}")
    print(f"saved {OUT_PNG}")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
