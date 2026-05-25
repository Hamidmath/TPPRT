"""Plot variance retained as a function of the diffusion strength gamma,
on the noise=4 popularity matrix.

For each gamma in a sweep:
  c_tilde = (1 - gamma) * c + gamma * (P @ c)
where c is one row (5-min bin) of the raw popularity matrix and P is the
row-stochastic adjacency of the directed road graph.

Variance retained per bin = Var(c_tilde) / Var(c). We average over bins.
"""
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project")
import sys
sys.path.insert(0, str(ROOT))
import config
from core.io import load_popularity_npz

GRAPH_FILE = config.GRAPH_FILE
RAW_NPZ = ROOT / "data" / "popularity_results.npz"
OUT_PNG = ROOT / "figures" / "diffusion_variance_noise4.png"
OUT_JSON = ROOT / "results" / "smoothing_variance_noise4.json"

GAMMAS = [0.00, 0.02, 0.05, 0.08, 0.10, 0.13, 0.15, 0.18, 0.20, 0.22, 0.24,
          0.26, 0.28, 0.30, 0.33, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.85, 1.00]


def build_P(graph_data):
    """Row-stochastic adjacency, N x N, uniform-over-outgoing."""
    links = list(graph_data["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    adj = graph_data.get("adjacency", {})
    row, col, data = [], [], []
    for i, lid in enumerate(links):
        out_lids = [ol for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not out_lids:
            continue
        w = 1.0 / len(out_lids)
        for ol in out_lids:
            row.append(i); col.append(lid_to_idx[ol]); data.append(w)
    P = csr_matrix((data, (row, col)), shape=(N, N))
    return P, links, lid_to_idx


def main():
    t0 = time.time()
    print(f"[graph] loading {GRAPH_FILE}", flush=True)
    with open(GRAPH_FILE) as f:
        graph = json.load(f)
    P, links, lid_to_idx = build_P(graph)
    N = len(links)
    print(f"  N = {N:,}, P nnz = {P.nnz:,}", flush=True)

    print(f"[npz] loading {RAW_NPZ}", flush=True)
    bundle = load_popularity_npz(str(RAW_NPZ))
    matrix = bundle["matrix"]  # CSR or _DenseCSR, shape T x N_pop
    times = bundle["times"]
    pop_link_ids = bundle["link_ids"]
    T = len(times)
    print(f"  T = {T:,}, matrix shape = {matrix.shape}", flush=True)

    # project popularity link order onto graph link order
    proj = np.fromiter(
        (lid_to_idx.get(lid, -1) for lid in pop_link_ids),
        dtype=np.int64, count=len(pop_link_ids),
    )
    valid = proj >= 0

    # Build dense T x N popularity matrix (T x N is ~8640 x 99716 = 860M cells
    # of float64 = ~6.4 GB; this fits on most machines but is big.
    # Process bin-by-bin instead.

    var_before = np.zeros(T)
    var_after = {gamma: np.zeros(T) for gamma in GAMMAS}

    print(f"[diffusion] sweeping {len(GAMMAS)} gammas over {T:,} bins...",
          flush=True)
    t_sw = time.time()
    for i in range(T):
        row = matrix.getrow(i).toarray().ravel()
        c = np.zeros(N)
        np.add.at(c, proj[valid], row[valid])
        s = c.sum()
        if s == 0:
            continue
        # raw bin -> probability distribution
        c_norm = c / s
        var_before[i] = float(c_norm.var())
        if var_before[i] == 0:
            continue
        # diffusion on the raw counts, then renormalise the mixture to
        # a probability distribution (matches the paper pipeline).
        Pc = P @ c
        for gamma in GAMMAS:
            c_tilde = (1 - gamma) * c + gamma * Pc
            ss = c_tilde.sum()
            if ss > 0:
                c_tilde = c_tilde / ss
            var_after[gamma][i] = float(c_tilde.var())
        if (i + 1) % 500 == 0:
            print(f"  bin {i+1}/{T}  ({time.time() - t_sw:.0f}s)", flush=True)
    print(f"[diffusion] done in {time.time() - t_sw:.0f}s", flush=True)

    # per-gamma var-retained (averaged over bins with var>0)
    mask = var_before > 0
    var_retained = {}
    for gamma in GAMMAS:
        ratios = var_after[gamma][mask] / var_before[mask]
        var_retained[gamma] = float(ratios.mean())

    # write json
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"gamma": g, "var_retained": var_retained[g]} for g in GAMMAS]
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nsaved {OUT_JSON}")
    for p in payload:
        marker = "  <- chosen" if abs(p["gamma"] - 0.26) < 1e-9 else ""
        print(f"  gamma = {p['gamma']:.2f}  var_retained = {p['var_retained']:.4f}{marker}")

    # plot
    gs = [p["gamma"] for p in payload]
    vrs = [p["var_retained"] for p in payload]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(gs, vrs, "o-", color="#1f77b4", lw=1.8, ms=5)
    # mark gamma=0.26
    idx_chosen = next(i for i, p in enumerate(payload)
                      if abs(p["gamma"] - 0.26) < 1e-9)
    ax.scatter([gs[idx_chosen]], [vrs[idx_chosen]],
               color="red", s=80, zorder=5, label=r"chosen $\gamma=0.26$")
    ax.annotate(f"$\\gamma=0.26$\nvar retained = {vrs[idx_chosen]:.2f}",
                (gs[idx_chosen], vrs[idx_chosen]),
                xytext=(20, 18), textcoords="offset points",
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    ax.axhline(1.0, color="gray", linestyle=":", lw=0.7)
    ax.set_xlabel(r"diffusion strength  $\gamma$")
    ax.set_ylabel(r"variance retained: $\mathrm{Var}(\tilde c_b) / \mathrm{Var}(c_b)$, mean over bins")
    ax.set_title(f"Diffusion variance trade-off on noise=4 popularity matrix\n"
                 f"({matrix.shape[0]:,} bins, $N = {N:,}$ links)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    print(f"\nsaved figure: {OUT_PNG}")
    print(f"\ntotal: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
