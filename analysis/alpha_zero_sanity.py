"""Sanity check for the single-phase iteration at alpha = 0.

Professor concern: if the implementation has the teleport probability
and damping swapped, running with alpha = 0 (no teleport, all
transition) will produce a vector that resembles E_b. The correct
behavior is the opposite: at alpha = 0 the iteration is

    v_{n+1} = P_cs @ v_n,

which converges to the dominant left eigenvector of the row-stochastic
graph and is *independent* of the initial v (so independent of E_b).

This script picks two bins with very different popularity vectors,
runs the single-phase iteration with alpha = 0 starting from each,
and reports:
  - max |v_a - v_b|        (should be ~0)
  - corr(v_a, E_a)         (should be small)
  - corr(v_a, v_b)         (should be ~1)

It also runs alpha = 1 as a contrast: at alpha = 1 the iteration is
v_{n+1} = E_b, so the output must equal E_b exactly.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz


TOL, MAX_ITER = 1e-9, 1000
GRAPH_JSON = str(config.GRAPH_FILE)
RAW_NPZ    = str(config.DATA_DIR / "popularity_results_osm.npz")


def build_P_cs(graph):
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    adj = graph.get("adjacency", {})
    rows, cols, data = [], [], []
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ:
            continue
        w = 1.0 / len(succ)
        for j in succ:
            rows.append(j); cols.append(i); data.append(w)
    return csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64), links, lid_to_idx


def single_phase_iter(P_cs, E_b, alpha_teleport, v_init, tol, max_iter):
    """v_{n+1} = alpha * E_b + (1 - alpha) * (P_cs @ v_n)"""
    v = v_init.copy()
    for k in range(max_iter):
        v_new = alpha_teleport * E_b + (1.0 - alpha_teleport) * (P_cs @ v)
        s = float(v_new.sum())
        if s > 0:
            v_new /= s
        diff = float(np.abs(v_new - v).sum())
        v = v_new
        if diff < tol:
            return v, k + 1
    return v, max_iter


def load_two_bins(graph_links, lid_to_idx, N):
    b = load_popularity_npz(RAW_NPZ)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_ids = b["link_ids"]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_ids], dtype=np.int64)
    valid = proj >= 0

    def E_at(t_idx):
        row = M.getrow(t_idx).toarray().ravel().astype(np.float64)
        E = np.zeros(N, dtype=np.float64)
        np.add.at(E, proj[valid], row[valid])
        s = E.sum()
        return E / s if s > 0 else E

    # Pick two very different bins: Monday 08:00 (rush) and Sun 03:00 (off-peak).
    target_a = "2018-09-03 08:00:00"
    target_b = "2018-09-09 03:00:00"
    if target_a not in times or target_b not in times:
        target_a, target_b = times[100], times[2000]
    return E_at(times.index(target_a)), E_at(times.index(target_b)), target_a, target_b


def main():
    print("[alpha=0 sanity] loading graph & data ...")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    P_cs, links, lid_to_idx = build_P_cs(graph)
    N = len(links)
    print(f"  N = {N:,}   P_cs nnz = {P_cs.nnz:,}")

    E_a, E_b, t_a, t_b = load_two_bins(links, lid_to_idx, N)
    corr_E = float(np.corrcoef(E_a, E_b)[0, 1])
    print(f"  bin A = {t_a}   sum = {E_a.sum():.4f}   nz = {(E_a > 0).sum()}")
    print(f"  bin B = {t_b}   sum = {E_b.sum():.4f}   nz = {(E_b > 0).sum()}")
    print(f"  corr(E_a, E_b) = {corr_E:.4f}   (should be << 1)")

    # alpha = 0 (never teleport): should converge to the dominant eigenvector,
    # independent of E_b.
    print("\n[alpha = 0]  v_{n+1} = P_cs @ v_n   (expect output INDEPENDENT of E_b)")
    v_a, k_a = single_phase_iter(P_cs, E_a, 0.0, E_a, TOL, MAX_ITER)
    v_b, k_b = single_phase_iter(P_cs, E_b, 0.0, E_b, TOL, MAX_ITER)
    delta = float(np.abs(v_a - v_b).max())
    delta_l1 = float(np.abs(v_a - v_b).sum())
    corr_vv = float(np.corrcoef(v_a, v_b)[0, 1])
    corr_va_Ea = float(np.corrcoef(v_a, E_a)[0, 1])
    corr_va_Eb = float(np.corrcoef(v_a, E_b)[0, 1])
    print(f"  iters: A {k_a}, B {k_b}")
    print(f"  max |v_a - v_b|     = {delta:.3e}   (should be ~0)")
    print(f"  L1  |v_a - v_b|     = {delta_l1:.3e}   (should be ~0)")
    print(f"  corr(v_a, v_b)      = {corr_vv:.6f}    (should be ~1.0)")
    print(f"  corr(v_a, E_a)      = {corr_va_Ea:.4f}    (should be small, not ~1)")
    print(f"  corr(v_a, E_b)      = {corr_va_Eb:.4f}    (should be similar to corr(v_a, E_a))")

    if corr_vv > 0.999 and delta < 1e-6:
        verdict_zero = "PASS - alpha=0 output is independent of E_b (eigenvector behaviour)"
    elif corr_va_Ea > 0.99 and corr_vv < 0.99:
        verdict_zero = "FAIL - alpha=0 output tracks E_b => alpha and (1-alpha) appear swapped"
    else:
        verdict_zero = "INCONCLUSIVE - see numbers"
    print(f"  verdict: {verdict_zero}")

    # alpha = 1 (always teleport): output must equal E_b exactly.
    print("\n[alpha = 1]  v_{n+1} = E_b   (expect output EQUAL to E_b)")
    v_a1, _ = single_phase_iter(P_cs, E_a, 1.0, E_b, TOL, MAX_ITER)
    delta1 = float(np.abs(v_a1 - E_a).max())
    corr_va1_Ea = float(np.corrcoef(v_a1, E_a)[0, 1])
    print(f"  max |v_a(alpha=1) - E_a| = {delta1:.3e}   (should be ~0)")
    print(f"  corr(v_a(alpha=1), E_a)  = {corr_va1_Ea:.6f}   (should be ~1.0)")
    verdict_one = "PASS" if delta1 < 1e-8 else "FAIL"
    print(f"  verdict: {verdict_one}")

    out = {
        "alpha_zero": {
            "max_abs_diff": delta,
            "l1_diff": delta_l1,
            "corr_v_a_v_b": corr_vv,
            "corr_v_a_E_a": corr_va_Ea,
            "corr_v_a_E_b": corr_va_Eb,
            "verdict": verdict_zero,
            "iters_a": k_a, "iters_b": k_b,
        },
        "alpha_one": {
            "max_abs_diff_to_E_a": delta1,
            "corr_to_E_a": corr_va1_Ea,
            "verdict": verdict_one,
        },
        "bins": {"a": t_a, "b": t_b, "corr_E_a_E_b": corr_E},
    }
    out_path = Path(config.RESULTS_DIR) / "alpha_zero_sanity.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
