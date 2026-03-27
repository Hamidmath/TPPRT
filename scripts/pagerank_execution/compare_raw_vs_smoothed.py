#!/usr/bin/env python3
"""
Compare Two-Phase PageRank results: Raw vs Smoothed (gamma=0.26).
Runs PageRank on both inputs and reports MRE metrics.
"""
import json
import logging
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, hstack, vstack

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
DATA = BASE / "../../data"
GRAPH_FILE = DATA / "city_graph_full.json"
RAW_NPZ = DATA / "popularity_results.npz"
SMOOTHED_NPZ = DATA / "popularity_results_smoothed_026.npz"

PARAMS = {
    'beta': 0.9, 'damping': 0.80, 'mu': 20.0,
    'tau': 1.0, 'top_k_boost': 1.0,
    'max_iters': 100, 'tol': 1e-6
}

TARGET_TIME = "2018-09-08 08:00:00"

def load_graph():
    with open(GRAPH_FILE) as f:
        return json.load(f)

def build_two_phase_matrix(graph_data):
    links = list(graph_data['links'].keys())
    N = len(links)
    id_to_idx = {lid: i for i, lid in enumerate(links)}
    mu = PARAMS['mu']
    adj = graph_data.get('adjacency', {})

    row, col, data = [], [], []
    for i, lid in enumerate(links):
        out_links = adj.get(lid, [])
        succs = [id_to_idx[o] for o in out_links if o in id_to_idx]
        edge = graph_data['links'][lid]
        self_w = mu * edge.get('length', 100.0) / max(edge.get('speed', 11.17), 0.1)
        total = len(succs) + self_w
        if total > 0:
            if self_w > 0:
                row.append(i); col.append(i); data.append(self_w / total)
            for j in succs:
                row.append(i); col.append(j); data.append(1.0 / total)

    P_up = csr_matrix((data, (row, col)), shape=(N, N))

    # Down phase: simple adjacency (no self-loops)
    row2, col2, data2 = [], [], []
    for i, lid in enumerate(links):
        out_links = adj.get(lid, [])
        succs = [id_to_idx[o] for o in out_links if o in id_to_idx]
        if succs:
            w = 1.0 / len(succs)
            for j in succs:
                row2.append(i); col2.append(j); data2.append(w)
    P_down = csr_matrix((data2, (row2, col2)), shape=(N, N))

    beta = PARAMS['beta']
    P_up_scaled = P_up.multiply(1.0 - beta)
    T_down = csr_matrix((np.ones(N) * beta, (np.arange(N), np.arange(N))), shape=(N, N))
    zero = csr_matrix((N, N))

    M = vstack([hstack([P_up_scaled, T_down]), hstack([zero, P_down])])
    return M, N

def build_teleportation(npz_path, graph_data, target_time):
    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    loader = np.load(npz_path, allow_pickle=True)
    matrix = csr_matrix((loader['matrix_data'], loader['matrix_indices'],
                         loader['matrix_indptr']), shape=loader['matrix_shape'])
    times = list(loader['times'])
    pop_ids = list(loader['link_ids'])

    if target_time not in times:
        target_time = times[0]
    t_idx = times.index(target_time)
    row = matrix.getrow(t_idx)

    E_N = np.zeros(N)
    for i, val in zip(row.indices, row.data):
        lid = pop_ids[i]
        if lid in lid_to_idx:
            E_N[lid_to_idx[lid]] = float(val)

    s = E_N.sum()
    if s > 0:
        E_N /= s
    else:
        E_N = np.ones(N) / N
    return E_N

def build_raw_teleportation(graph_data, target_time, alpha=0.1):
    """Build teleportation from raw data with only Laplace prior (no smoothing)."""
    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    loader = np.load(RAW_NPZ, allow_pickle=True)
    matrix = csr_matrix((loader['matrix_data'], loader['matrix_indices'],
                         loader['matrix_indptr']), shape=loader['matrix_shape'])
    times = list(loader['times'])
    pop_ids = list(loader['link_ids'])

    if target_time not in times:
        target_time = times[0]
    t_idx = times.index(target_time)
    row = matrix.getrow(t_idx)

    c = np.zeros(N) + alpha  # Laplace prior
    for i, val in zip(row.indices, row.data):
        lid = pop_ids[i]
        if lid in lid_to_idx:
            c[lid_to_idx[lid]] += float(val)

    E_N = c / c.sum()
    return E_N

def power_iteration(M, E_2N):
    d = PARAMS['damping']
    M_T = M.transpose()
    v = E_2N.copy()
    for i in range(PARAMS['max_iters']):
        v_next = d * M_T.dot(v) + (1 - d) * E_2N
        s = v_next.sum()
        if s < 1.0:
            v_next += E_2N * (1.0 - s)
        diff = np.abs(v_next - v).sum()
        v = v_next
        if diff < PARAMS['tol']:
            logger.info(f"  Converged at iteration {i} (diff={diff:.2e})")
            break
    return v

def compute_mre(truth, pred, N, top_k=None):
    if top_k:
        idx = np.argsort(truth)[::-1][:top_k]
        t, p = truth[idx], pred[idx]
    else:
        t, p = truth, pred
    return np.mean(np.abs(t - p) / (t + 1e-9))

def main():
    graph = load_graph()
    links = list(graph['links'].keys())
    N = len(links)

    logger.info(f"Building Two-Phase Matrix (N={N})...")
    M, N = build_two_phase_matrix(graph)
    logger.info(f"Matrix shape: {M.shape}")

    # ─── Run 1: Raw data (Laplace only, no smoothing) ───
    logger.info("=" * 60)
    logger.info("RUN 1: Raw data (no smoothing, Laplace alpha=0.1 only)")
    logger.info("=" * 60)
    E_raw = build_raw_teleportation(graph, TARGET_TIME)
    E_2N_raw = np.concatenate([E_raw, np.zeros(N)])
    v_raw = power_iteration(M, E_2N_raw)
    v_raw_final = v_raw[:N] + v_raw[N:]
    v_raw_final /= v_raw_final.sum()

    mre_raw_all = compute_mre(E_raw, v_raw_final, N)
    mre_raw_100 = compute_mre(E_raw, v_raw_final, N, top_k=100)
    logger.info(f"  Full MRE:   {mre_raw_all:.4f}")
    logger.info(f"  Top-100 MRE: {mre_raw_100:.4f}")

    # ─── Run 2: Smoothed data (gamma=0.26) ───
    logger.info("=" * 60)
    logger.info("RUN 2: Smoothed data (gamma=0.26)")
    logger.info("=" * 60)
    E_smooth = build_teleportation(SMOOTHED_NPZ, graph, TARGET_TIME)
    E_2N_smooth = np.concatenate([E_smooth, np.zeros(N)])
    v_smooth = power_iteration(M, E_2N_smooth)
    v_smooth_final = v_smooth[:N] + v_smooth[N:]
    v_smooth_final /= v_smooth_final.sum()

    mre_smooth_all = compute_mre(E_smooth, v_smooth_final, N)
    mre_smooth_100 = compute_mre(E_smooth, v_smooth_final, N, top_k=100)
    logger.info(f"  Full MRE:   {mre_smooth_all:.4f}")
    logger.info(f"  Top-100 MRE: {mre_smooth_100:.4f}")

    # ─── Cross-comparison: model output vs the OTHER ground truth ───
    logger.info("=" * 60)
    logger.info("CROSS-COMPARISON")
    logger.info("=" * 60)
    cross_raw_vs_smooth = compute_mre(E_smooth, v_raw_final, N)
    cross_raw_vs_smooth_100 = compute_mre(E_smooth, v_raw_final, N, top_k=100)
    cross_smooth_vs_raw = compute_mre(E_raw, v_smooth_final, N)
    cross_smooth_vs_raw_100 = compute_mre(E_raw, v_smooth_final, N, top_k=100)

    logger.info(f"  Raw model vs smoothed GT:    Full={cross_raw_vs_smooth:.4f}, Top-100={cross_raw_vs_smooth_100:.4f}")
    logger.info(f"  Smooth model vs raw GT:      Full={cross_smooth_vs_raw:.4f}, Top-100={cross_smooth_vs_raw_100:.4f}")

    # ─── Summary ───
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<35} {'Raw (no smooth)':<18} {'Smooth (g=0.26)':<18}")
    print("-" * 70)
    print(f"{'Full MRE (self-consistency)':<35} {mre_raw_all:<18.4f} {mre_smooth_all:<18.4f}")
    print(f"{'Top-100 MRE (self-consistency)':<35} {mre_raw_100:<18.4f} {mre_smooth_100:<18.4f}")
    print(f"{'Full MRE vs smoothed GT':<35} {cross_raw_vs_smooth:<18.4f} {'(baseline)':<18}")
    print(f"{'Full MRE vs raw GT':<35} {'(baseline)':<18} {cross_smooth_vs_raw:<18.4f}")
    print(f"{'Top-100 MRE vs smoothed GT':<35} {cross_raw_vs_smooth_100:<18.4f} {'(baseline)':<18}")
    print(f"{'Top-100 MRE vs raw GT':<35} {'(baseline)':<18} {cross_smooth_vs_raw_100:<18.4f}")
    print("=" * 70)

    raw_nonzero = np.sum(E_raw > 1e-6)
    smooth_nonzero = np.sum(E_smooth > 1e-6)
    print(f"\nTeleportation vector stats:")
    print(f"  Raw:      {raw_nonzero:,} links with significant mass")
    print(f"  Smoothed: {smooth_nonzero:,} links with significant mass")
    print(f"  Raw max/min ratio:      {E_raw.max()/E_raw[E_raw>0].min():.0f}x")
    print(f"  Smoothed max/min ratio: {E_smooth.max()/E_smooth[E_smooth>0].min():.0f}x")

if __name__ == "__main__":
    main()