#!/usr/bin/env python3
"""
Compare Two-Phase PageRank results: Raw vs Smoothed (gamma=0.26).
Runs PageRank on both inputs and reports MRE metrics.
"""
import json
import logging
import sys
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core import build_two_phase_matrix, run_power_iteration, PARAMS, build_teleportation_vector
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Override PARAMS with this script's specific settings
PARAMS['mu'] = 20.0
PARAMS['beta'] = 0.9
PARAMS['damping'] = 0.80
PARAMS['tau'] = 1.0
PARAMS['top_k_boost'] = 1.0
PARAMS['max_iters'] = 100
PARAMS['tol'] = 1e-6

TARGET_TIME = "2018-09-08 08:00:00"

def build_raw_teleportation(graph_data, target_time, alpha=0.1):
    """Build teleportation from raw data with only Laplace prior (no smoothing)."""
    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    loader = np.load(config.POPULARITY_RAW_NPZ, allow_pickle=True)
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

def compute_mre(truth, pred, N, top_k=None):
    if top_k:
        idx = np.argsort(truth)[::-1][:top_k]
        t, p = truth[idx], pred[idx]
    else:
        t, p = truth, pred
    return np.mean(np.abs(t - p) / (t + 1e-9))

def main():
    with open(config.GRAPH_FILE) as f:
        graph = json.load(f)
    links = list(graph['links'].keys())
    N = len(links)

    logger.info(f"Building Two-Phase Matrix (N={N})...")
    M_2N = build_two_phase_matrix(graph)
    logger.info(f"Matrix shape: {M_2N.shape}")

    # --- Run 1: Raw data (Laplace only, no smoothing) ---
    logger.info("=" * 60)
    logger.info("RUN 1: Raw data (no smoothing, Laplace alpha=0.1 only)")
    logger.info("=" * 60)
    E_raw = build_raw_teleportation(graph, TARGET_TIME)
    E_2N_raw = np.concatenate([E_raw, np.zeros(N)])
    v_raw = run_power_iteration(M_2N, E_2N_raw)
    v_raw_final = v_raw[:N] + v_raw[N:]
    v_raw_final /= v_raw_final.sum()

    mre_raw_all = compute_mre(E_raw, v_raw_final, N)
    mre_raw_100 = compute_mre(E_raw, v_raw_final, N, top_k=100)
    logger.info(f"  Full MRE:   {mre_raw_all:.4f}")
    logger.info(f"  Top-100 MRE: {mre_raw_100:.4f}")

    # --- Run 2: Smoothed data (gamma=0.26) ---
    logger.info("=" * 60)
    logger.info("RUN 2: Smoothed data (gamma=0.26)")
    logger.info("=" * 60)
    E_smooth = build_teleportation_vector(graph, TARGET_TIME, npz_path=str(config.POPULARITY_NPZ_026))
    E_2N_smooth = np.concatenate([E_smooth, np.zeros(N)])
    v_smooth = run_power_iteration(M_2N, E_2N_smooth)
    v_smooth_final = v_smooth[:N] + v_smooth[N:]
    v_smooth_final /= v_smooth_final.sum()

    mre_smooth_all = compute_mre(E_smooth, v_smooth_final, N)
    mre_smooth_100 = compute_mre(E_smooth, v_smooth_final, N, top_k=100)
    logger.info(f"  Full MRE:   {mre_smooth_all:.4f}")
    logger.info(f"  Top-100 MRE: {mre_smooth_100:.4f}")

    # --- Cross-comparison: model output vs the OTHER ground truth ---
    logger.info("=" * 60)
    logger.info("CROSS-COMPARISON")
    logger.info("=" * 60)
    cross_raw_vs_smooth = compute_mre(E_smooth, v_raw_final, N)
    cross_raw_vs_smooth_100 = compute_mre(E_smooth, v_raw_final, N, top_k=100)
    cross_smooth_vs_raw = compute_mre(E_raw, v_smooth_final, N)
    cross_smooth_vs_raw_100 = compute_mre(E_raw, v_smooth_final, N, top_k=100)

    logger.info(f"  Raw model vs smoothed GT:    Full={cross_raw_vs_smooth:.4f}, Top-100={cross_raw_vs_smooth_100:.4f}")
    logger.info(f"  Smooth model vs raw GT:      Full={cross_smooth_vs_raw:.4f}, Top-100={cross_smooth_vs_raw_100:.4f}")

    # --- Summary ---
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
