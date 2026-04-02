#!/usr/bin/env python3
"""
Rigorous validation of demand-weighted Two-Phase PageRank.

Validation strategy:
  1. Temporal split: train demand on Sept 1-15, evaluate on Sept 16-30
  2. Day-of-week split: train on Mon-Wed-Fri, test on Tue-Thu-Sat-Sun
  3. Random 80/20 split (5 folds)

For each split, compare:
  - Baseline (gamma=0) at d=0.80 and d=0.96
  - Demand-weighted (gamma=0.10, 0.15, 0.20) at d=0.92, 0.94, 0.96

Runs on GPU (CuPy) if available, falls back to CPU (SciPy).
"""

import sys, json, random, time as _time
from pathlib import Path
from datetime import datetime
from itertools import product
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config
from core.pagerank import (
    compute_speed_lane_weights, PARAMS,
)
import logging
logging.basicConfig(level=logging.WARNING)

OUT_DIR = Path(__file__).resolve().parent

# ── GPU / CPU backend selection ──────────────────────────────────

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    # Test that sparse actually works (needs CUDA libraries)
    _test = cp_sparse.csr_matrix(csr_matrix(np.eye(2)))
    del _test
    GPU_AVAILABLE = True
    print("[BACKEND] GPU detected (CuPy). Using GPU-accelerated power iteration.")
except Exception:
    GPU_AVAILABLE = False
    print("[BACKEND] GPU not usable. Using CPU (SciPy).")


def power_iteration_gpu(M_csr, E_2N_np, damping, tol=1e-6, max_iter=100):
    """GPU-accelerated power iteration using CuPy."""
    M_gpu = cp_sparse.csr_matrix(M_csr)
    E_gpu = cp.asarray(E_2N_np)
    v = E_gpu.copy()
    for k in range(max_iter):
        v_new = damping * M_gpu.T.dot(v) + (1.0 - damping) * E_gpu
        diff = float(cp.sum(cp.abs(v_new - v)))
        v = v_new
        if diff < tol:
            break
    return cp.asnumpy(v)


def power_iteration_cpu(M_csr, E_2N_np, damping, tol=1e-6, max_iter=100):
    """CPU power iteration using SciPy."""
    v = E_2N_np.copy()
    for k in range(max_iter):
        v_new = damping * M_csr.T.dot(v) + (1.0 - damping) * E_2N_np
        diff = np.sum(np.abs(v_new - v))
        v = v_new
        if diff < tol:
            break
    return v


def power_iteration(M_csr, E_2N_np, damping, tol=1e-6, max_iter=100):
    if GPU_AVAILABLE:
        return power_iteration_gpu(M_csr, E_2N_np, damping, tol, max_iter)
    return power_iteration_cpu(M_csr, E_2N_np, damping, tol, max_iter)


# ── Data loading ─────────────────────────────────────────────────

def load_all():
    with open(config.GRAPH_FILE) as f:
        gd = json.load(f)
    links = list(gd['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    loader = np.load(str(config.POPULARITY_NPZ), allow_pickle=True)
    mat = csr_matrix((loader['matrix_data'], loader['matrix_indices'],
                      loader['matrix_indptr']), shape=loader['matrix_shape'])
    times = list(loader['times'])
    pop_ids = list(loader['link_ids'])
    return gd, links, N, lid_to_idx, mat, times, pop_ids


def extract_E(mat, t_idx, pop_ids, lid_to_idx, N):
    row = mat.getrow(t_idx)
    E = np.zeros(N)
    for i, val in zip(row.indices, row.data):
        lid = pop_ids[i]
        if lid in lid_to_idx:
            E[lid_to_idx[lid]] = float(val)
    s = E.sum()
    return E / s if s > 0 else np.ones(N) / N


def compute_demand(mat, time_indices, pop_ids, lid_to_idx, N):
    """Compute average demand from a SET of timeframe indices."""
    dem = np.zeros(N)
    for ti in time_indices:
        row = mat.getrow(ti)
        for i, val in zip(row.indices, row.data):
            lid = pop_ids[i]
            if lid in lid_to_idx:
                dem[lid_to_idx[lid]] += float(val)
    s = dem.sum()
    return dem / s if s > 0 else np.ones(N) / N


# ── Demand-weighted matrix builder ───────────────────────────────

def build_demand_2phase(gd, demand, gamma):
    from scipy.sparse import hstack, vstack
    links = list(gd['links'].keys())
    N = len(links)
    id_to_idx = {lid: i for i, lid in enumerate(links)}
    mu = PARAMS['mu']
    beta = PARAMS['beta']
    adj = gd.get('adjacency', {})

    up_w = compute_speed_lane_weights(gd, is_up_phase=True)
    down_w = compute_speed_lane_weights(gd, is_up_phase=False)
    dfac = np.power(demand + 1e-15, gamma) if gamma != 0 else np.ones(N)

    def build_phase(weights):
        r, c, d = [], [], []
        for i, lid in enumerate(links):
            outs = adj.get(lid, [])
            succ = [id_to_idx[s] for s in outs if s in id_to_idx]
            ow = [weights[j] * dfac[j] for j in succ]
            ed = gd['links'][lid]
            sw = mu * ed.get('length', 100) / max(ed.get('speed', 11.17), 0.1)
            tot = sum(ow) + sw
            if tot > 0:
                if sw > 0:
                    r.append(i); c.append(i); d.append(sw / tot)
                for j, w in zip(succ, ow):
                    r.append(i); c.append(j); d.append(w / tot)
        return csr_matrix((d, (r, c)), shape=(N, N))

    Pu = build_phase(up_w)
    Pd = build_phase(down_w)
    Pu2 = Pu.multiply(1 - beta)
    Td = csr_matrix((np.ones(N) * beta, (np.arange(N), np.arange(N))), shape=(N, N))
    Z = csr_matrix((N, N))
    return vstack([hstack([Pu2, Td]), hstack([Z, Pd])])


# ── Evaluation ───────────────────────────────────────────────────

def evaluate_config(M_2N, E_vectors, N, d):
    """Evaluate a matrix on a list of E_N vectors. Returns mean/std overall & top100 MRE."""
    overalls, top100s = [], []
    for E_N in E_vectors:
        E_2N = np.concatenate([E_N, np.zeros(N)])
        v_2N = power_iteration(M_2N, E_2N, d)
        v = v_2N[:N] + v_2N[N:]
        v /= v.sum()
        rel = np.abs(E_N - v) / (E_N + 1e-9)
        overalls.append(float(np.mean(rel)))
        top100s.append(float(np.mean(rel[np.argsort(E_N)[::-1][:100]])))
    return {
        'overall_mean': np.mean(overalls), 'overall_std': np.std(overalls),
        'top100_mean': np.mean(top100s), 'top100_std': np.std(top100s),
        'n_frames': len(E_vectors),
    }


# ── Splits ───────────────────────────────────────────────────────

def make_temporal_split(times):
    """Train: Sept 1-15, Test: Sept 16-30."""
    train_idx, test_idx = [], []
    for i, ts in enumerate(times):
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        if dt.day <= 15:
            train_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, test_idx


def make_dow_split(times):
    """Train: Mon/Wed/Fri, Test: Tue/Thu/Sat/Sun."""
    train_idx, test_idx = [], []
    for i, ts in enumerate(times):
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        if dt.weekday() in [0, 2, 4]:  # Mon, Wed, Fri
            train_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, test_idx


def make_random_splits(times, n_folds=5, seed=42):
    """5-fold random 80/20 split."""
    rng = random.Random(seed)
    indices = list(range(len(times)))
    folds = []
    for fold in range(n_folds):
        rng.shuffle(indices)
        split = int(0.8 * len(indices))
        folds.append((indices[:split], indices[split:]))
    return folds


# ── Main ─────────────────────────────────────────────────────────

def main():
    t0 = _time.time()
    print("=" * 70)
    print("DEMAND-WEIGHTED PAGERANK — RIGOROUS VALIDATION")
    print("=" * 70)

    gd, links, N, lid_to_idx, mat, times, pop_ids = load_all()
    print(f"  {N:,} links, {len(times):,} timeframes")

    # Configurations to test
    configs = [
        {'gamma': 0.00, 'd': 0.80, 'label': 'Baseline d=0.80'},
        {'gamma': 0.00, 'd': 0.96, 'label': 'No demand d=0.96'},
        {'gamma': 0.10, 'd': 0.91, 'label': 'γ=0.10 d=0.91'},
        {'gamma': 0.10, 'd': 0.94, 'label': 'γ=0.10 d=0.94'},
        {'gamma': 0.15, 'd': 0.94, 'label': 'γ=0.15 d=0.94'},
        {'gamma': 0.15, 'd': 0.96, 'label': 'γ=0.15 d=0.96'},
        {'gamma': 0.20, 'd': 0.94, 'label': 'γ=0.20 d=0.94'},
        {'gamma': 0.20, 'd': 0.96, 'label': 'γ=0.20 d=0.96'},
    ]

    all_results = {}

    # ── Validation 1: Temporal split ─────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION 1: TEMPORAL SPLIT (train Sept 1-15, test Sept 16-30)")
    print("=" * 70)

    train_idx, test_idx = make_temporal_split(times)
    print(f"  Train: {len(train_idx)} frames (Sept 1-15)")
    print(f"  Test:  {len(test_idx)} frames (Sept 16-30)")

    # Sample 49 test frames
    rng = random.Random(42)
    test_sample = rng.sample(test_idx, min(49, len(test_idx)))
    test_E = [extract_E(mat, ti, pop_ids, lid_to_idx, N) for ti in test_sample]

    temporal_results = {}
    for cfg in configs:
        # Compute demand ONLY from training set
        demand = compute_demand(mat, train_idx, pop_ids, lid_to_idx, N)
        M = build_demand_2phase(gd, demand, cfg['gamma'])
        res = evaluate_config(M, test_E, N, cfg['d'])
        temporal_results[cfg['label']] = res
        tag = ' ***' if res['overall_mean'] <= 0.05 and cfg['d'] > 0.90 else ''
        print(f"  {cfg['label']:25s}  OV={res['overall_mean']:.4f}±{res['overall_std']:.4f}  "
              f"T100={res['top100_mean']:.4f}±{res['top100_std']:.4f}{tag}")

    all_results['temporal'] = temporal_results

    # ── Validation 2: Day-of-week split ──────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION 2: DAY-OF-WEEK SPLIT (train Mon/Wed/Fri, test Tue/Thu/Sat/Sun)")
    print("=" * 70)

    train_idx2, test_idx2 = make_dow_split(times)
    print(f"  Train: {len(train_idx2)} frames")
    print(f"  Test:  {len(test_idx2)} frames")

    test_sample2 = rng.sample(test_idx2, min(49, len(test_idx2)))
    test_E2 = [extract_E(mat, ti, pop_ids, lid_to_idx, N) for ti in test_sample2]

    dow_results = {}
    for cfg in configs:
        demand = compute_demand(mat, train_idx2, pop_ids, lid_to_idx, N)
        M = build_demand_2phase(gd, demand, cfg['gamma'])
        res = evaluate_config(M, test_E2, N, cfg['d'])
        dow_results[cfg['label']] = res
        tag = ' ***' if res['overall_mean'] <= 0.05 and cfg['d'] > 0.90 else ''
        print(f"  {cfg['label']:25s}  OV={res['overall_mean']:.4f}±{res['overall_std']:.4f}  "
              f"T100={res['top100_mean']:.4f}±{res['top100_std']:.4f}{tag}")

    all_results['dow'] = dow_results

    # ── Validation 3: 5-fold random split ────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION 3: 5-FOLD RANDOM SPLIT (80% train, 20% test)")
    print("=" * 70)

    folds = make_random_splits(times, n_folds=5, seed=42)
    fold_results = defaultdict(lambda: defaultdict(list))

    for fold_i, (tr_idx, te_idx) in enumerate(folds):
        print(f"\n  Fold {fold_i+1}/5: train={len(tr_idx)}, test={len(te_idx)}")
        te_sample = rng.sample(te_idx, min(49, len(te_idx)))
        te_E = [extract_E(mat, ti, pop_ids, lid_to_idx, N) for ti in te_sample]

        for cfg in configs:
            demand = compute_demand(mat, tr_idx, pop_ids, lid_to_idx, N)
            M = build_demand_2phase(gd, demand, cfg['gamma'])
            res = evaluate_config(M, te_E, N, cfg['d'])
            fold_results[cfg['label']]['overall'].append(res['overall_mean'])
            fold_results[cfg['label']]['top100'].append(res['top100_mean'])

    print("\n  5-FOLD AVERAGED RESULTS:")
    print(f"  {'Config':25s}  {'Overall MRE':>14s}  {'Top-100 MRE':>14s}")
    print(f"  {'─'*57}")
    cv_results = {}
    for cfg in configs:
        lbl = cfg['label']
        ov = fold_results[lbl]['overall']
        t1 = fold_results[lbl]['top100']
        cv_results[lbl] = {
            'overall_mean': np.mean(ov), 'overall_std': np.std(ov),
            'top100_mean': np.mean(t1), 'top100_std': np.std(t1),
        }
        tag = ' ***' if np.mean(ov) <= 0.05 and cfg['d'] > 0.90 else ''
        print(f"  {lbl:25s}  {np.mean(ov):.4f}±{np.std(ov):.4f}  "
              f"  {np.mean(t1):.4f}±{np.std(t1):.4f}{tag}")

    all_results['cv5'] = cv_results

    # ── Summary across all validations ───────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL VALIDATIONS")
    print("=" * 70)
    print(f"\n  {'Config':25s}  {'Temporal':>10s}  {'DoW':>10s}  {'5-Fold':>10s}  {'Avg':>10s}")
    print(f"  {'─'*70}")
    summary = {}
    for cfg in configs:
        lbl = cfg['label']
        t = temporal_results[lbl]['overall_mean']
        d = dow_results[lbl]['overall_mean']
        c = cv_results[lbl]['overall_mean']
        avg = (t + d + c) / 3
        summary[lbl] = {'temporal': t, 'dow': d, 'cv5': c, 'avg': avg, 'd': cfg['d']}
        tag = ' ***' if avg <= 0.05 and cfg['d'] > 0.90 else ''
        print(f"  {lbl:25s}  {t:10.4f}  {d:10.4f}  {c:10.4f}  {avg:10.4f}{tag}")

    all_results['summary'] = summary

    # ── Save ─────────────────────────────────────────────────────
    with open(OUT_DIR / 'validation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results: {OUT_DIR / 'validation_results.json'}")

    # ── Plot ─────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    val_names = ['Temporal\n(Sept 1-15→16-30)', 'Day-of-Week\n(MWF→TThSSu)', '5-Fold CV\n(80/20 random)']
    val_keys = ['temporal', 'dow', 'cv5']

    # Configs to show
    show = [c['label'] for c in configs]
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd',
              '#ff7f0e', '#8c564b', '#e377c2', '#17becf']

    for ax_i, (ax, vname, vkey) in enumerate(zip(axes, val_names, val_keys)):
        vdata = all_results[vkey]
        x = np.arange(len(show))
        ovs = [vdata[lbl]['overall_mean'] for lbl in show]
        t1s = [vdata[lbl]['top100_mean'] for lbl in show]

        w = 0.35
        bars1 = ax.bar(x - w/2, ovs, w, color=[colors[i] for i in range(len(show))],
                       alpha=0.85, edgecolor='black', linewidth=0.5, label='Overall MRE')
        bars2 = ax.bar(x + w/2, t1s, w, color=[colors[i] for i in range(len(show))],
                       alpha=0.45, edgecolor='black', linewidth=0.5, label='Top-100 MRE')

        ax.axhline(0.05, color='red', ls='--', lw=2, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl.replace(' ', '\n') for lbl in show],
                           fontsize=7, rotation=30, ha='right')
        ax.set_ylabel('MRE')
        ax.set_title(vname, fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, v in zip(bars1, ovs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{v:.3f}', ha='center', fontsize=6.5, fontweight='bold')

    axes[0].legend(fontsize=8, loc='upper left')
    plt.suptitle('Demand-Weighted PageRank — Rigorous Validation (Train/Test Splits)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = OUT_DIR / 'fig_validation.png'
    plt.savefig(str(p), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Figure: {p}")

    elapsed = _time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Backend: {'GPU (CuPy)' if GPU_AVAILABLE else 'CPU (SciPy)'}")


if __name__ == '__main__':
    main()
