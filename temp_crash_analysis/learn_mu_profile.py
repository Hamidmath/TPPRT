#!/usr/bin/env python3
"""
Learn optimal mu profile for Tuesdays (96 timeframes).

Leave-one-out cross-validation across 4 Tuesdays:
  Sep 4, Sep 11, Sep 18, Sep 25

For each fold:
  - Training: average E_N from 3 Tuesdays
  - Test: held-out Tuesday
  - For each of 96 timeframes: sweep mu, find optimal value
"""

import sys, json, time as _time
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, hstack, vstack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config
from core.pagerank import compute_speed_lane_weights, PARAMS
from core.gpu_backend import power_iteration, BACKEND_NAME
import logging; logging.basicConfig(level=logging.WARNING)

OUT_DIR = Path(__file__).resolve().parent

TUESDAYS = ['2018-09-04', '2018-09-11', '2018-09-18', '2018-09-25']
GAMMA, DAMPING = 0.20, 0.96
MU_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50]

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'figure.dpi': 200})


def load_all():
    with open(config.GRAPH_FILE) as f:
        gd = json.load(f)
    links = list(gd['links'].keys()); N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    loader = np.load(str(config.POPULARITY_NPZ), allow_pickle=True)
    mat = csr_matrix((loader['matrix_data'], loader['matrix_indices'],
                      loader['matrix_indptr']), shape=loader['matrix_shape'])
    times = list(loader['times']); pop_ids = list(loader['link_ids'])
    return gd, links, N, lid_to_idx, mat, times, pop_ids


def compute_demand(mat, times, pop_ids, lid_to_idx, N):
    train_idx = [i for i, ts in enumerate(times)
                 if datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').day <= 15]
    dem = np.zeros(N)
    for ti in train_idx:
        row = mat.getrow(ti)
        for i, v in zip(row.indices, row.data):
            lid = pop_ids[i]
            if lid in lid_to_idx: dem[lid_to_idx[lid]] += float(v)
    s = dem.sum()
    return dem / s if s > 0 else np.ones(N) / N


def extract_E(mat, times, pop_ids, lid_to_idx, N, ts):
    if ts not in times: return None
    ti = times.index(ts); row = mat.getrow(ti)
    E = np.zeros(N)
    for i, v in zip(row.indices, row.data):
        lid = pop_ids[i]
        if lid in lid_to_idx: E[lid_to_idx[lid]] = float(v)
    s = E.sum()
    return E / s if s > 0 else np.ones(N) / N


def build_demand_M(gd, demand, gamma, mu_global):
    links2 = list(gd['links'].keys()); N2 = len(links2)
    idx2 = {lid: i for i, lid in enumerate(links2)}
    beta = PARAMS['beta']; adj = gd.get('adjacency', {})
    up_w = compute_speed_lane_weights(gd, True)
    down_w = compute_speed_lane_weights(gd, False)
    dfac = np.power(demand + 1e-15, gamma) if gamma != 0 else np.ones(N2)

    def bpm(w):
        r, c, d = [], [], []
        for i, lid in enumerate(links2):
            outs = adj.get(lid, [])
            succ = [idx2[s] for s in outs if s in idx2]
            ow = [w[j] * dfac[j] for j in succ]
            ed = gd['links'][lid]
            sw = mu_global * ed.get('length', 100) / max(ed.get('speed', 11.17), 0.1)
            tot = sum(ow) + sw
            if tot > 0:
                if sw > 0: r.append(i); c.append(i); d.append(sw / tot)
                for j, ww in zip(succ, ow): r.append(i); c.append(j); d.append(ww / tot)
        return csr_matrix((d, (r, c)), shape=(N2, N2))

    Pu = bpm(up_w); Pd = bpm(down_w)
    Pu2 = Pu.multiply(1 - beta)
    Td = csr_matrix((np.ones(N2) * beta, (np.arange(N2), np.arange(N2))), shape=(N2, N2))
    Z = csr_matrix((N2, N2))
    return vstack([hstack([Pu2, Td]), hstack([Z, Pd])])


def run_pr(M, E_N, N):
    E_2N = np.concatenate([E_N, np.zeros(N)])
    v_2N = power_iteration(M, E_2N, damping=DAMPING)
    v = v_2N[:N] + v_2N[N:]
    v /= v.sum()
    return v


def main():
    t0 = _time.time()
    print("=" * 70)
    print(f"LEARN MU PROFILE — TUESDAYS (GPU: {BACKEND_NAME})")
    print(f"Tuesdays: {', '.join(TUESDAYS)}")
    print(f"Config: d={DAMPING}, gamma={GAMMA}")
    print("=" * 70)

    gd, links, N, lid_to_idx, mat, times, pop_ids = load_all()
    demand = compute_demand(mat, times, pop_ids, lid_to_idx, N)

    # Build time slot labels
    slots = []
    for h in range(24):
        for m in [0, 15, 30, 45]:
            slots.append(f"{h:02d}:{m:02d}")

    # Cache ALL Tuesday teleportation vectors
    print("  Caching teleportation vectors...")
    E_all = {}  # {date: {slot_idx: E_N}}
    for date in TUESDAYS:
        E_all[date] = {}
        for si, sl in enumerate(slots):
            ts = f"{date} {sl}:00"
            E = extract_E(mat, times, pop_ids, lid_to_idx, N, ts)
            if E is not None:
                E_all[date][si] = E
    print(f"  Cached {sum(len(v) for v in E_all.values())} vectors")

    # Pre-build matrices for all mu values
    print(f"  Building {len(MU_VALUES)} matrices...")
    matrices = {}
    for mu in MU_VALUES:
        matrices[mu] = build_demand_M(gd, demand, GAMMA, mu)
    print("  Done.")

    # ── Leave-one-out cross-validation ───────────────────────────
    # For each fold: test on 1 Tuesday, train on other 3
    # For each slot: find best mu minimizing Overall MRE

    fold_results = []  # list of {slot: best_mu} per fold

    for fold_i, test_date in enumerate(TUESDAYS):
        train_dates = [d for d in TUESDAYS if d != test_date]
        print(f"\n  Fold {fold_i+1}/4: test={test_date}, train={train_dates}")

        fold_mu = np.zeros(96)
        fold_err_base = np.zeros(96)
        fold_err_best = np.zeros(96)

        for si in range(96):
            if si not in E_all[test_date]:
                fold_mu[si] = 20; continue

            # Truth: PageRank on test date with normal matrix (mu=20)
            v_truth = run_pr(matrices[20], E_all[test_date][si], N)

            # Training input: average E_N from train dates
            train_Es = [E_all[d][si] for d in train_dates if si in E_all[d]]
            if not train_Es:
                fold_mu[si] = 20; continue
            E_avg = np.mean(train_Es, axis=0)
            E_avg /= E_avg.sum()

            # Sweep mu
            best_mu, best_mre = 20, 1e9
            err_base = None
            for mu in MU_VALUES:
                v_pred = run_pr(matrices[mu], E_avg, N)
                rel = np.abs(v_truth - v_pred) / (v_truth + 1e-9)
                mre = float(np.mean(rel))
                if mu == 20: err_base = mre
                if mre < best_mre:
                    best_mu, best_mre = mu, mre

            fold_mu[si] = best_mu
            fold_err_base[si] = err_base if err_base else 0
            fold_err_best[si] = best_mre

        fold_results.append({
            'test_date': test_date,
            'mu_profile': fold_mu.tolist(),
            'err_base': fold_err_base.tolist(),
            'err_best': fold_err_best.tolist(),
        })

        # Print summary for this fold
        improved = np.sum(fold_err_best < fold_err_base - 0.001)
        print(f"    Slots improved: {improved}/96")
        print(f"    Avg err base: {np.mean(fold_err_base):.4f}, "
              f"best: {np.mean(fold_err_best):.4f}")

    # ── Aggregate across folds ───────────────────────────────────
    all_mu = np.array([f['mu_profile'] for f in fold_results])  # (4, 96)
    all_err_base = np.array([f['err_base'] for f in fold_results])
    all_err_best = np.array([f['err_best'] for f in fold_results])

    mu_median = np.median(all_mu, axis=0)
    mu_mean = np.mean(all_mu, axis=0)
    mu_std = np.std(all_mu, axis=0)
    err_base_mean = np.mean(all_err_base, axis=0)
    err_best_mean = np.mean(all_err_best, axis=0)

    # ── Figures ──────────────────────────────────────────────────
    print("\n  Generating figures...")
    hours_axis = np.arange(96) / 4  # 0.0, 0.25, ..., 23.75

    # Fig 1: Mu profile with CV error bars
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    ax = axes[0]
    ax.fill_between(hours_axis, mu_mean - mu_std, mu_mean + mu_std,
                    alpha=0.2, color='#1f77b4')
    ax.plot(hours_axis, mu_median, '-', color='#1f77b4', lw=2, label='Median mu (4 folds)')
    ax.plot(hours_axis, mu_mean, '--', color='#d62728', lw=1.5, alpha=0.7, label='Mean mu')
    ax.axhline(20, color='gray', ls=':', lw=1.5, label='Default mu=20')
    ax.set_ylabel('Optimal mu')
    ax.set_title('(A) Learned mu Profile — Tuesdays (Leave-One-Out CV)', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel('Hour of Day')

    # Fig: Per-fold mu profiles
    ax = axes[1]
    colors_fold = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a']
    for i, fr in enumerate(fold_results):
        ax.plot(hours_axis, fr['mu_profile'], '-', color=colors_fold[i],
                lw=1.5, alpha=0.8, label=f'Fold {i+1} (test={fr["test_date"][5:]})')
    ax.axhline(20, color='gray', ls=':', lw=1.5)
    ax.set_ylabel('Optimal mu')
    ax.set_title('(B) Per-Fold mu Profiles', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel('Hour of Day')

    # Fig: MRE comparison
    ax = axes[2]
    ax.plot(hours_axis, err_base_mean, '-', color='#d62728', lw=2, label='Fixed mu=20')
    ax.plot(hours_axis, err_best_mean, '-', color='#2ca02c', lw=2, label='Optimal mu')
    ax.fill_between(hours_axis, err_best_mean,
                    err_base_mean, alpha=0.15, color='#2ca02c',
                    label='Improvement')
    ax.set_ylabel('Overall MRE')
    ax.set_title('(C) Prediction Error: Fixed mu=20 vs Optimal mu', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel('Hour of Day')

    plt.suptitle('Tuesday mu Profile — Cross-Validated (d=0.96, γ=0.20)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'fig_mu_profile_tuesday.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig_mu_profile_tuesday.png")

    # Fig 2: Heatmap — mu across folds and time
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(np.log10(all_mu + 0.001), aspect='auto', cmap='RdYlBu_r',
                   extent=[0, 24, 3.5, -0.5], interpolation='nearest')
    ax.set_yticks(range(4))
    ax.set_yticklabels([f'Fold {i+1}\n({fr["test_date"][5:]})' for i, fr in enumerate(fold_results)])
    ax.set_xlabel('Hour of Day')
    ax.set_xticks(range(0, 25, 2))
    ax.set_title('Optimal mu Heatmap (log10 scale) — Each Row is a CV Fold', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, label='log10(mu)')
    # Add tick labels showing actual mu values
    cbar.set_ticks([np.log10(v) for v in [0.01, 0.1, 1, 5, 20, 50]])
    cbar.set_ticklabels(['0.01', '0.1', '1', '5', '20', '50'])
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'fig_mu_heatmap_tuesday.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  fig_mu_heatmap_tuesday.png")

    # Fig 3: Improvement bar chart by hour
    fig, ax = plt.subplots(figsize=(14, 5))
    hourly_base = [np.mean(err_base_mean[h*4:(h+1)*4]) for h in range(24)]
    hourly_best = [np.mean(err_best_mean[h*4:(h+1)*4]) for h in range(24)]
    hourly_imp = [(b - o) / b * 100 if b > 0.001 else 0
                  for b, o in zip(hourly_base, hourly_best)]
    bars = ax.bar(range(24), hourly_imp, color=['#2ca02c' if v > 0 else '#d62728' for v in hourly_imp],
                  alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('MRE Improvement (%)')
    ax.set_title('Hourly MRE Improvement: Optimal mu vs Fixed mu=20', fontweight='bold')
    ax.set_xticks(range(24))
    ax.axhline(0, color='gray', lw=0.8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'fig_mu_improvement_tuesday.png'), dpi=200)
    plt.close()
    print("  fig_mu_improvement_tuesday.png")

    # ── Save results ─────────────────────────────────────────────
    save_data = {
        'slots': slots,
        'mu_median': mu_median.tolist(),
        'mu_mean': mu_mean.tolist(),
        'mu_std': mu_std.tolist(),
        'err_base_mean': err_base_mean.tolist(),
        'err_best_mean': err_best_mean.tolist(),
        'folds': fold_results,
        'config': {'damping': DAMPING, 'gamma': GAMMA, 'days': TUESDAYS,
                   'mu_values_tested': MU_VALUES},
    }
    with open(OUT_DIR / 'mu_profile_tuesday.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    # ── Print summary ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"SUMMARY")
    print(f"{'─'*60}")
    print(f"  Avg MRE at mu=20:       {np.mean(err_base_mean):.4f}")
    print(f"  Avg MRE at optimal mu:  {np.mean(err_best_mean):.4f}")
    print(f"  Overall improvement:    {(np.mean(err_base_mean)-np.mean(err_best_mean))/np.mean(err_base_mean)*100:+.1f}%")
    print(f"\n  Slots where optimal mu != 20: {np.sum(mu_median != 20)}/96")
    print(f"  Mu range: {mu_median.min():.2f} — {mu_median.max():.2f}")
    print(f"\n  Total time: {_time.time()-t0:.1f}s | Backend: {BACKEND_NAME}")


if __name__ == '__main__':
    main()
