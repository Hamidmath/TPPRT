#!/usr/bin/env python3
"""
Independent justification for graph-diffusion smoothing.

Four experiments that do NOT use PageRank:
  1. Leave-link-out CV: recover held-out observations
  2. Temporal consistency: frame-to-frame stability
  3. Sparsity simulation: artificial dropout + reconstruction
  4. Bootstrap stability: resampling variance reduction

All experiments evaluate gamma independently of the model.
"""

import sys, json, random, time as _time
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config

OUT_DIR = Path(__file__).resolve().parent
SEED = 42
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'figure.dpi': 200})


def load_data():
    with open(config.GRAPH_FILE) as f:
        gd = json.load(f)
    links = list(gd['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    adj = gd.get('adjacency', {})

    # Build row-stochastic adjacency matrix P_adj
    rows, cols, vals = [], [], []
    for i, lid in enumerate(links):
        outs = adj.get(lid, [])
        succ = [lid_to_idx[s] for s in outs if s in lid_to_idx]
        if succ:
            w = 1.0 / len(succ)
            for j in succ:
                rows.append(i); cols.append(j); vals.append(w)
        else:
            rows.append(i); cols.append(i); vals.append(1.0)
    P_adj = csr_matrix((vals, (rows, cols)), shape=(N, N))

    # Load raw popularity
    loader = np.load(str(config.POPULARITY_RAW_NPZ), allow_pickle=True)
    raw_mat = csr_matrix((loader['matrix_data'], loader['matrix_indices'],
                          loader['matrix_indptr']), shape=loader['matrix_shape'])
    raw_times = list(loader['times'])
    raw_ids = list(loader['link_ids'])

    return gd, links, N, lid_to_idx, P_adj, raw_mat, raw_times, raw_ids


def smooth(c, P_adj, gamma, alpha=0.1):
    """Apply graph-diffusion smoothing: c_smooth = (1-gamma)*c_tilde + gamma*P_adj*c_tilde"""
    c_tilde = c + alpha  # Laplace smoothing
    c_smooth = (1 - gamma) * c_tilde + gamma * P_adj.dot(c_tilde)
    s = c_smooth.sum()
    return c_smooth / s if s > 0 else c_smooth


def extract_raw_count(raw_mat, raw_times, raw_ids, lid_to_idx, N, t_idx):
    """Extract raw count vector for a timeframe."""
    row = raw_mat.getrow(t_idx)
    c = np.zeros(N)
    for i, val in zip(row.indices, row.data):
        lid = raw_ids[i]
        if lid in lid_to_idx:
            c[lid_to_idx[lid]] = float(val)
    return c


def main():
    t0 = _time.time()
    print("=" * 70)
    print("SMOOTHING JUSTIFICATION — INDEPENDENT OF PAGERANK")
    print("=" * 70)

    gd, links, N, lid_to_idx, P_adj, raw_mat, raw_times, raw_ids = load_data()
    print(f"  N={N:,} links, {len(raw_times):,} timeframes")

    gamma_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.26, 0.30, 0.40, 0.50, 0.60]

    random.seed(SEED)
    sample_frames = random.sample(range(len(raw_times)), 30)

    # ── Experiment 1: Leave-Link-Out Cross-Validation ────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: LEAVE-LINK-OUT CV")
    print("  Remove each observed link, smooth, check recovery")
    print("=" * 70)

    llo_results = {g: [] for g in gamma_values}

    for fi, t_idx in enumerate(sample_frames[:15]):
        c_raw = extract_raw_count(raw_mat, raw_times, raw_ids, lid_to_idx, N, t_idx)
        observed = np.where(c_raw > 0)[0]
        if len(observed) < 50:
            continue

        # Sample 200 observed links to test (for speed)
        test_links = random.sample(list(observed), min(200, len(observed)))

        for gamma in gamma_values:
            errors = []
            for li in test_links:
                # Remove this link
                c_dropped = c_raw.copy()
                true_val = c_dropped[li]
                c_dropped[li] = 0

                # Smooth
                c_sm = smooth(c_dropped, P_adj, gamma)

                # The smoothed value at li should recover some signal
                # Compare: smoothed prediction vs true proportion
                true_prop = true_val / c_raw.sum()
                pred_prop = c_sm[li]

                if true_prop > 1e-12:
                    errors.append(abs(pred_prop - true_prop) / true_prop)

            if errors:
                llo_results[gamma].append(np.mean(errors))

        if (fi + 1) % 5 == 0:
            print(f"  {fi+1}/15 frames done")

    print(f"\n  Leave-Link-Out Recovery Error (lower = better):")
    print(f"  {'gamma':>6s}  {'Mean Error':>12s}  {'Std':>10s}")
    print(f"  {'─'*32}")
    best_gamma_llo = min(gamma_values, key=lambda g: np.mean(llo_results[g]) if llo_results[g] else 1e9)
    for g in gamma_values:
        vals = llo_results[g]
        if vals:
            marker = ' ◄── best' if g == best_gamma_llo else ''
            print(f"  {g:6.2f}  {np.mean(vals):12.4f}  {np.std(vals):10.4f}{marker}")

    # ── Experiment 2: Temporal Consistency ────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: TEMPORAL CONSISTENCY")
    print("  Adjacent 15-min frames should be similar")
    print("=" * 70)

    # Use consecutive frame pairs
    consecutive_pairs = [(i, i+1) for i in range(0, len(raw_times)-1, 4)]
    random.shuffle(consecutive_pairs)
    consecutive_pairs = consecutive_pairs[:100]

    temp_results = {g: [] for g in gamma_values}

    for t1, t2 in consecutive_pairs:
        c1 = extract_raw_count(raw_mat, raw_times, raw_ids, lid_to_idx, N, t1)
        c2 = extract_raw_count(raw_mat, raw_times, raw_ids, lid_to_idx, N, t2)

        if c1.sum() < 10 or c2.sum() < 10:
            continue

        for gamma in gamma_values:
            s1 = smooth(c1, P_adj, gamma)
            s2 = smooth(c2, P_adj, gamma)

            # Pearson correlation between consecutive smoothed frames
            mask = (s1 > 1e-12) | (s2 > 1e-12)
            if np.sum(mask) > 100:
                corr = np.corrcoef(s1[mask], s2[mask])[0, 1]
                temp_results[gamma].append(corr)

    print(f"\n  Temporal Correlation (higher = more consistent):")
    print(f"  {'gamma':>6s}  {'Mean Corr':>12s}  {'Std':>10s}")
    print(f"  {'─'*32}")
    for g in gamma_values:
        vals = temp_results[g]
        if vals:
            print(f"  {g:6.2f}  {np.mean(vals):12.6f}  {np.std(vals):10.6f}")

    # ── Experiment 3: Sparsity Simulation ────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: SPARSITY SIMULATION")
    print("  Artificially drop links, smooth, measure reconstruction")
    print("=" * 70)

    dropout_rates = [0.3, 0.5, 0.7, 0.9]
    sparsity_results = {g: {dr: [] for dr in dropout_rates} for g in gamma_values}

    for fi, t_idx in enumerate(sample_frames[:20]):
        c_raw = extract_raw_count(raw_mat, raw_times, raw_ids, lid_to_idx, N, t_idx)
        observed = np.where(c_raw > 0)[0]
        if len(observed) < 50:
            continue

        c_prop = c_raw / c_raw.sum()  # true proportions

        for dr in dropout_rates:
            # Drop links
            keep = random.sample(list(observed), int(len(observed) * (1 - dr)))
            c_dropped = np.zeros(N)
            for i in keep:
                c_dropped[i] = c_raw[i]

            dropped_links = set(observed) - set(keep)

            for gamma in gamma_values:
                c_sm = smooth(c_dropped, P_adj, gamma)

                # Measure reconstruction on DROPPED links only
                errors = []
                for li in dropped_links:
                    if c_prop[li] > 1e-12:
                        errors.append(abs(c_sm[li] - c_prop[li]) / c_prop[li])

                if errors:
                    sparsity_results[gamma][dr].append(np.mean(errors))

    print(f"\n  Reconstruction Error on Dropped Links:")
    print(f"  {'gamma':>6s}", end='')
    for dr in dropout_rates:
        print(f"  {f'drop={dr:.0%}':>12s}", end='')
    print()
    print(f"  {'─'*60}")
    for g in gamma_values:
        print(f"  {g:6.2f}", end='')
        for dr in dropout_rates:
            vals = sparsity_results[g][dr]
            if vals:
                print(f"  {np.mean(vals):12.4f}", end='')
            else:
                print(f"  {'N/A':>12s}", end='')
        print()

    # ── Experiment 4: Bootstrap Stability ─────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: BOOTSTRAP STABILITY")
    print("  Resample GPS routes, measure variance of estimates")
    print("=" * 70)

    # For a few timeframes, bootstrap the raw counts
    bootstrap_results = {g: [] for g in gamma_values}

    for fi, t_idx in enumerate(sample_frames[:10]):
        c_raw = extract_raw_count(raw_mat, raw_times, raw_ids, lid_to_idx, N, t_idx)
        if c_raw.sum() < 20:
            continue

        total_count = int(c_raw.sum())
        # Create "route" list from counts
        route_links = []
        for i in range(N):
            route_links.extend([i] * int(c_raw[i]))

        for gamma in gamma_values:
            # Bootstrap: resample routes 20 times
            boot_props = []
            for _ in range(20):
                resampled = random.choices(route_links, k=total_count)
                c_boot = np.zeros(N)
                for i in resampled:
                    c_boot[i] += 1
                s_boot = smooth(c_boot, P_adj, gamma)
                boot_props.append(s_boot)

            # Measure coefficient of variation across bootstraps
            boot_arr = np.array(boot_props)  # (20, N)
            mean_prop = boot_arr.mean(axis=0)
            std_prop = boot_arr.std(axis=0)

            # CV on observed links only
            observed = mean_prop > 1e-10
            if np.sum(observed) > 0:
                cv = np.mean(std_prop[observed] / mean_prop[observed])
                bootstrap_results[gamma].append(cv)

    print(f"\n  Bootstrap Coefficient of Variation (lower = more stable):")
    print(f"  {'gamma':>6s}  {'Mean CV':>12s}  {'Std':>10s}")
    print(f"  {'─'*32}")
    for g in gamma_values:
        vals = bootstrap_results[g]
        if vals:
            print(f"  {g:6.2f}  {np.mean(vals):12.4f}  {np.std(vals):10.4f}")

    # ── Figures ──────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Leave-link-out
    ax = axes[0, 0]
    means = [np.mean(llo_results[g]) for g in gamma_values]
    stds = [np.std(llo_results[g]) for g in gamma_values]
    ax.errorbar(gamma_values, means, yerr=stds, fmt='-o', color='#e63946',
                capsize=4, lw=2, ms=8)
    ax.axvline(0.26, color='blue', ls='--', lw=1.5, alpha=0.7, label='γ=0.26')
    best_idx = np.argmin(means)
    ax.scatter([gamma_values[best_idx]], [means[best_idx]], s=150, c='gold',
               edgecolor='black', zorder=10, label=f'Best: γ={gamma_values[best_idx]:.2f}')
    ax.set_xlabel('γ (smoothing strength)')
    ax.set_ylabel('Leave-Link-Out Recovery Error')
    ax.set_title('(A) Leave-Link-Out CV: Can Smoothing Recover\nHeld-Out Observations?',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel B: Temporal consistency
    ax = axes[0, 1]
    means_t = [np.mean(temp_results[g]) for g in gamma_values]
    ax.plot(gamma_values, means_t, '-s', color='#2a9d8f', lw=2, ms=8)
    ax.axvline(0.26, color='blue', ls='--', lw=1.5, alpha=0.7, label='γ=0.26')
    ax.set_xlabel('γ (smoothing strength)')
    ax.set_ylabel('Consecutive-Frame Pearson Correlation')
    ax.set_title('(B) Temporal Consistency: Do Adjacent\nFrames Become More Similar?',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel C: Sparsity simulation
    ax = axes[1, 0]
    colors_dr = ['#457b9d', '#e63946', '#2a9d8f', '#e9c46a']
    for di, dr in enumerate(dropout_rates):
        means_s = [np.mean(sparsity_results[g][dr]) if sparsity_results[g][dr] else np.nan
                   for g in gamma_values]
        ax.plot(gamma_values, means_s, '-o', color=colors_dr[di], lw=2, ms=6,
                label=f'{dr:.0%} dropout')
    ax.axvline(0.26, color='blue', ls='--', lw=1.5, alpha=0.7, label='γ=0.26')
    ax.set_xlabel('γ (smoothing strength)')
    ax.set_ylabel('Reconstruction Error on Dropped Links')
    ax.set_title('(C) Sparsity Simulation: Reconstruction\nAccuracy at Various Dropout Rates',
                 fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel D: Bootstrap stability
    ax = axes[1, 1]
    means_b = [np.mean(bootstrap_results[g]) for g in gamma_values]
    ax.plot(gamma_values, means_b, '-D', color='#e9c46a', lw=2, ms=8)
    ax.axvline(0.26, color='blue', ls='--', lw=1.5, alpha=0.7, label='γ=0.26')
    ax.set_xlabel('γ (smoothing strength)')
    ax.set_ylabel('Bootstrap Coefficient of Variation')
    ax.set_title('(D) Bootstrap Stability: Are Estimates\nRobust to Resampling?', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle('Independent Justification for Graph-Diffusion Smoothing\n'
                 '(No PageRank involved — data-only experiments)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'fig_smoothing_justification.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  fig_smoothing_justification.png")

    # Save results
    save_data = {
        'gamma_values': gamma_values,
        'llo': {str(g): {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                for g, v in llo_results.items() if v},
        'temporal': {str(g): {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                     for g, v in temp_results.items() if v},
        'sparsity': {str(g): {str(dr): float(np.mean(v)) if v else None
                               for dr, v in drs.items()}
                     for g, drs in sparsity_results.items()},
        'bootstrap': {str(g): {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                      for g, v in bootstrap_results.items() if v},
        'best_gamma_llo': float(best_gamma_llo),
    }
    with open(OUT_DIR / 'smoothing_justification_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n  Total time: {_time.time()-t0:.1f}s")
    print(f"\n  KEY FINDING: Best gamma from leave-link-out CV = {best_gamma_llo:.2f}")
    print(f"  (Compare to gamma=0.26 chosen by downstream PageRank CV)")


if __name__ == '__main__':
    main()
