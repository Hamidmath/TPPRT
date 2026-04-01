#!/usr/bin/env python3
"""
Demand-weighted transition matrix optimization.

Key idea: multiply each transition weight by demand[j]^gamma, where demand[j]
is the AVERAGE popularity of link j across all timeframes. This teaches the
matrix about typical traffic demand so its stationary distribution naturally
matches observations — enabling high damping with low MRE.

This is a principled modification: we encode prior knowledge about traffic
demand into the topology, not the per-timeframe teleportation.
"""

import sys, json, random, copy
import time as _time
from pathlib import Path
from itertools import product

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, vstack, hstack
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config
from core.pagerank import (
    compute_speed_lane_weights, run_power_iteration,
    sharpen_teleportation, boost_top_k, PARAMS,
)

logging.basicConfig(level=logging.WARNING)
OUT_DIR = Path(__file__).resolve().parent
SEED = 42


# ── Data loading ────────────────────────────────────────────────

def load_all():
    with open(config.GRAPH_FILE, 'r') as f:
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


def compute_average_demand(mat, times, pop_ids, lid_to_idx, N, n_samples=300):
    """Average teleportation vector across many timeframes → demand prior."""
    random.seed(0)
    sample = random.sample(range(len(times)), min(n_samples, len(times)))
    E_sum = np.zeros(N)
    for t_idx in sample:
        row = mat.getrow(t_idx)
        for i, val in zip(row.indices, row.data):
            lid = pop_ids[i]
            if lid in lid_to_idx:
                E_sum[lid_to_idx[lid]] += float(val)
    s = E_sum.sum()
    if s > 0:
        E_sum /= s
    return E_sum


def cache_E(mat, times, sample, pop_ids, lid_to_idx, N):
    cache = {}
    for ts in sample:
        t_idx = times.index(ts)
        row = mat.getrow(t_idx)
        E = np.zeros(N)
        for i, val in zip(row.indices, row.data):
            lid = pop_ids[i]
            if lid in lid_to_idx:
                E[lid_to_idx[lid]] = float(val)
        s = E.sum()
        E = E / s if s > 0 else np.ones(N) / N
        cache[ts] = E
    return cache


# ── Demand-weighted matrix builder ──────────────────────────────

def build_demand_phase_matrix(graph_data, weights, demand, gamma):
    """Build phase matrix with demand-weighted transitions.
    P(i→j) ∝ weights[j] * demand[j]^gamma
    """
    links = list(graph_data['links'].keys())
    N = len(links)
    id_to_idx = {lid: i for i, lid in enumerate(links)}
    mu = PARAMS.get('mu', 0.0)

    row, col, data = [], [], []
    adj = graph_data.get('adjacency', {})

    # Precompute demand factors
    dfac = np.power(demand + 1e-15, gamma) if gamma != 0 else np.ones(N)

    for i, lid in enumerate(links):
        out_links = adj.get(lid, [])
        successors = [id_to_idx[s] for s in out_links if s in id_to_idx]

        out_w = [weights[j] * dfac[j] for j in successors]

        edge = graph_data['links'][lid]
        self_w = mu * edge.get('length', 100.0) / max(edge.get('speed', 11.17), 0.1) if mu > 0 else 0.0

        total = sum(out_w) + self_w
        if total > 0:
            if self_w > 0:
                row.append(i); col.append(i); data.append(self_w / total)
            for j, w in zip(successors, out_w):
                row.append(i); col.append(j); data.append(w / total)

    return csr_matrix((data, (row, col)), shape=(N, N))


def build_demand_2phase(graph_data, demand, gamma):
    """Build 2N×2N two-phase matrix with demand-weighted transitions."""
    N = len(graph_data['links'])

    up_w = compute_speed_lane_weights(graph_data, is_up_phase=True)
    down_w = compute_speed_lane_weights(graph_data, is_up_phase=False)

    P_up_raw = build_demand_phase_matrix(graph_data, up_w, demand, gamma)
    P_down_raw = build_demand_phase_matrix(graph_data, down_w, demand, gamma)

    beta = PARAMS['beta']
    P_up = P_up_raw.multiply(1.0 - beta)

    T_down = csr_matrix((np.ones(N) * beta, (np.arange(N), np.arange(N))), shape=(N, N))
    zero = csr_matrix((N, N))

    top = hstack([P_up, T_down])
    bot = hstack([zero, P_down_raw])
    return vstack([top, bot])


# ── Evaluation ──────────────────────────────────────────────────

def evaluate(M_2N, E_cache, N, d, tau=1.0, boost=1.0):
    orig_d = PARAMS['damping']
    PARAMS['damping'] = d
    overalls, top100s = [], []
    for E_N in E_cache.values():
        E_t = sharpen_teleportation(E_N, tau)
        E_t = boost_top_k(E_t, PARAMS['top_k'], boost)
        E_2N = np.concatenate([E_t, np.zeros(N)])
        v_2N = run_power_iteration(M_2N, E_2N)
        v = v_2N[:N] + v_2N[N:]
        v /= v.sum()
        rel = np.abs(E_N - v) / (E_N + 1e-9)
        overalls.append(float(np.mean(rel)))
        top100s.append(float(np.mean(rel[np.argsort(E_N)[::-1][:100]])))
    PARAMS['damping'] = orig_d
    return np.mean(overalls), np.std(overalls), np.mean(top100s), np.std(top100s)


# ── Main ────────────────────────────────────────────────────────

def main():
    t0 = _time.time()
    print("=" * 70)
    print("DEMAND-WEIGHTED OPTIMIZATION")
    print("=" * 70)

    gd, links, N, lid_to_idx, mat, times, pop_ids = load_all()
    print(f"  {N:,} links")

    # Compute demand prior
    print("  Computing average demand prior (300 timeframes)...")
    demand = compute_average_demand(mat, times, pop_ids, lid_to_idx, N, 300)
    nz = np.sum(demand > 0)
    print(f"  Demand: {nz:,} nonzero links, max={demand.max():.6f}")

    # Cache evaluation timeframes
    random.seed(SEED)
    s10 = random.sample(times, 10)
    random.seed(SEED)
    s49 = random.sample(times, 49)
    E10 = cache_E(mat, times, s10, pop_ids, lid_to_idx, N)
    E49 = cache_E(mat, times, s49, pop_ids, lid_to_idx, N)

    # ── Phase 1: Grid search (gamma, mu, d) with 10 timeframes ──
    print("\n" + "=" * 70)
    print("PHASE 1: GRID SEARCH (10 timeframes)")
    print("=" * 70)

    gamma_vals = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    mu_vals    = [1, 5, 10, 15, 20]
    d_vals     = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]

    total_builds = len(gamma_vals) * len(mu_vals)
    total_evals = total_builds * len(d_vals)
    print(f"  {len(gamma_vals)} gamma × {len(mu_vals)} mu = "
          f"{total_builds} matrices × {len(d_vals)} d = {total_evals} evals")

    orig = {k: PARAMS[k] for k in PARAMS}
    results = []
    count = 0

    for gamma, mu in product(gamma_vals, mu_vals):
        PARAMS['mu'] = mu
        M = build_demand_2phase(gd, demand, gamma)

        for d in d_vals:
            ov, ov_s, t100, t100_s = evaluate(M, E10, N, d)
            results.append({
                'gamma': gamma, 'mu': mu, 'd': d,
                'overall': ov, 'top100': t100,
                'overall_std': ov_s, 'top100_std': t100_s,
            })

        count += 1
        if count % 10 == 0:
            best = min(r['overall'] for r in results if r['d'] >= 0.91)
            print(f"  {count}/{total_builds} matrices... best OV(d≥0.91)={best:.4f}")

    for k, v in orig.items():
        PARAMS[k] = v

    # Sort
    results.sort(key=lambda r: r['overall'])

    # Filter feasible
    feasible = [r for r in results if r['d'] > 0.90 and r['overall'] <= 0.06]
    feasible.sort(key=lambda r: (-r['d'], r['overall']))

    print(f"\n  Feasible (d>0.90, OV≤0.06): {len(feasible)}")
    print(f"\n  TOP 20 (d > 0.90):")
    print(f"  {'gamma':>6s} {'mu':>5s} {'d':>5s} │ {'Overall':>10s} {'Top-100':>10s}")
    print(f"  {'─'*42}")
    shown = 0
    for r in results:
        if r['d'] > 0.90 and shown < 20:
            print(f"  {r['gamma']:6.1f} {r['mu']:5.0f} {r['d']:5.2f} │ "
                  f"{r['overall']:8.4f}±{r['overall_std']:.3f} "
                  f"{r['top100']:8.4f}±{r['top100_std']:.3f}")
            shown += 1

    # ── Phase 2: Fine-tune best configs ──────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: FINE-TUNE tau & boost on top configs")
    print("=" * 70)

    # Get unique (gamma, mu) pairs from top 15 feasible or overall
    candidates = feasible[:10] if feasible else sorted(
        [r for r in results if r['d'] > 0.90], key=lambda r: r['overall'])[:10]

    seen_mat = set()
    fine_results = []
    tau_vals = [1.0, 1.5, 2.0, 3.0]
    boost_vals = [1.0, 2.0, 5.0]

    for cfg in candidates:
        key = (cfg['gamma'], cfg['mu'])
        if key in seen_mat:
            continue
        seen_mat.add(key)

        PARAMS['mu'] = cfg['mu']
        M = build_demand_2phase(gd, demand, cfg['gamma'])

        for d, tau, boost in product(d_vals, tau_vals, boost_vals):
            if d <= 0.90:
                continue
            ov, ov_s, t100, t100_s = evaluate(M, E10, N, d, tau, boost)
            fine_results.append({
                'gamma': cfg['gamma'], 'mu': cfg['mu'], 'd': d,
                'tau': tau, 'boost': boost,
                'overall': ov, 'top100': t100,
            })

    for k, v in orig.items():
        PARAMS[k] = v

    fine_feasible = [r for r in fine_results if r['overall'] <= 0.06]
    fine_feasible.sort(key=lambda r: (-r['d'], r['overall']))

    fine_results.sort(key=lambda r: r['overall'])

    print(f"  Tested {len(fine_results)} fine combos")
    print(f"  Feasible (d>0.90, OV≤0.06): {len(fine_feasible)}")
    print(f"\n  TOP 15 (d>0.90):")
    print(f"  {'gamma':>6s} {'mu':>5s} {'d':>5s} {'tau':>5s} {'bst':>5s} │ "
          f"{'Overall':>8s} {'Top-100':>8s}")
    print(f"  {'─'*52}")
    for r in fine_results[:15]:
        print(f"  {r['gamma']:6.1f} {r['mu']:5.0f} {r['d']:5.2f} "
              f"{r['tau']:5.1f} {r['boost']:5.1f} │ "
              f"{r['overall']:8.4f} {r['top100']:8.4f}")

    # ── Phase 3: Validate on 49 timeframes ───────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: VALIDATION (49 timeframes)")
    print("=" * 70)

    to_val = []
    # Best from fine-tuning (top 5)
    seen_v = set()
    for r in fine_results[:10]:
        key = (r['gamma'], r['mu'], r['d'], r.get('tau',1), r.get('boost',1))
        if key not in seen_v and len(to_val) < 5:
            to_val.append(r)
            seen_v.add(key)
    # Best feasible (highest d)
    if fine_feasible:
        for r in fine_feasible[:3]:
            key = (r['gamma'], r['mu'], r['d'], r.get('tau',1), r.get('boost',1))
            if key not in seen_v:
                to_val.append(r)
                seen_v.add(key)

    # Baselines
    to_val.append({'gamma': 0, 'mu': 20, 'd': 0.80, 'tau': 1.0, 'boost': 1.0,
                   '_label': 'BASELINE d=0.80'})
    to_val.append({'gamma': 0, 'mu': 20, 'd': 0.90, 'tau': 1.0, 'boost': 1.0,
                   '_label': 'BASELINE d=0.90'})

    validated = []
    for cfg in to_val:
        label = cfg.get('_label', f"g={cfg['gamma']:.1f} mu={cfg['mu']:.0f} "
                        f"d={cfg['d']:.2f} t={cfg.get('tau',1):.1f}")
        PARAMS['mu'] = cfg['mu']
        M = build_demand_2phase(gd, demand, cfg['gamma'])
        ov, ov_s, t100, t100_s = evaluate(
            M, E49, N, cfg['d'], cfg.get('tau', 1.0), cfg.get('boost', 1.0))
        cfg['v_ov'] = ov; cfg['v_ov_std'] = ov_s
        cfg['v_t100'] = t100; cfg['v_t100_std'] = t100_s
        validated.append(cfg)
        mark = ' *** TARGET MET ***' if (ov <= 0.05 and cfg['d'] > 0.90) else ''
        print(f"  {label}: OV={ov:.4f}±{ov_s:.4f}  T100={t100:.4f}±{t100_s:.4f}{mark}")

    for k, v in orig.items():
        PARAMS[k] = v

    # ── Save & Plot ──────────────────────────────────────────────

    # Pareto plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    for r in results:
        c = plt.cm.viridis(r['gamma'] / max(gamma_vals))
        ax.scatter(r['d'], r['overall'], c=[c], s=15, alpha=0.4)
    # Validated
    for v in validated:
        marker = '*' if '_label' in v else 'D'
        ax.scatter(v['d'], v['v_ov'], s=200, marker=marker,
                   edgecolor='black', linewidth=1.5, zorder=10,
                   c='gold' if '_label' not in v else 'red')
        ax.annotate(f"{v['v_ov']:.4f}", (v['d'], v['v_ov']),
                    textcoords='offset points', xytext=(8, 4), fontsize=7)
    ax.axhline(0.05, color='red', ls='--', lw=2, label='Target OV ≤ 0.05')
    ax.axvline(0.90, color='blue', ls='--', lw=2, label='Target d > 0.90')
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(0, max(gamma_vals)))
    plt.colorbar(sm, ax=ax, label='gamma (demand weight)')
    ax.set_xlabel('Damping Factor (d)'); ax.set_ylabel('Overall MRE')
    ax.set_title('(A) Optimization Landscape', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Bar chart
    ax = axes[1]
    labels_v = [v.get('_label', f"g={v['gamma']:.1f}\nd={v['d']:.2f}")
                for v in validated]
    ovs = [v['v_ov'] for v in validated]
    t100s = [v['v_t100'] for v in validated]
    x = np.arange(len(validated)); w = 0.35
    ax.bar(x-w/2, ovs, w, color='#457b9d', alpha=.85, label='Overall MRE')
    ax.bar(x+w/2, t100s, w, color='#e63946', alpha=.85, label='Top-100 MRE')
    ax.axhline(0.05, color='red', ls='--', lw=2)
    ax.set_xticks(x); ax.set_xticklabels(labels_v, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('MRE'); ax.legend(fontsize=9)
    ax.set_title('(B) Validated Configs (49 frames)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Demand-Weighted Transition Matrix Optimization',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = OUT_DIR / 'fig_optimization.png'
    plt.savefig(str(p), dpi=200, bbox_inches='tight'); plt.close()
    print(f"\n  Figure: {p}")

    with open(OUT_DIR / 'optimization_results.json', 'w') as f:
        json.dump({'validated': [{k: v for k, v in c.items()} for c in validated],
                   'best_feasible': fine_feasible[:10] if fine_feasible else []},
                  f, indent=2, default=str)

    print(f"\n  Total time: {_time.time()-t0:.0f}s ({(_time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
