#!/usr/bin/env python3
"""
Conditional Disruption Modeling — Speed-Sensitive Experiment
============================================================
With α_s > 0 activated, speed reductions on crash links propagate
through transition WEIGHTS (not just self-loops), making crash
conditioning effective.

Levels:
  L0 — Unconditioned : PageRank(M_std,   E_sept13)
  L1 — Matrix-cond.  : PageRank(M_crash, E_sept13)
  L2 — Tele-cond.    : PageRank(M_std,   E_crash)
  L3 — Full cond.    : PageRank(M_crash, E_crash)

  Truth: PageRank(M_std, E_sept20)

Optimized: NPZ loaded once, all E_N cached, runtime ~15 seconds.
"""

import sys
import json
import copy
import time as _time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config
from core.pagerank import (
    build_two_phase_matrix, run_power_iteration,
    sharpen_teleportation, boost_top_k, PARAMS,
)

OUT_DIR = Path(__file__).resolve().parent

# ── Crash metadata ──────────────────────────────────────────────────
CRASH_LAT, CRASH_LON = 40.524, -111.893
CRASH_DATE    = '2018-09-20'
BASELINE_DATE = '2018-09-13'

CORRIDOR_RADIUS    = 0.06
FREEWAY_SPEED      = 25.0
ARTERIAL_SPEED_MIN = 11.0
ARTERIAL_SPEED_MAX = 25.0

# Speed-sensitive parameters for crash analysis
ALPHA_S_CRASH = 1.0    # speed sensitivity
ALPHA_L_CRASH = 0.5    # lane sensitivity

# Crash conditioning: link attribute modifications
CRASH_SITE_SPEED  = 2.0     # m/s (near standstill)
CRASH_SITE_LANES  = 1.0     # emergency shoulder only
UPSTREAM_SPD_FAC  = 0.30    # 70% speed reduction
UPSTREAM_LN_FAC   = 0.50    # 50% lane reduction
NB_SPD_FAC        = 0.70    # 30% speed reduction (rubbernecking)

# Teleportation conditioning factors
TELE_CRASH_FAC    = 0.20    # 80% reduction at crash site
TELE_UPSTREAM_FAC = 0.50    # 50% reduction upstream
TELE_NB_FAC       = 0.80    # 20% reduction NB

SNAPSHOT_TIMES = ['06:00', '07:00', '07:30', '08:00', '09:00', '10:00', '11:00']
CRASH_WINDOW   = {'07:30', '08:00', '09:00', '10:00'}

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'figure.dpi': 200,
})


# ====================================================================
# 1.  Parse network & classify links
# ====================================================================

def parse_network():
    tree = ET.parse(str(config.NETWORK_XML))
    root = tree.getroot()
    nodes = {}
    for n in root.find('nodes'):
        nodes[n.get('id')] = (float(n.get('y')), float(n.get('x')))
    links = {}
    for l in root.find('links'):
        links[l.get('id')] = {
            'from': l.get('from'), 'to': l.get('to'),
            'length': float(l.get('length')),
            'freespeed': float(l.get('freespeed')),
            'lanes': float(l.get('permlanes')),
        }
    return nodes, links


def classify_links(nodes, xml_links):
    crash_site_sb, upstream_sb = [], []
    all_sb, all_nb, surface = [], [], []

    for lid, ld in xml_links.items():
        fn, tn = ld['from'], ld['to']
        if fn not in nodes or tn not in nodes:
            continue
        lat1, lon1 = nodes[fn]
        lat2, lon2 = nodes[tn]
        mid_lat = (lat1 + lat2) / 2
        dist = ((mid_lat - CRASH_LAT)**2 +
                ((lon1+lon2)/2 - CRASH_LON)**2)**0.5

        if ld['freespeed'] >= FREEWAY_SPEED and dist < CORRIDOR_RADIUS:
            is_sb = lat2 < lat1
            if is_sb:
                all_sb.append(lid)
                if dist < 0.025:
                    crash_site_sb.append(lid)
                elif mid_lat > CRASH_LAT:
                    upstream_sb.append(lid)
            else:
                all_nb.append(lid)
        elif (dist < 0.04
              and ARTERIAL_SPEED_MIN <= ld['freespeed'] < ARTERIAL_SPEED_MAX
              and ld['lanes'] >= 2):
            surface.append(lid)

    print(f"  SB: {len(all_sb)} (crash-site={len(crash_site_sb)}, "
          f"upstream={len(upstream_sb)}), NB: {len(all_nb)}, "
          f"Surface: {len(surface)}")
    return crash_site_sb, upstream_sb, all_sb, all_nb, surface


# ====================================================================
# 2.  Load NPZ once & cache teleportation vectors
# ====================================================================

def load_and_cache_teleportation(graph_data, needed_times):
    """Load NPZ once, extract E_N for all needed timestamps."""
    t0 = _time.time()
    loader = np.load(str(config.POPULARITY_NPZ), allow_pickle=True)
    matrix = csr_matrix((
        loader['matrix_data'], loader['matrix_indices'],
        loader['matrix_indptr']), shape=loader['matrix_shape'])
    times = list(loader['times'])
    pop_ids = list(loader['link_ids'])

    link_list = list(graph_data['links'].keys())
    N = len(link_list)
    lid_to_idx = {lid: i for i, lid in enumerate(link_list)}

    cache = {}
    for ts in needed_times:
        if ts not in times:
            print(f"  WARNING: {ts} not in NPZ, using {times[0]}")
            ts = times[0]
        t_idx = times.index(ts)
        row = matrix.getrow(t_idx)
        E = np.zeros(N)
        for i, val in zip(row.indices, row.data):
            lid = pop_ids[i]
            if lid in lid_to_idx:
                E[lid_to_idx[lid]] = float(val)
        s = E.sum()
        E = E / s if s > 0 else np.ones(N) / N
        cache[ts] = E

    print(f"  Cached {len(cache)} teleportation vectors in "
          f"{_time.time()-t0:.1f}s")
    return cache


# ====================================================================
# 3.  Build matrices
# ====================================================================

def build_crash_graph(graph_data, crash_ids, upstream_ids, nb_ids):
    """Deep-copy graph data and modify crash-affected link attributes."""
    gd = copy.deepcopy(graph_data)
    counts = {'crash': 0, 'upstream': 0, 'nb': 0}

    for lid in crash_ids:
        if lid in gd['links']:
            gd['links'][lid]['speed'] = CRASH_SITE_SPEED
            gd['links'][lid]['lanes'] = CRASH_SITE_LANES
            counts['crash'] += 1

    for lid in upstream_ids:
        if lid in gd['links']:
            gd['links'][lid]['speed'] *= UPSTREAM_SPD_FAC
            gd['links'][lid]['lanes'] = max(
                1.0, gd['links'][lid].get('lanes', 1.0) * UPSTREAM_LN_FAC)
            counts['upstream'] += 1

    for lid in nb_ids:
        if lid in gd['links']:
            gd['links'][lid]['speed'] *= NB_SPD_FAC
            counts['nb'] += 1

    print(f"  Modified: {counts}")
    return gd


# ====================================================================
# 4.  Teleportation conditioning
# ====================================================================

def condition_teleportation(E_N, link_list,
                            crash_ids, upstream_ids, nb_ids, surface_ids):
    lid_to_idx = {lid: i for i, lid in enumerate(link_list)}
    E = E_N.copy()
    removed = 0.0

    for lid, fac in [(crash_ids, TELE_CRASH_FAC),
                     (upstream_ids, TELE_UPSTREAM_FAC),
                     (nb_ids, TELE_NB_FAC)]:
        for l in lid:
            if l in lid_to_idx:
                i = lid_to_idx[l]
                delta = E[i] * (1.0 - fac)
                E[i] *= fac
                removed += delta

    sf_idx = [lid_to_idx[l] for l in surface_ids if l in lid_to_idx]
    sf_sum = sum(E[i] for i in sf_idx)
    if sf_sum > 0 and removed > 0:
        for i in sf_idx:
            E[i] += removed * (E[i] / sf_sum)
    E /= E.sum()
    return E


# ====================================================================
# 5.  Fast PageRank wrapper
# ====================================================================

def run_pr(M_2N, E_N, N):
    E_tele = sharpen_teleportation(E_N, PARAMS['tau'])
    E_tele = boost_top_k(E_tele, PARAMS['top_k'], PARAMS['top_k_boost'])
    E_2N = np.concatenate([E_tele, np.zeros(N)])
    v_2N = run_power_iteration(M_2N, E_2N)
    v = v_2N[:N] + v_2N[N:]
    v /= v.sum()
    return v


# ====================================================================
# 6.  Experiment
# ====================================================================

def run_experiment(M_std, M_crash, E_cache, link_list, N,
                   crash_ids, upstream_ids, nb_ids, surface_ids,
                   sb_ids):
    lid_to_idx = {lid: i for i, lid in enumerate(link_list)}
    corridors = {
        'I-15 SB': [lid_to_idx[l] for l in sb_ids     if l in lid_to_idx],
        'I-15 NB': [lid_to_idx[l] for l in nb_ids      if l in lid_to_idx],
        'Surface': [lid_to_idx[l] for l in surface_ids  if l in lid_to_idx],
    }

    rows = []
    for st in SNAPSHOT_TIMES:
        crash_ts = f"{CRASH_DATE} {st}:00"
        base_ts  = f"{BASELINE_DATE} {st}:00"
        in_window = st in CRASH_WINDOW

        E_truth = E_cache[crash_ts]
        E_base  = E_cache[base_ts]

        v_truth = run_pr(M_std, E_truth, N)
        v_L0    = run_pr(M_std, E_base, N)
        v_L1    = run_pr(M_crash, E_base, N)

        if in_window:
            E_cond = condition_teleportation(
                E_base, link_list,
                crash_ids, upstream_ids, nb_ids, surface_ids)
            v_L2 = run_pr(M_std, E_cond, N)
            v_L3 = run_pr(M_crash, E_cond, N)
        else:
            v_L2 = v_L0
            v_L3 = v_L1

        for cname, cidx in corridors.items():
            tm = sum(v_truth[i] for i in cidx)
            m0 = sum(v_L0[i]   for i in cidx)
            m1 = sum(v_L1[i]   for i in cidx)
            m2 = sum(v_L2[i]   for i in cidx)
            m3 = sum(v_L3[i]   for i in cidx)

            e0 = abs(m0 - tm) / (tm + 1e-15) * 100
            e1 = abs(m1 - tm) / (tm + 1e-15) * 100
            e2 = abs(m2 - tm) / (tm + 1e-15) * 100
            e3 = abs(m3 - tm) / (tm + 1e-15) * 100

            rows.append({
                'time': st, 'corridor': cname, 'in_window': in_window,
                'truth': tm, 'L0': m0, 'L1': m1, 'L2': m2, 'L3': m3,
                'e0': e0, 'e1': e1, 'e2': e2, 'e3': e3,
            })

        tag = ' [CRASH]' if in_window else ''
        sb_r = next(r for r in rows[-3:] if r['corridor'] == 'I-15 SB')
        if in_window:
            print(f"  {st}{tag}  SB: L0={sb_r['e0']:5.1f}%  "
                  f"L1={sb_r['e1']:5.1f}%  L2={sb_r['e2']:5.1f}%  "
                  f"L3={sb_r['e3']:5.1f}%")
        else:
            print(f"  {st}{tag}  SB: L0={sb_r['e0']:5.1f}%")

    return rows


# ====================================================================
# 7.  Figures
# ====================================================================

def fig6_mre_over_time(results):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    corridors = ['I-15 SB', 'I-15 NB', 'Surface']
    styles = {
        'e0': ('Unconditioned',       '#d62728', '-o'),
        'e1': ('Matrix-conditioned',  '#ff7f0e', '-v'),
        'e2': ('Tele-conditioned',    '#2ca02c', '-s'),
        'e3': ('Fully conditioned',   '#1f77b4', '-D'),
    }

    for ax, cname in zip(axes, corridors):
        cd = [r for r in results if r['corridor'] == cname]
        times = [r['time'] for r in cd]
        x = np.arange(len(times))

        for key, (lab, col, fmt) in styles.items():
            vals = [r[key] for r in cd]
            ax.plot(x, vals, fmt, color=col, lw=2, ms=7, label=lab)

        cw = [i for i, r in enumerate(cd) if r['in_window']]
        if cw:
            ax.axvspan(min(cw)-0.4, max(cw)+0.4, alpha=0.08, color='red',
                       label='Crash window')
        ax.set_xticks(x); ax.set_xticklabels(times, rotation=45)
        ax.set_xlabel('Time of Day'); ax.set_ylabel('Error (%)')
        ax.set_title(cname, fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Corridor Prediction Error: Conditioning Levels '
        r'($\alpha_s$=1.0, $\alpha_l$=0.5)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = OUT_DIR / 'fig6_conditional_mre.png'
    plt.savefig(str(p), dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved {p}")


def fig7_mass_at_peak(results):
    sb = [r for r in results if r['corridor'] == 'I-15 SB' and r['in_window']]
    pk = max(sb, key=lambda r: r['e0'])['time']
    peak = [r for r in results if r['time'] == pk]

    fig, ax = plt.subplots(figsize=(13, 6))
    names = [r['corridor'] for r in peak]
    s = 1e4
    x = np.arange(len(names)); w = 0.18

    for i, (key, lab, col) in enumerate([
        ('truth', 'Truth (Sept 20)',    'steelblue'),
        ('L0',    'Unconditioned',      '#d62728'),
        ('L1',    'Matrix-conditioned', '#ff7f0e'),
        ('L3',    'Fully conditioned',  '#1f77b4'),
    ]):
        vals = [r[key] * s for r in peak]
        ax.bar(x + (i-1.5)*w, vals, w, color=col, alpha=.85, label=lab)

    for i, r in enumerate(peak):
        y = max(r['truth'], r['L0'], r['L1'], r['L3']) * s * 1.05
        ax.annotate(f'L0:{r["e0"]:.0f}%  L1:{r["e1"]:.0f}%  '
                    f'L3:{r["e3"]:.0f}%',
                    (i, y), ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('PageRank Mass (×10⁻⁴)')
    ax.set_title(f'Corridor Mass at Peak Disruption ({pk})',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    p = OUT_DIR / 'fig7_mass_redistribution.png'
    plt.savefig(str(p), dpi=200); plt.close()
    print(f"  Saved {p}")
    return pk


def fig8_summary(results):
    corridors = ['I-15 SB', 'I-15 NB', 'Surface']
    avgs = {}
    for cn in corridors:
        cw = [r for r in results
              if r['corridor'] == cn and r['in_window']]
        avgs[cn] = {k: np.mean([r[k] for r in cw])
                    for k in ['e0','e1','e2','e3']}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(corridors)); w = 0.18

    ax = axes[0]
    for i, (key, lab, col) in enumerate([
        ('e0', 'Unconditioned',      '#d62728'),
        ('e1', 'Matrix-cond.',       '#ff7f0e'),
        ('e2', 'Tele-cond.',         '#2ca02c'),
        ('e3', 'Full',               '#1f77b4'),
    ]):
        vals = [avgs[cn][key] for cn in corridors]
        ax.bar(x + (i-1.5)*w, vals, w, color=col, alpha=.85, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(corridors)
    ax.set_ylabel('Mean Error (%)')
    ax.set_title('(A)  Avg Error During Crash Window', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    for i, (key, lab, col) in enumerate([
        ('e1', 'Matrix', '#ff7f0e'),
        ('e2', 'Tele',   '#2ca02c'),
        ('e3', 'Full',   '#1f77b4'),
    ]):
        imp = [(avgs[cn]['e0'] - avgs[cn][key]) / avgs[cn]['e0'] * 100
               if avgs[cn]['e0'] > 0 else 0 for cn in corridors]
        bars = ax.bar(x + (i-1)*w, imp, w, color=col, alpha=.85, label=lab)
        for b, v in zip(bars, imp):
            if abs(v) > 0.5:
                ax.text(b.get_x() + b.get_width()/2,
                        b.get_height() + (0.5 if v >= 0 else -2),
                        f'{v:+.1f}%', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(corridors)
    ax.set_ylabel('Error Reduction vs Uncond. (%)')
    ax.set_title('(B)  Improvement from Conditioning', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='gray', lw=0.8)

    plt.suptitle(
        'Conditional Disruption Modeling — Summary '
        r'($\alpha_s$=1.0, $\alpha_l$=0.5)',
        fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = OUT_DIR / 'fig8_improvement_summary.png'
    plt.savefig(str(p), dpi=200, bbox_inches='tight'); plt.close()
    print(f"  Saved {p}")
    return avgs


# ====================================================================
# 8.  Main
# ====================================================================

def main():
    t_start = _time.time()
    print("=" * 70)
    print("CONDITIONAL DISRUPTION (speed-sensitive, α_s=1.0, α_l=0.5)")
    print("=" * 70)

    nodes, xml_links = parse_network()
    crash_ids, upstream_ids, all_sb, all_nb, surface = \
        classify_links(nodes, xml_links)

    # Load graph
    with open(config.GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)
    link_list = list(graph_data['links'].keys())
    N = len(link_list)
    print(f"  Graph: {N:,} links")

    # Cache ALL teleportation vectors (single NPZ load)
    needed = []
    for st in SNAPSHOT_TIMES:
        needed.append(f"{CRASH_DATE} {st}:00")
        needed.append(f"{BASELINE_DATE} {st}:00")
    E_cache = load_and_cache_teleportation(graph_data, needed)

    # Activate speed/lane sensitivity
    orig_as, orig_al = PARAMS['alpha_s'], PARAMS['alpha_l']
    PARAMS['alpha_s'] = ALPHA_S_CRASH
    PARAMS['alpha_l'] = ALPHA_L_CRASH

    # Build standard matrix (speed-sensitive)
    print(f"\n  Building M_std (α_s={ALPHA_S_CRASH}, α_l={ALPHA_L_CRASH})...")
    M_std = build_two_phase_matrix(graph_data)

    # Build crash-conditioned matrix (modified link attributes)
    print("  Building M_crash (modified crash-link speeds/lanes)...")
    gd_crash = build_crash_graph(
        graph_data, crash_ids, upstream_ids, all_nb)
    M_crash = build_two_phase_matrix(gd_crash)

    # Restore params
    PARAMS['alpha_s'] = orig_as
    PARAMS['alpha_l'] = orig_al

    # Run experiment
    print(f"\n  Running predictions...")
    results = run_experiment(
        M_std, M_crash, E_cache, link_list, N,
        crash_ids, upstream_ids, all_nb, surface, all_sb)

    # Figures
    print("\n  Generating figures...")
    fig6_mre_over_time(results)
    fig7_mass_at_peak(results)
    avgs = fig8_summary(results)

    # Save JSON
    with open(OUT_DIR / 'conditional_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Full table
    print("\n" + "=" * 70)
    print("FULL RESULTS")
    print("=" * 70)
    print(f"{'Time':>6s} {'Corr':>10s} │ {'Truth':>10s} {'L0':>10s} "
          f"{'L1':>10s} {'L2':>10s} {'L3':>10s} │ "
          f"{'E0':>6s} {'E1':>6s} {'E2':>6s} {'E3':>6s}")
    print("─" * 100)
    for r in results:
        if r['in_window']:
            print(f"{r['time']:>6s} {r['corridor']:>10s} │ "
                  f"{r['truth']:10.6f} {r['L0']:10.6f} "
                  f"{r['L1']:10.6f} {r['L2']:10.6f} {r['L3']:10.6f} │ "
                  f"{r['e0']:5.1f}% {r['e1']:5.1f}% "
                  f"{r['e2']:5.1f}% {r['e3']:5.1f}%")
        else:
            print(f"{r['time']:>6s} {r['corridor']:>10s} │ "
                  f"{r['truth']:10.6f} {r['L0']:10.6f} "
                  f"{'─':>10s} {'─':>10s} {'─':>10s} │ "
                  f"{r['e0']:5.1f}%")

    print(f"\n  CRASH-WINDOW AVERAGES")
    print(f"  {'Corridor':>10s}  {'L0':>8s}  {'L1':>8s}  "
          f"{'L2':>8s}  {'L3':>8s}  {'Improv':>8s}")
    for cn in ['I-15 SB', 'I-15 NB', 'Surface']:
        a = avgs[cn]
        imp = (a['e0'] - a['e3']) / a['e0'] * 100 if a['e0'] > 0 else 0
        print(f"  {cn:>10s}  {a['e0']:7.1f}%  {a['e1']:7.1f}%  "
              f"{a['e2']:7.1f}%  {a['e3']:7.1f}%  {imp:+7.1f}%")

    print(f"\n  Total time: {_time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
