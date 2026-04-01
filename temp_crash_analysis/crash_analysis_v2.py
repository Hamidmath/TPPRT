#!/usr/bin/env python3
"""
I-15 Crash Anomaly Analysis v2
================================
Improved version:
- Uses smoothed popularity data for more stable signals
- Wider I-15 corridor (longer stretch, both directions)
- Aggregates baselines into mean + confidence band
- Includes nearby surface street analysis for diversion detection
- Cleaner publication-quality figures
"""

import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from datetime import datetime
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config
from core.pagerank import (
    build_two_phase_matrix, build_teleportation_vector,
    run_power_iteration, PARAMS,
)

OUT_DIR = Path(__file__).resolve().parent

# Crash location: I-15 SB at 14400 South, Draper UT
CRASH_LAT, CRASH_LON = 40.524, -111.893
I15_CORRIDOR_RADIUS = 0.06  # ~6 km along I-15 (wider corridor)
DIVERSION_RADIUS = 0.04     # ~4 km for surface street diversion analysis
FREEWAY_SPEED = 25.0        # m/s threshold for freeways
ARTERIAL_SPEED_MIN = 11.0   # m/s min for arterials
ARTERIAL_SPEED_MAX = 25.0   # m/s max for arterials

CRASH_DATE = '2018-09-20'
BASELINE_DATES = ['2018-09-06', '2018-09-13', '2018-09-27']
HOURS = list(range(5, 16))  # 5 AM – 3 PM

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 200,
})

# =============================================================================
# Step 1: Parse network and find links
# =============================================================================

def parse_network():
    print("Parsing slc_network.xml...")
    tree = ET.parse(str(config.NETWORK_XML))
    root = tree.getroot()

    nodes = {}
    for node in root.find('nodes'):
        nid = node.get('id')
        nodes[nid] = (float(node.get('y')), float(node.get('x')))  # (lat, lon)

    links = {}
    for link in root.find('links'):
        lid = link.get('id')
        links[lid] = {
            'from': link.get('from'), 'to': link.get('to'),
            'length': float(link.get('length')),
            'freespeed': float(link.get('freespeed')),
            'lanes': float(link.get('permlanes')),
        }
    print(f"  {len(nodes)} nodes, {len(links)} links")
    return nodes, links


def classify_links(nodes, xml_links):
    """Classify links into freeway SB, freeway NB, and surface street categories."""
    i15_sb, i15_nb, surface = [], [], []

    for lid, ldata in xml_links.items():
        fn, tn = ldata['from'], ldata['to']
        if fn not in nodes or tn not in nodes:
            continue
        lat1, lon1 = nodes[fn]
        lat2, lon2 = nodes[tn]
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2
        dist = ((mid_lat - CRASH_LAT)**2 + (mid_lon - CRASH_LON)**2)**0.5

        info = {'link_id': lid, 'lat': mid_lat, 'lon': mid_lon,
                'length': ldata['length'], 'speed': ldata['freespeed'],
                'lanes': ldata['lanes']}

        # Freeway links in I-15 corridor
        if dist < I15_CORRIDOR_RADIUS and ldata['freespeed'] >= FREEWAY_SPEED:
            if lat2 < lat1:  # southbound
                i15_sb.append(info)
            else:
                i15_nb.append(info)
        # Surface streets for diversion analysis
        elif (dist < DIVERSION_RADIUS
              and ARTERIAL_SPEED_MIN <= ldata['freespeed'] < ARTERIAL_SPEED_MAX
              and ldata['lanes'] >= 2):
            surface.append(info)

    i15_sb.sort(key=lambda x: -x['lat'])
    i15_nb.sort(key=lambda x: -x['lat'])

    print(f"  I-15 SB: {len(i15_sb)} links, NB: {len(i15_nb)} links")
    print(f"  Surface arterials nearby: {len(surface)} links")
    return i15_sb, i15_nb, surface


# =============================================================================
# Step 2: Load data
# =============================================================================

def load_smoothed_matrix():
    """Load the smoothed popularity matrix (dense, stable values)."""
    print("Loading smoothed popularity matrix...")
    loader = np.load(str(config.POPULARITY_NPZ), allow_pickle=True)
    matrix = csr_matrix((
        loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']
    ), shape=loader['matrix_shape'])
    times = list(loader['times'])
    link_ids = list(loader['link_ids'])
    print(f"  Shape: {matrix.shape}")
    return matrix, times, link_ids


def load_raw_matrix():
    print("Loading raw popularity matrix...")
    loader = np.load(str(config.POPULARITY_RAW_NPZ), allow_pickle=True)
    matrix = csr_matrix((
        loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']
    ), shape=loader['matrix_shape'])
    times = list(loader['times'])
    link_ids = list(loader['link_ids'])
    print(f"  Shape: {matrix.shape}")
    return matrix, times, link_ids


def extract_corridor_timeseries(matrix, times, link_ids, target_link_ids, date, hours):
    """Extract aggregate traffic mass for a set of links across time slots."""
    lid_to_col = {lid: i for i, lid in enumerate(link_ids)}
    cols = [lid_to_col[lid] for lid in target_link_ids if lid in lid_to_col]

    ts_vals = []
    for h in hours:
        for m in [0, 15, 30, 45]:
            ts = f"{date} {h:02d}:{m:02d}:00"
            if ts in times:
                row_idx = times.index(ts)
                row = matrix.getrow(row_idx)
                total = sum(row[0, c] for c in cols)
                ts_vals.append((ts, float(total)))
    return ts_vals


# =============================================================================
# Step 3: Figures
# =============================================================================

def fig1_smoothed_timeseries(smoothed_matrix, times, link_ids, sb_ids, nb_ids, surface_ids):
    """Smoothed traffic mass: crash day vs baseline average + band."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for ax, ids, title in [
        (axes[0], sb_ids, 'I-15 Southbound (crash direction)'),
        (axes[1], nb_ids, 'I-15 Northbound (opposite direction)'),
        (axes[2], surface_ids, 'Nearby Surface Arterials'),
    ]:
        # Crash day
        crash_ts = extract_corridor_timeseries(
            smoothed_matrix, times, link_ids, ids, CRASH_DATE, HOURS)
        crash_times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t, _ in crash_ts]
        crash_vals = np.array([v for _, v in crash_ts])

        # Baselines
        bl_arrays = []
        for bd in BASELINE_DATES:
            bl_ts = extract_corridor_timeseries(
                smoothed_matrix, times, link_ids, ids, bd, HOURS)
            bl_arrays.append(np.array([v for _, v in bl_ts]))

        bl_stack = np.array(bl_arrays)
        bl_mean = bl_stack.mean(axis=0)
        bl_std = bl_stack.std(axis=0)

        ax.fill_between(crash_times[:len(bl_mean)],
                        bl_mean - bl_std, bl_mean + bl_std,
                        alpha=0.2, color='steelblue', label='Baseline avg +/- 1 std')
        ax.plot(crash_times[:len(bl_mean)], bl_mean, 'b-', linewidth=1.5,
                alpha=0.7, label='Baseline avg (3 Thursdays)')
        ax.plot(crash_times, crash_vals, 'r-o', linewidth=2, markersize=3,
                label='Crash day (Sept 20)', zorder=5)

        crash_start = datetime.strptime(f'{CRASH_DATE} 07:30:00', '%Y-%m-%d %H:%M:%S')
        crash_end = datetime.strptime(f'{CRASH_DATE} 11:00:00', '%Y-%m-%d %H:%M:%S')
        ax.axvspan(crash_start, crash_end, alpha=0.12, color='red', label='Crash window')

        ax.set_ylabel('Smoothed Traffic Mass')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    axes[-1].set_xlabel('Time of Day')
    plt.suptitle('Traffic on I-15 Corridor Near Draper: Crash Day vs Baseline Thursdays',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = OUT_DIR / 'fig1_corridor_timeseries.png'
    plt.savefig(str(path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig2_raw_counts(raw_matrix, times, link_ids, sb_ids):
    """Raw GPS counts on I-15 SB: crash vs baselines."""
    fig, ax = plt.subplots(figsize=(14, 5))

    crash_ts = extract_corridor_timeseries(
        raw_matrix, times, link_ids, sb_ids, CRASH_DATE, HOURS)
    crash_times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t, _ in crash_ts]
    crash_vals = np.array([v for _, v in crash_ts])

    bl_arrays = []
    for bd in BASELINE_DATES:
        bl_ts = extract_corridor_timeseries(
            raw_matrix, times, link_ids, sb_ids, bd, HOURS)
        bl_arrays.append(np.array([v for _, v in bl_ts]))

    bl_stack = np.array(bl_arrays)
    bl_mean = bl_stack.mean(axis=0)
    bl_std = bl_stack.std(axis=0)

    ax.fill_between(crash_times[:len(bl_mean)],
                    bl_mean - bl_std, bl_mean + bl_std,
                    alpha=0.2, color='steelblue', label='Baseline +/- 1 std')
    ax.plot(crash_times[:len(bl_mean)], bl_mean, 'b-', linewidth=1.5, alpha=0.7,
            label='Baseline avg')
    ax.bar(crash_times, crash_vals, width=0.008, color='red', alpha=0.6,
           label='Crash day', zorder=4)

    crash_start = datetime.strptime(f'{CRASH_DATE} 07:30:00', '%Y-%m-%d %H:%M:%S')
    crash_end = datetime.strptime(f'{CRASH_DATE} 11:00:00', '%Y-%m-%d %H:%M:%S')
    ax.axvspan(crash_start, crash_end, alpha=0.12, color='red', label='Crash window')

    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Raw GPS Traversal Count')
    ax.set_title('Raw GPS Counts on I-15 SB (Draper Corridor): Crash Day vs Baselines',
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUT_DIR / 'fig2_raw_counts.png'
    plt.savefig(str(path), dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def fig3_pagerank_snapshot(graph_data, M_2N, sb_ids, nb_ids, surface_ids, link_list, times):
    """PageRank mass comparison at key times: before, during, after crash."""
    lid_to_idx = {lid: i for i, lid in enumerate(link_list)}
    sb_idx = [lid_to_idx[l] for l in sb_ids if l in lid_to_idx]
    nb_idx = [lid_to_idx[l] for l in nb_ids if l in lid_to_idx]
    sf_idx = [lid_to_idx[l] for l in surface_ids if l in lid_to_idx]

    snapshot_times = ['06:00', '07:30', '08:30', '09:30', '10:30', '11:30', '13:00']
    categories = [('I-15 SB', sb_idx), ('I-15 NB', nb_idx), ('Surface', sf_idx)]

    def run_pr(date, time_str):
        ts = f"{date} {time_str}:00"
        if ts not in times:
            return None
        pr, _ = run_pagerank_for_timeframe(graph_data, M_2N, ts)
        return pr

    def run_pagerank_for_timeframe(gd, m2n, target_time):
        N = len(gd['links'])
        E_N = build_teleportation_vector(gd, target_time)
        E_2N = np.concatenate([E_N, np.zeros(N)])
        v_2N = run_power_iteration(m2n, E_2N)
        v_final = v_2N[:N] + v_2N[N:]
        v_final /= np.sum(v_final)
        return v_final, E_N

    # Run for crash day
    crash_results = {}
    for st in snapshot_times:
        print(f"    PR crash {st}...")
        pr = run_pr(CRASH_DATE, st)
        if pr is not None:
            crash_results[st] = {cat: sum(pr[i] for i in idx)
                                 for cat, idx in categories}

    # Run for baselines
    baseline_results = defaultdict(lambda: defaultdict(list))
    for bd in BASELINE_DATES:
        for st in snapshot_times:
            print(f"    PR baseline {bd} {st}...")
            pr = run_pr(bd, st)
            if pr is not None:
                for cat, idx in categories:
                    baseline_results[st][cat].append(sum(pr[i] for i in idx))

    # Compute baseline mean
    bl_means = {}
    for st in snapshot_times:
        bl_means[st] = {cat: np.mean(vals) for cat, vals in baseline_results[st].items()}

    # Plot: grouped bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for ax, (cat, _) in zip(axes, categories):
        crash_vals = [crash_results.get(st, {}).get(cat, 0) * 1e4 for st in snapshot_times]
        bl_vals = [bl_means.get(st, {}).get(cat, 0) * 1e4 for st in snapshot_times]

        x = np.arange(len(snapshot_times))
        width = 0.35

        bars1 = ax.bar(x - width/2, crash_vals, width, color='red', alpha=0.7,
                       label='Crash day')
        bars2 = ax.bar(x + width/2, bl_vals, width, color='steelblue', alpha=0.7,
                       label='Baseline avg')

        # Highlight crash window
        ax.axvspan(0.5, 4.5, alpha=0.08, color='red')

        ax.set_xticks(x)
        ax.set_xticklabels(snapshot_times, rotation=45)
        ax.set_title(cat, fontweight='bold')
        ax.set_ylabel('PageRank Mass (x10$^{-4}$)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Two-Phase PageRank Mass at Snapshot Times: Crash Day vs Baseline',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = OUT_DIR / 'fig3_pagerank_snapshots.png'
    plt.savefig(str(path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    return crash_results, bl_means


def fig4_deviation_summary(crash_results, bl_means, snapshot_times):
    """Percentage deviation of crash day from baseline at each snapshot."""
    categories = ['I-15 SB', 'I-15 NB', 'Surface']

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'I-15 SB': '#d62728', 'I-15 NB': '#ff7f0e', 'Surface': '#2ca02c'}
    markers = {'I-15 SB': 'o', 'I-15 NB': 's', 'Surface': '^'}

    for cat in categories:
        devs = []
        for st in snapshot_times:
            crash_val = crash_results.get(st, {}).get(cat, 0)
            bl_val = bl_means.get(st, {}).get(cat, 1e-12)
            dev = (crash_val - bl_val) / bl_val * 100
            devs.append(dev)

        ax.plot(snapshot_times, devs, f'-{markers[cat]}', color=colors[cat],
                linewidth=2, markersize=8, label=cat)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvspan(0.5, 4.5, alpha=0.08, color='red', label='Crash window')
    ax.fill_between(range(len(snapshot_times)), -100, 100, alpha=0, label='')

    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Deviation from Baseline (%)')
    ax.set_title('PageRank Deviation from Baseline: Crash Day Effect',
                 fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(snapshot_times)))
    ax.set_xticklabels(snapshot_times)

    plt.tight_layout()
    path = OUT_DIR / 'fig4_deviation.png'
    plt.savefig(str(path), dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def fig5_combined_panel(smoothed_matrix, raw_matrix, times_s, times_r,
                        link_ids_s, link_ids_r, sb_ids, surface_ids,
                        crash_results, bl_means, snapshot_times):
    """4-panel summary figure for paper."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # Panel A: Smoothed traffic time series (SB)
    ax = axes[0, 0]
    crash_ts = extract_corridor_timeseries(
        smoothed_matrix, times_s, link_ids_s, sb_ids, CRASH_DATE, HOURS)
    crash_times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t, _ in crash_ts]
    crash_vals = np.array([v for _, v in crash_ts])

    bl_arrays = []
    for bd in BASELINE_DATES:
        bl_ts = extract_corridor_timeseries(
            smoothed_matrix, times_s, link_ids_s, sb_ids, bd, HOURS)
        bl_arrays.append(np.array([v for _, v in bl_ts]))
    bl_stack = np.array(bl_arrays)
    bl_mean = bl_stack.mean(axis=0)
    bl_std = bl_stack.std(axis=0)

    ax.fill_between(crash_times[:len(bl_mean)], bl_mean - bl_std, bl_mean + bl_std,
                    alpha=0.2, color='steelblue')
    ax.plot(crash_times[:len(bl_mean)], bl_mean, 'b-', linewidth=1.5, alpha=0.7,
            label='Baseline avg')
    ax.plot(crash_times, crash_vals, 'r-', linewidth=2, label='Crash day')

    cs = datetime.strptime(f'{CRASH_DATE} 07:30:00', '%Y-%m-%d %H:%M:%S')
    ce = datetime.strptime(f'{CRASH_DATE} 11:00:00', '%Y-%m-%d %H:%M:%S')
    ax.axvspan(cs, ce, alpha=0.12, color='red')
    ax.set_ylabel('Smoothed Traffic Mass')
    ax.set_title('(A) I-15 SB: Smoothed Traffic', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, alpha=0.3)

    # Panel B: Raw GPS counts (SB)
    ax = axes[0, 1]
    crash_ts_r = extract_corridor_timeseries(
        raw_matrix, times_r, link_ids_r, sb_ids, CRASH_DATE, HOURS)
    crash_times_r = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t, _ in crash_ts_r]
    crash_vals_r = np.array([v for _, v in crash_ts_r])

    bl_arrays_r = []
    for bd in BASELINE_DATES:
        bl_ts_r = extract_corridor_timeseries(
            raw_matrix, times_r, link_ids_r, sb_ids, bd, HOURS)
        bl_arrays_r.append(np.array([v for _, v in bl_ts_r]))
    bl_stack_r = np.array(bl_arrays_r)
    bl_mean_r = bl_stack_r.mean(axis=0)
    bl_std_r = bl_stack_r.std(axis=0)

    ax.fill_between(crash_times_r[:len(bl_mean_r)], bl_mean_r - bl_std_r,
                    bl_mean_r + bl_std_r, alpha=0.2, color='steelblue')
    ax.plot(crash_times_r[:len(bl_mean_r)], bl_mean_r, 'b-', linewidth=1.5, alpha=0.7,
            label='Baseline avg')
    ax.bar(crash_times_r, crash_vals_r, width=0.008, color='red', alpha=0.6,
           label='Crash day', zorder=4)
    ax.axvspan(cs, ce, alpha=0.12, color='red')
    ax.set_ylabel('Raw GPS Count')
    ax.set_title('(B) I-15 SB: Raw GPS Traversals', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, alpha=0.3)

    # Panel C: PageRank deviation
    ax = axes[1, 0]
    cats = ['I-15 SB', 'I-15 NB', 'Surface']
    colors_c = {'I-15 SB': '#d62728', 'I-15 NB': '#ff7f0e', 'Surface': '#2ca02c'}
    for cat in cats:
        devs = []
        for st in snapshot_times:
            cv = crash_results.get(st, {}).get(cat, 0)
            bv = bl_means.get(st, {}).get(cat, 1e-12)
            devs.append((cv - bv) / bv * 100)
        ax.plot(snapshot_times, devs, '-o', color=colors_c[cat], linewidth=2,
                markersize=6, label=cat)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Deviation from Baseline (%)')
    ax.set_title('(C) PageRank Deviation from Baseline', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel D: Surface street diversion (smoothed)
    ax = axes[1, 1]
    crash_sf = extract_corridor_timeseries(
        smoothed_matrix, times_s, link_ids_s, surface_ids, CRASH_DATE, HOURS)
    crash_times_sf = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t, _ in crash_sf]
    crash_vals_sf = np.array([v for _, v in crash_sf])

    bl_sf = []
    for bd in BASELINE_DATES:
        bt = extract_corridor_timeseries(
            smoothed_matrix, times_s, link_ids_s, surface_ids, bd, HOURS)
        bl_sf.append(np.array([v for _, v in bt]))
    bl_sf_stack = np.array(bl_sf)
    bl_sf_mean = bl_sf_stack.mean(axis=0)
    bl_sf_std = bl_sf_stack.std(axis=0)

    ax.fill_between(crash_times_sf[:len(bl_sf_mean)], bl_sf_mean - bl_sf_std,
                    bl_sf_mean + bl_sf_std, alpha=0.2, color='steelblue')
    ax.plot(crash_times_sf[:len(bl_sf_mean)], bl_sf_mean, 'b-', linewidth=1.5,
            alpha=0.7, label='Baseline avg')
    ax.plot(crash_times_sf, crash_vals_sf, 'g-', linewidth=2,
            label='Crash day', zorder=5)
    ax.axvspan(cs, ce, alpha=0.12, color='red')
    ax.set_xlabel('Time of Day')
    ax.set_ylabel('Smoothed Traffic Mass')
    ax.set_title('(D) Nearby Surface Streets: Diversion Effect', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        'I-15 Multi-Vehicle Crash Analysis (Sept 20, 2018, 7:30–11:00 AM, Draper UT)',
        fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = OUT_DIR / 'fig5_combined_panel.png'
    plt.savefig(str(path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("I-15 CRASH ANALYSIS v2")
    print("=" * 70)

    # Parse network
    nodes, xml_links = parse_network()
    i15_sb, i15_nb, surface = classify_links(nodes, xml_links)
    sb_ids = [l['link_id'] for l in i15_sb]
    nb_ids = [l['link_id'] for l in i15_nb]
    sf_ids = [l['link_id'] for l in surface]

    print(f"\n  I-15 SB links ({len(sb_ids)}):")
    for l in i15_sb[:10]:
        print(f"    Link {l['link_id']}: {l['length']:.0f}m, "
              f"{l['speed']:.1f} m/s, {l['lanes']:.0f} lanes, lat {l['lat']:.4f}")
    if len(i15_sb) > 10:
        print(f"    ... and {len(i15_sb) - 10} more")

    # Load data
    smoothed_matrix, times_s, link_ids_s = load_smoothed_matrix()
    raw_matrix, times_r, link_ids_r = load_raw_matrix()

    # Figure 1: Smoothed corridor time series (3 panels)
    print("\n[FIG 1] Smoothed corridor time series...")
    fig1_smoothed_timeseries(smoothed_matrix, times_s, link_ids_s,
                             sb_ids, nb_ids, sf_ids)

    # Figure 2: Raw GPS counts
    print("\n[FIG 2] Raw GPS counts...")
    fig2_raw_counts(raw_matrix, times_r, link_ids_r, sb_ids)

    # Figure 3: PageRank snapshots
    print("\n[FIG 3] PageRank snapshots (running 28 PageRank computations)...")
    with open(config.GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)
    link_list = list(graph_data['links'].keys())
    M_2N = build_two_phase_matrix(graph_data)

    snapshot_times = ['06:00', '07:30', '08:30', '09:30', '10:30', '11:30', '13:00']
    crash_results, bl_means = fig3_pagerank_snapshot(
        graph_data, M_2N, sb_ids, nb_ids, sf_ids, link_list, times_s)

    # Figure 4: Deviation summary
    print("\n[FIG 4] Deviation summary...")
    fig4_deviation_summary(crash_results, bl_means, snapshot_times)

    # Figure 5: Combined 4-panel for paper
    print("\n[FIG 5] Combined panel figure...")
    fig5_combined_panel(smoothed_matrix, raw_matrix, times_s, times_r,
                        link_ids_s, link_ids_r, sb_ids, sf_ids,
                        crash_results, bl_means, snapshot_times)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for st in snapshot_times:
        if st in crash_results and st in bl_means:
            for cat in ['I-15 SB', 'I-15 NB', 'Surface']:
                cv = crash_results[st].get(cat, 0)
                bv = bl_means[st].get(cat, 1e-12)
                dev = (cv - bv) / bv * 100
                print(f"  {st} {cat:10s}: crash={cv:.6f} baseline={bv:.6f} dev={dev:+.1f}%")

    # Write report
    report_lines = [
        "I-15 Crash Anomaly Analysis Report",
        "=" * 40,
        f"Event: Multi-vehicle crash, I-15 SB at 14400 S, Draper UT",
        f"Date:  Thursday, September 20, 2018, 7:30-11:00 AM",
        f"Links: {len(sb_ids)} SB, {len(nb_ids)} NB, {len(sf_ids)} surface",
        "",
        "PageRank Deviations from Baseline:",
    ]
    for st in snapshot_times:
        if st in crash_results and st in bl_means:
            for cat in ['I-15 SB', 'I-15 NB', 'Surface']:
                cv = crash_results[st].get(cat, 0)
                bv = bl_means[st].get(cat, 1e-12)
                dev = (cv - bv) / bv * 100
                report_lines.append(f"  {st} {cat:10s}: {dev:+.1f}%")

    report_lines.extend([
        "",
        "Figures:",
        "  fig1_corridor_timeseries.png  — Smoothed traffic (SB/NB/Surface)",
        "  fig2_raw_counts.png           — Raw GPS counts on I-15 SB",
        "  fig3_pagerank_snapshots.png   — PageRank mass at 7 time points",
        "  fig4_deviation.png            — % deviation from baseline",
        "  fig5_combined_panel.png       — 4-panel summary for paper",
    ])

    report = '\n'.join(report_lines)
    with open(OUT_DIR / 'crash_report.txt', 'w') as f:
        f.write(report)
    print(f"\nReport saved to {OUT_DIR / 'crash_report.txt'}")


if __name__ == '__main__':
    main()
