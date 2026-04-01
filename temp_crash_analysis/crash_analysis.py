#!/usr/bin/env python3
"""
I-15 Crash Anomaly Analysis
============================
Analyzes the Sept 20, 2018 multi-vehicle crash on I-15 SB at 14400 South, Draper.
Compares crash-day traffic patterns against baseline Thursdays using both raw GPS
counts and Two-Phase PageRank model output.
"""

import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy.sparse import csr_matrix

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config
from core.pagerank import (
    build_two_phase_matrix,
    build_teleportation_vector,
    run_power_iteration,
    PARAMS,
)

OUT_DIR = Path(__file__).resolve().parent
CRASH_LAT, CRASH_LON = 40.524, -111.893  # 14400 South & I-15, Draper UT
SEARCH_RADIUS_DEG = 0.025  # ~2.5 km radius for I-15 links
FREEWAY_SPEED_THRESH = 25.0  # m/s (~56 mph) — freeway links only

CRASH_DATE = '2018-09-20'
BASELINE_DATES = ['2018-09-06', '2018-09-13', '2018-09-27']
ANALYSIS_HOURS = list(range(5, 16))  # 5 AM – 3 PM

# =============================================================================
# Step 1: Parse network XML and identify I-15 links near crash
# =============================================================================

def parse_network_xml():
    """Parse slc_network.xml to extract node coords and link definitions."""
    print("Parsing slc_network.xml...")
    tree = ET.parse(str(config.NETWORK_XML))
    root = tree.getroot()

    nodes = {}
    for node in root.find('nodes'):
        nid = node.get('id')
        x = float(node.get('x'))  # longitude
        y = float(node.get('y'))  # latitude
        nodes[nid] = (y, x)  # (lat, lon)

    links = {}
    for link in root.find('links'):
        lid = link.get('id')
        links[lid] = {
            'from': link.get('from'),
            'to': link.get('to'),
            'length': float(link.get('length')),
            'freespeed': float(link.get('freespeed')),
            'capacity': float(link.get('capacity')),
            'lanes': float(link.get('permlanes')),
        }

    print(f"  Parsed {len(nodes)} nodes, {len(links)} links")
    return nodes, links


def find_crash_zone_links(nodes, links):
    """Find high-speed (freeway) links near the crash location."""
    crash_links = []
    for lid, ldata in links.items():
        from_node = ldata['from']
        to_node = ldata['to']
        if from_node not in nodes or to_node not in nodes:
            continue

        # Midpoint of the link
        lat1, lon1 = nodes[from_node]
        lat2, lon2 = nodes[to_node]
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2

        dist_deg = ((mid_lat - CRASH_LAT)**2 + (mid_lon - CRASH_LON)**2)**0.5

        if dist_deg < SEARCH_RADIUS_DEG and ldata['freespeed'] >= FREEWAY_SPEED_THRESH:
            # Determine direction: southbound = negative latitude change
            direction = 'SB' if lat2 < lat1 else 'NB'
            crash_links.append({
                'link_id': lid,
                'mid_lat': mid_lat,
                'mid_lon': mid_lon,
                'length': ldata['length'],
                'freespeed': ldata['freespeed'],
                'lanes': ldata['lanes'],
                'direction': direction,
                'from_lat': lat1, 'from_lon': lon1,
                'to_lat': lat2, 'to_lon': lon2,
            })

    # Sort by latitude (north to south)
    crash_links.sort(key=lambda x: -x['mid_lat'])
    print(f"  Found {len(crash_links)} freeway links near crash site")
    sb = [l for l in crash_links if l['direction'] == 'SB']
    nb = [l for l in crash_links if l['direction'] == 'NB']
    print(f"    Southbound: {len(sb)}, Northbound: {len(nb)}")
    return crash_links


# =============================================================================
# Step 2: Load data and extract timeframe traffic
# =============================================================================

def load_raw_matrix():
    """Load raw (unsmoothed) popularity matrix."""
    print("Loading raw popularity matrix...")
    loader = np.load(str(config.POPULARITY_RAW_NPZ), allow_pickle=True)
    matrix = csr_matrix((
        loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']
    ), shape=loader['matrix_shape'])
    times = list(loader['times'])
    link_ids = list(loader['link_ids'])
    print(f"  Shape: {matrix.shape}, links: {len(link_ids)}, timeframes: {len(times)}")
    return matrix, times, link_ids


def get_timeframe_indices(times, date, hours):
    """Get matrix row indices for a given date and list of hours."""
    indices = {}
    for h in hours:
        for m in [0, 15, 30, 45]:
            ts = f"{date} {h:02d}:{m:02d}:00"
            if ts in times:
                indices[ts] = times.index(ts)
    return indices


def extract_link_counts(matrix, link_ids, target_link_ids, time_indices):
    """Extract raw traversal counts for specific links across timeframes."""
    lid_to_col = {lid: i for i, lid in enumerate(link_ids)}
    result = {}  # timestamp -> {link_id: count}

    for ts, row_idx in time_indices.items():
        row = matrix.getrow(row_idx)
        counts = {}
        for lid in target_link_ids:
            if lid in lid_to_col:
                col = lid_to_col[lid]
                # Get the value at this column from the sparse row
                val = row[0, col] if col < row.shape[1] else 0
                counts[lid] = float(val)
            else:
                counts[lid] = 0.0
        result[ts] = counts

    return result


# =============================================================================
# Step 3: Run PageRank on selected timeframes
# =============================================================================

def run_pagerank_for_timeframe(graph_data, M_2N, target_time, npz_path=None):
    """Run Two-Phase PageRank for a single timeframe and return N-dim result."""
    N = len(graph_data['links'])
    E_N = build_teleportation_vector(graph_data, target_time, npz_path)
    E_2N = np.concatenate([E_N, np.zeros(N)])
    v_2N = run_power_iteration(M_2N, E_2N)
    v_final = v_2N[:N] + v_2N[N:]
    v_final /= np.sum(v_final)
    return v_final, E_N


# =============================================================================
# Step 4: Analysis and visualization
# =============================================================================

def plot_time_series(crash_counts, baseline_counts_list, sb_link_ids,
                     baseline_dates, title_suffix=""):
    """Plot time series of aggregate traffic on SB links: crash day vs baselines."""
    # Aggregate counts across all SB links per timeframe
    def aggregate(counts_dict):
        times_sorted = sorted(counts_dict.keys())
        agg = []
        for ts in times_sorted:
            total = sum(counts_dict[ts].get(lid, 0) for lid in sb_link_ids)
            agg.append((ts, total))
        return agg

    crash_agg = aggregate(crash_counts)
    baseline_aggs = [aggregate(bc) for bc in baseline_counts_list]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Parse timestamps
    crash_times = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t, _ in crash_agg]
    crash_vals = [v for _, v in crash_agg]

    # Normalize times to just hour:minute for overlay
    base_times = crash_times  # Use crash day times as x-axis

    ax.plot(base_times, crash_vals, 'r-o', linewidth=2.5, markersize=5,
            label=f'Crash day (Sept 20)', zorder=5)

    colors = ['#1f77b4', '#2ca02c', '#9467bd']
    for i, (bagg, bdate) in enumerate(zip(baseline_aggs, baseline_dates)):
        bvals = [v for _, v in bagg]
        # Align to crash day x-axis
        ax.plot(base_times[:len(bvals)], bvals, '-', color=colors[i],
                linewidth=1.5, alpha=0.7, label=f'Baseline ({bdate})')

    # Mark crash window
    crash_start = datetime.strptime(f'{CRASH_DATE} 07:30:00', '%Y-%m-%d %H:%M:%S')
    crash_end = datetime.strptime(f'{CRASH_DATE} 11:00:00', '%Y-%m-%d %H:%M:%S')
    ax.axvspan(crash_start, crash_end, alpha=0.15, color='red', label='Crash window (7:30–11:00)')

    ax.set_xlabel('Time of Day', fontsize=12)
    ax.set_ylabel('Total GPS Traversals on I-15 SB Links', fontsize=12)
    ax.set_title(f'I-15 Southbound Traffic Near Draper: Crash Day vs Baselines{title_suffix}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / 'fig_crash_timeseries.png'
    plt.savefig(str(out_path), dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_pagerank_comparison(crash_pr, baseline_prs, sb_link_ids, link_list, title=""):
    """Compare PageRank distributions on SB links: crash vs baselines."""
    lid_to_idx = {lid: i for i, lid in enumerate(link_list)}

    # Get SB link PageRank values
    sb_indices = [lid_to_idx[lid] for lid in sb_link_ids if lid in lid_to_idx]

    crash_sb = sum(crash_pr[i] for i in sb_indices)
    baseline_sbs = [sum(bpr[i] for i in sb_indices) for bpr in baseline_prs]
    baseline_mean = np.mean(baseline_sbs)

    print(f"  I-15 SB PageRank mass — Crash: {crash_sb:.6f}, "
          f"Baseline mean: {baseline_mean:.6f}, "
          f"Change: {(crash_sb - baseline_mean) / baseline_mean * 100:.1f}%")

    return crash_sb, baseline_mean


def plot_anomaly_heatmap(crash_pr, baseline_prs, link_list, nodes, xml_links,
                         crash_zone_links, n_top=30):
    """Show top links with biggest positive and negative PageRank deviations."""
    baseline_mean = np.mean(baseline_prs, axis=0)
    baseline_std = np.std(baseline_prs, axis=0)
    baseline_std[baseline_std < 1e-12] = 1e-12

    # Z-scores
    z_scores = (crash_pr - baseline_mean) / baseline_std

    # Get coordinates for each link
    lid_to_idx = {lid: i for i, lid in enumerate(link_list)}
    crash_zone_ids = set(l['link_id'] for l in crash_zone_links)

    # Top positive deviations (alternate routes getting more traffic)
    top_pos = np.argsort(z_scores)[::-1][:n_top]
    # Top negative deviations (crash zone losing traffic)
    top_neg = np.argsort(z_scores)[:n_top]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, indices, title, cmap in [
        (axes[0], top_neg, 'Largest Traffic Decrease\n(crash impact zone)', 'Reds_r'),
        (axes[1], top_pos, 'Largest Traffic Increase\n(diversion routes)', 'Greens'),
    ]:
        lats, lons, zs, labels = [], [], [], []
        for idx in indices:
            lid = link_list[idx]
            if lid in xml_links:
                xl = xml_links[lid]
                from_node = xl['from']
                to_node = xl['to']
                if from_node in nodes and to_node in nodes:
                    lat1, lon1 = nodes[from_node]
                    lat2, lon2 = nodes[to_node]
                    lats.append((lat1 + lat2) / 2)
                    lons.append((lon1 + lon2) / 2)
                    zs.append(z_scores[idx])
                    marker = '*' if lid in crash_zone_ids else 'o'
                    labels.append((lid, marker, xl['freespeed'], xl['lanes']))

        scatter = ax.scatter(lons, lats, c=zs, cmap=cmap, s=80, edgecolors='k',
                            linewidths=0.5, zorder=3)
        # Mark crash location
        ax.plot(CRASH_LON, CRASH_LAT, 'rX', markersize=18, markeredgewidth=2,
                zorder=5, label='Crash site')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        plt.colorbar(scatter, ax=ax, label='Z-score', shrink=0.8)
        ax.grid(True, alpha=0.2)

    plt.suptitle('Spatial Anomaly Map: PageRank Z-Scores (Crash Day vs Baselines)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = OUT_DIR / 'fig_crash_anomaly_map.png'
    plt.savefig(str(out_path), dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_recovery_curve(all_crash_counts, all_baseline_counts_list, sb_link_ids):
    """Show how SB traffic recovers after the crash is cleared."""
    def hourly_agg(counts_dict):
        hourly = {}
        for ts, link_counts in counts_dict.items():
            hour = ts.split(' ')[1][:2]
            total = sum(link_counts.get(lid, 0) for lid in sb_link_ids)
            if hour not in hourly:
                hourly[hour] = []
            hourly[hour].append(total)
        return {h: sum(v) for h, v in hourly.items()}

    crash_hourly = hourly_agg(all_crash_counts)
    baseline_hourlys = [hourly_agg(bc) for bc in all_baseline_counts_list]

    hours = sorted(set(crash_hourly.keys()) & set(baseline_hourlys[0].keys()))
    hours_int = [int(h) for h in hours]

    crash_vals = [crash_hourly.get(h, 0) for h in hours]
    baseline_vals = [np.mean([bh.get(h, 0) for bh in baseline_hourlys]) for h in hours]

    # Ratio
    ratios = [c / b if b > 0 else 1.0 for c, b in zip(crash_vals, baseline_vals)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.bar(hours_int, crash_vals, width=0.35, label='Crash day', color='red', alpha=0.7, align='center')
    ax1.bar([h + 0.35 for h in hours_int], baseline_vals, width=0.35,
            label='Baseline avg', color='steelblue', alpha=0.7, align='center')
    ax1.axvspan(7.5, 11.0, alpha=0.1, color='red')
    ax1.set_ylabel('Hourly GPS Count', fontsize=12)
    ax1.set_title('I-15 SB Hourly Traffic: Crash Day vs Baseline Average', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(hours_int, ratios, 'ko-', linewidth=2, markersize=8)
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvspan(7.5, 11.0, alpha=0.1, color='red', label='Crash window')
    ax2.set_xlabel('Hour of Day', fontsize=12)
    ax2.set_ylabel('Ratio (Crash / Baseline)', fontsize=12)
    ax2.set_title('Traffic Ratio: Recovery After Crash Clearance', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(hours_int)
    ax2.set_xticklabels([f'{h}:00' for h in hours_int], rotation=45)

    plt.tight_layout()
    out_path = OUT_DIR / 'fig_crash_recovery.png'
    plt.savefig(str(out_path), dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_pagerank_timeseries(graph_data, M_2N, sb_link_ids, link_list,
                             times, crash_date, baseline_dates, hours):
    """Run PageRank for each 15-min slot and compare SB mass over time."""
    lid_to_idx = {lid: i for i, lid in enumerate(link_list)}
    sb_indices = [lid_to_idx[lid] for lid in sb_link_ids if lid in lid_to_idx]

    def get_pr_timeseries(date, target_hours):
        ts_list = []
        pr_vals = []
        for h in target_hours:
            for m in [0, 15, 30, 45]:
                ts = f"{date} {h:02d}:{m:02d}:00"
                if ts in times:
                    pr, _ = run_pagerank_for_timeframe(graph_data, M_2N, ts)
                    sb_mass = sum(pr[i] for i in sb_indices)
                    ts_list.append(ts)
                    pr_vals.append(sb_mass)
        return ts_list, pr_vals

    print("  Running PageRank for crash day timeframes...")
    crash_ts, crash_vals = get_pr_timeseries(crash_date, hours)

    baseline_results = []
    for bdate in baseline_dates:
        print(f"  Running PageRank for baseline {bdate}...")
        bts, bvals = get_pr_timeseries(bdate, hours)
        baseline_results.append((bts, bvals))

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    crash_times_dt = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in crash_ts]
    ax.plot(crash_times_dt, crash_vals, 'r-o', linewidth=2.5, markersize=5,
            label='Crash day (Sept 20)', zorder=5)

    colors = ['#1f77b4', '#2ca02c', '#9467bd']
    for i, (bts, bvals) in enumerate(baseline_results):
        # Align x-axis to crash day
        ax.plot(crash_times_dt[:len(bvals)], bvals, '-', color=colors[i],
                linewidth=1.5, alpha=0.7, label=f'Baseline ({baseline_dates[i]})')

    crash_start = datetime.strptime(f'{crash_date} 07:30:00', '%Y-%m-%d %H:%M:%S')
    crash_end = datetime.strptime(f'{crash_date} 11:00:00', '%Y-%m-%d %H:%M:%S')
    ax.axvspan(crash_start, crash_end, alpha=0.15, color='red', label='Crash window')

    ax.set_xlabel('Time of Day', fontsize=12)
    ax.set_ylabel('PageRank Mass on I-15 SB Links', fontsize=12)
    ax.set_title('Two-Phase PageRank: I-15 SB Mass During Crash vs Baselines',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / 'fig_crash_pagerank_timeseries.png'
    plt.savefig(str(out_path), dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")

    return crash_ts, crash_vals, baseline_results


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("I-15 CRASH ANOMALY ANALYSIS")
    print("Event: Multi-vehicle crash, I-15 SB at 14400 South, Draper UT")
    print("Date:  Thursday, September 20, 2018, 7:30–11:00 AM")
    print("=" * 70)

    # Step 1: Find crash-zone links
    print("\n[STEP 1] Identifying I-15 links near crash site...")
    nodes, xml_links = parse_network_xml()
    crash_zone_links = find_crash_zone_links(nodes, xml_links)

    if not crash_zone_links:
        print("ERROR: No freeway links found near crash site. Adjusting search...")
        return

    sb_links = [l for l in crash_zone_links if l['direction'] == 'SB']
    nb_links = [l for l in crash_zone_links if l['direction'] == 'NB']
    sb_link_ids = [l['link_id'] for l in sb_links]
    all_crash_link_ids = [l['link_id'] for l in crash_zone_links]

    print(f"\n  I-15 SB links ({len(sb_links)}):")
    for l in sb_links:
        print(f"    Link {l['link_id']}: {l['length']:.0f}m, "
              f"{l['freespeed']:.1f} m/s, {l['lanes']:.0f} lanes, "
              f"lat {l['mid_lat']:.4f}")

    # Step 2: Load raw data
    print("\n[STEP 2] Loading raw GPS counts...")
    raw_matrix, times, link_ids = load_raw_matrix()

    # Extract counts for crash day and baselines
    crash_indices = get_timeframe_indices(times, CRASH_DATE, ANALYSIS_HOURS)
    baseline_indices_list = [
        get_timeframe_indices(times, d, ANALYSIS_HOURS) for d in BASELINE_DATES
    ]

    crash_counts = extract_link_counts(raw_matrix, link_ids, sb_link_ids, crash_indices)
    baseline_counts_list = [
        extract_link_counts(raw_matrix, link_ids, sb_link_ids, bi)
        for bi in baseline_indices_list
    ]

    # Step 3: Raw GPS time-series plot
    print("\n[STEP 3] Plotting raw GPS time series...")
    plot_time_series(crash_counts, baseline_counts_list, sb_link_ids, BASELINE_DATES)

    # Step 4: Recovery curve
    print("\n[STEP 4] Plotting recovery curve...")
    plot_recovery_curve(crash_counts, baseline_counts_list, sb_link_ids)

    # Step 5: PageRank analysis
    print("\n[STEP 5] Running Two-Phase PageRank comparison...")
    print("  Loading graph data...")
    with open(config.GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)
    link_list = list(graph_data['links'].keys())

    print("  Building 2N x 2N transition matrix (one-time)...")
    M_2N = build_two_phase_matrix(graph_data)

    # Run PageRank for a crash window slot and baseline slots
    crash_window_time = f'{CRASH_DATE} 08:30:00'  # Mid-crash
    pre_crash_time = f'{CRASH_DATE} 06:00:00'     # Before crash

    print(f"\n  PageRank at crash time ({crash_window_time})...")
    crash_pr, crash_en = run_pagerank_for_timeframe(graph_data, M_2N, crash_window_time)

    print(f"  PageRank before crash ({pre_crash_time})...")
    pre_pr, _ = run_pagerank_for_timeframe(graph_data, M_2N, pre_crash_time)

    baseline_prs = []
    for bdate in BASELINE_DATES:
        bt = f'{bdate} 08:30:00'
        print(f"  PageRank at baseline ({bt})...")
        bpr, _ = run_pagerank_for_timeframe(graph_data, M_2N, bt)
        baseline_prs.append(bpr)

    # Compare SB mass
    print("\n  === I-15 SB PageRank Mass Comparison ===")
    print(f"  Pre-crash (06:00):")
    plot_pagerank_comparison(pre_pr, baseline_prs, sb_link_ids, link_list)
    print(f"  During crash (08:30):")
    crash_sb, bl_sb = plot_pagerank_comparison(crash_pr, baseline_prs, sb_link_ids, link_list)

    # Step 6: Spatial anomaly map
    print("\n[STEP 6] Generating spatial anomaly map...")
    plot_anomaly_heatmap(crash_pr, np.array(baseline_prs), link_list,
                         nodes, xml_links, crash_zone_links)

    # Step 7: Full PageRank time series (this is slow — runs PR for each 15-min slot)
    print("\n[STEP 7] Running full PageRank time series (5AM–3PM, crash + 3 baselines)...")
    pr_hours = list(range(5, 16))
    crash_ts, crash_pr_vals, bl_pr_results = plot_pagerank_timeseries(
        graph_data, M_2N, sb_link_ids, link_list,
        times, CRASH_DATE, BASELINE_DATES, pr_hours
    )

    # Step 8: Summary report
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    # Compute crash-window statistics
    crash_window_slots = [ts for ts in sorted(crash_counts.keys())
                          if '07:' in ts or '08:' in ts or '09:' in ts or '10:' in ts]
    pre_slots = [ts for ts in sorted(crash_counts.keys())
                 if '05:' in ts or '06:' in ts]

    def slot_total(counts, slots):
        return sum(sum(counts[ts].get(lid, 0) for lid in sb_link_ids)
                   for ts in slots if ts in counts)

    def match_slots_by_time(counts, reference_slots):
        """Match slots by time-of-day, ignoring date."""
        ref_times = {ts.split(' ')[1] for ts in reference_slots}
        return [ts for ts in counts.keys() if ts.split(' ')[1] in ref_times]

    crash_window_total = slot_total(crash_counts, crash_window_slots)
    crash_pre_total = slot_total(crash_counts, pre_slots)

    bl_window_totals = [slot_total(bc, match_slots_by_time(bc, crash_window_slots))
                        for bc in baseline_counts_list]
    bl_pre_totals = [slot_total(bc, match_slots_by_time(bc, pre_slots))
                     for bc in baseline_counts_list]

    bl_window_mean = np.mean(bl_window_totals)
    bl_pre_mean = np.mean(bl_pre_totals)

    report = f"""
I-15 Crash Anomaly Analysis Report
===================================

Event: Multi-vehicle crash on I-15 Southbound at 14400 South, Draper, UT
Date:  Thursday, September 20, 2018, approximately 7:30 AM – 11:00 AM
Links identified: {len(sb_link_ids)} southbound I-15 links near crash site

RAW GPS COUNTS (I-15 SB links)
─────────────────────────────────
  Pre-crash (5–7 AM):
    Crash day:      {crash_pre_total:.0f} traversals
    Baseline avg:   {bl_pre_mean:.0f} traversals
    Difference:     {(crash_pre_total - bl_pre_mean)/bl_pre_mean*100:+.1f}%

  Crash window (7–11 AM):
    Crash day:      {crash_window_total:.0f} traversals
    Baseline avg:   {bl_window_mean:.0f} traversals
    Difference:     {(crash_window_total - bl_window_mean)/bl_window_mean*100:+.1f}%

PAGERANK ANALYSIS (8:30 AM snapshot)
─────────────────────────────────
  Crash day SB mass:    {crash_sb:.6f}
  Baseline avg SB mass: {bl_sb:.6f}
  Change:               {(crash_sb - bl_sb)/bl_sb*100:+.1f}%

Figures generated:
  1. fig_crash_timeseries.png     — Raw GPS counts time series
  2. fig_crash_recovery.png       — Hourly traffic + recovery ratio
  3. fig_crash_anomaly_map.png    — Spatial anomaly map (z-scores)
  4. fig_crash_pagerank_timeseries.png — PageRank mass time series
"""
    print(report)

    with open(OUT_DIR / 'crash_report.txt', 'w') as f:
        f.write(report)
    print(f"Report saved to {OUT_DIR / 'crash_report.txt'}")


if __name__ == '__main__':
    main()
