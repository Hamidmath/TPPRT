"""
Comprehensive data analysis for the Two-Phase PageRank project.

Analyzes all available data sources:
  1. Road network topology (city_graph_full.json)
  2. Matched routes (matched_routes.json)
  3. Raw popularity matrix (popularity_results.npz)
  4. Smoothed popularity matrix (popularity_results_smoothed.npz)
  5. PageRank output (two_phase_pagerank_vector.json)

Outputs:
  - data/statistics.json          (full statistics)
  - figures/analysis/*.png        (visualizations)
  - Console summary
"""
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DATE = datetime(2018, 1, 1)
OUTPUT_DIR = config.FIGURES_DIR / 'analysis'
STATS_FILE = config.DATA_DIR / 'statistics.json'


def load_npz(path):
    loader = np.load(str(path), allow_pickle=True)
    matrix = sparse.csr_matrix(
        (loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']),
        shape=loader['matrix_shape']
    )
    return matrix, list(loader['times']), list(loader['link_ids'])


def percentiles(arr, name=""):
    a = np.array(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {}
    return {
        'count': int(len(a)),
        'mean': float(np.mean(a)),
        'std': float(np.std(a)),
        'min': float(np.min(a)),
        '5%': float(np.percentile(a, 5)),
        '25%': float(np.percentile(a, 25)),
        '50%_median': float(np.median(a)),
        '75%': float(np.percentile(a, 75)),
        '95%': float(np.percentile(a, 95)),
        '99%': float(np.percentile(a, 99)),
        'max': float(np.max(a)),
        'total': float(np.sum(a)),
        'skewness': float(skewness(a)),
        'kurtosis': float(kurtosis(a)),
    }


def skewness(a):
    m = np.mean(a)
    s = np.std(a)
    if s == 0:
        return 0.0
    return float(np.mean(((a - m) / s) ** 3))


def kurtosis(a):
    m = np.mean(a)
    s = np.std(a)
    if s == 0:
        return 0.0
    return float(np.mean(((a - m) / s) ** 4) - 3.0)


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_stat(label, value, unit=""):
    if isinstance(value, float):
        print(f"  {label:.<45} {value:>12.4f} {unit}")
    else:
        print(f"  {label:.<45} {str(value):>12} {unit}")


def analyze_network(graph_data):
    """Analyze road network topology."""
    print_section("ROAD NETWORK TOPOLOGY")

    links = graph_data['links']
    adj = graph_data.get('adjacency', {})
    N = len(links)

    speeds = []
    lengths = []
    lanes = []
    travel_times = []

    for lid, info in links.items():
        s = info.get('speed', 11.17)
        l = info.get('length', 100.0)
        la = info.get('lanes', 1.0)
        speeds.append(s)
        lengths.append(l)
        lanes.append(la)
        travel_times.append(l / max(s, 0.1))

    speeds = np.array(speeds)
    lengths = np.array(lengths)
    lanes = np.array(lanes)
    travel_times = np.array(travel_times)

    # Degree analysis
    out_degrees = []
    in_degree_count = Counter()
    for lid in links:
        outs = adj.get(lid, [])
        out_degrees.append(len(outs))
        for out_lid in outs:
            in_degree_count[out_lid] += 1

    in_degrees = [in_degree_count.get(lid, 0) for lid in links]
    out_degrees = np.array(out_degrees)
    in_degrees = np.array(in_degrees)

    # Dead ends and sources
    dead_ends = int(np.sum(out_degrees == 0))
    sources = int(np.sum(in_degrees == 0))
    isolated = int(np.sum((out_degrees == 0) & (in_degrees == 0)))

    print_stat("Total links (N)", N)
    print_stat("Total edges", sum(out_degrees))
    print_stat("Dead-end links (out-degree=0)", dead_ends)
    print_stat("Source links (in-degree=0)", sources)
    print_stat("Isolated links (both=0)", isolated)
    print()
    print_stat("Link length mean", np.mean(lengths), "m")
    print_stat("Link length median", np.median(lengths), "m")
    print_stat("Link length max", np.max(lengths), "m")
    print_stat("Total network length", np.sum(lengths) / 1000, "km")
    print()
    print_stat("Speed mean", np.mean(speeds), "m/s")
    print_stat("Speed median", np.median(speeds), "m/s")
    print_stat("Speed mean (km/h)", np.mean(speeds) * 3.6, "km/h")
    print()
    print_stat("Lanes mean", np.mean(lanes))
    print_stat("Lanes median", np.median(lanes))
    print_stat("Multi-lane links (>1)", int(np.sum(lanes > 1)))
    print_stat("Highway links (>=3 lanes)", int(np.sum(lanes >= 3)))
    print()
    print_stat("Travel time mean", np.mean(travel_times), "s")
    print_stat("Travel time median", np.median(travel_times), "s")
    print_stat("Travel time max", np.max(travel_times), "s")
    print()
    print_stat("Out-degree mean", np.mean(out_degrees))
    print_stat("Out-degree max", np.max(out_degrees))
    print_stat("In-degree mean", np.mean(in_degrees))
    print_stat("In-degree max", np.max(in_degrees))

    # Lane distribution
    lane_counts = Counter()
    for la in lanes:
        lane_counts[la] += 1
    print("\n  Lane distribution:")
    for la in sorted(lane_counts.keys()):
        pct = lane_counts[la] / N * 100
        print(f"    {la:5.1f} lanes: {lane_counts[la]:>7} links ({pct:5.1f}%)")

    stats = {
        'total_links': N,
        'total_edges': int(sum(out_degrees)),
        'dead_ends': dead_ends,
        'sources': sources,
        'isolated': isolated,
        'total_network_length_km': float(np.sum(lengths) / 1000),
        'link_length': percentiles(lengths),
        'speed_ms': percentiles(speeds),
        'lanes': percentiles(lanes),
        'travel_time_sec': percentiles(travel_times),
        'out_degree': percentiles(out_degrees),
        'in_degree': percentiles(in_degrees),
    }

    return stats, speeds, lengths, lanes, travel_times, out_degrees, in_degrees


def analyze_matched_routes(matched_json, graph_links):
    """Analyze matched route characteristics."""
    print_section("MATCHED ROUTES")

    num_routes = len(matched_json)
    links_per_route = []
    durations = []
    unique_links_seen = set()
    link_traversal_counts = Counter()
    routes_per_hour = Counter()

    for item in matched_json:
        route = item['route']
        routelinks = route.get('routelinks', [])
        n_links = len(routelinks)
        links_per_route.append(n_links)

        if routelinks:
            for rl in routelinks:
                lid = str(rl[0])
                unique_links_seen.add(lid)
                link_traversal_counts[lid] += 1

            first_time = float(routelinks[0][1])
            last_time = float(routelinks[-1][2]) if len(routelinks[-1]) > 2 else float(routelinks[-1][1])
            dur = last_time - first_time
            if dur > 0:
                durations.append(dur)

            # Temporal distribution
            abs_time = datetime(2018, 1, 1).timestamp() + first_time
            dt = datetime.fromtimestamp(abs_time)
            routes_per_hour[dt.hour] += 1

    links_per_route = np.array(links_per_route)
    N_graph = len(graph_links)
    coverage = len(unique_links_seen)
    coverage_pct = coverage / N_graph * 100

    # Link usage distribution
    traversal_values = list(link_traversal_counts.values())
    top_10_links = link_traversal_counts.most_common(10)

    print_stat("Total matched routes", num_routes)
    print_stat("Links per route (mean)", np.mean(links_per_route))
    print_stat("Links per route (median)", np.median(links_per_route))
    print_stat("Links per route (max)", np.max(links_per_route))
    print_stat("Single-link routes", int(np.sum(links_per_route <= 1)))
    print()
    if durations:
        print_stat("Route duration mean", np.mean(durations), "s")
        print_stat("Route duration median", np.median(durations), "s")
        print_stat("Route duration max", np.max(durations), "s")
        print()
    print_stat("Unique links traversed", coverage)
    print_stat("Network coverage", coverage_pct, "%")
    print_stat("Never-traversed links", N_graph - coverage)
    print()
    print_stat("Link traversal mean", np.mean(traversal_values), "visits")
    print_stat("Link traversal median", np.median(traversal_values), "visits")
    print_stat("Link traversal max", np.max(traversal_values), "visits")

    print("\n  Top 10 most traversed links:")
    for lid, count in top_10_links:
        print(f"    Link {lid}: {count:>8} traversals")

    print("\n  Routes by hour of day:")
    for h in range(24):
        count = routes_per_hour.get(h, 0)
        bar = '#' * (count // max(1, max(routes_per_hour.values()) // 40))
        print(f"    {h:02d}:00  {count:>7}  {bar}")

    stats = {
        'total_routes': num_routes,
        'links_per_route': percentiles(links_per_route),
        'route_duration_sec': percentiles(durations) if durations else {},
        'network_coverage_links': coverage,
        'network_coverage_pct': coverage_pct,
        'link_traversal_counts': percentiles(traversal_values),
        'top_10_links': {lid: count for lid, count in top_10_links},
        'routes_per_hour': {str(h): routes_per_hour.get(h, 0) for h in range(24)},
    }

    return stats, links_per_route, durations, traversal_values, routes_per_hour


def analyze_popularity(matrix, times, link_ids, label):
    """Analyze a popularity matrix (raw or smoothed)."""
    print_section(f"POPULARITY MATRIX ({label})")

    n_times, n_links = matrix.shape

    # Parse times for temporal analysis
    datetimes = []
    for t in times:
        try:
            datetimes.append(datetime.strptime(t, "%Y-%m-%d %H:%M:%S"))
        except:
            pass

    # Time range
    if datetimes:
        print_stat("Time range start", str(datetimes[0]))
        print_stat("Time range end", str(datetimes[-1]))
        total_days = (datetimes[-1] - datetimes[0]).total_seconds() / 86400
        print_stat("Total days covered", f"{total_days:.1f}")
    print_stat("Total timeframes", n_times)
    print_stat("Total links", n_links)
    print()

    # Sparsity
    nnz = matrix.nnz
    total_elements = n_times * n_links
    density = nnz / total_elements * 100
    print_stat("Non-zero entries", nnz)
    print_stat("Matrix density", density, "%")
    print_stat("Sparsity", 100 - density, "%")
    print()

    # Per-timeframe stats
    nnz_per_time = np.array([matrix.getrow(i).nnz for i in range(min(n_times, 200))])
    sums_per_time = np.array([matrix.getrow(i).sum() for i in range(min(n_times, 200))])

    print_stat("Active links per timeframe (mean)", np.mean(nnz_per_time))
    print_stat("Active links per timeframe (median)", np.median(nnz_per_time))
    print_stat("Active links per timeframe (min)", np.min(nnz_per_time))
    print_stat("Active links per timeframe (max)", np.max(nnz_per_time))
    print()

    # Per-link stats (sum across all timeframes)
    link_totals = np.array(matrix.sum(axis=0)).flatten()
    active_links = int(np.sum(link_totals > 0))

    print_stat("Links with any traffic", active_links)
    print_stat("Links with zero traffic", n_links - active_links)
    print_stat("Link total traffic (mean)", np.mean(link_totals))
    print_stat("Link total traffic (median)", np.median(link_totals))
    print_stat("Link total traffic (max)", np.max(link_totals))

    # Concentration: what fraction of links carry what fraction of traffic
    sorted_totals = np.sort(link_totals)[::-1]
    cumsum = np.cumsum(sorted_totals)
    total_traffic = cumsum[-1] if cumsum[-1] > 0 else 1.0

    for pct in [1, 5, 10, 20, 50]:
        n_links_pct = max(1, int(n_links * pct / 100))
        traffic_share = cumsum[n_links_pct - 1] / total_traffic * 100
        print_stat(f"Top {pct}% links carry", traffic_share, "% of traffic")

    # Temporal patterns (by hour)
    if datetimes:
        hourly_traffic = defaultdict(float)
        hourly_count = defaultdict(int)
        for i in range(min(n_times, len(datetimes))):
            h = datetimes[i].hour
            hourly_traffic[h] += float(matrix.getrow(i).sum())
            hourly_count[h] += 1

        print("\n  Average traffic by hour of day:")
        max_traffic = max(hourly_traffic.values()) / max(1, max(hourly_count.values()))
        for h in range(24):
            avg = hourly_traffic.get(h, 0) / max(1, hourly_count.get(h, 1))
            bar = '#' * int(avg / max(1, max_traffic) * 40) if max_traffic > 0 else ''
            print(f"    {h:02d}:00  {avg:>10.1f}  {bar}")

    stats = {
        'shape': [n_times, n_links],
        'non_zero_entries': nnz,
        'density_pct': density,
        'active_links': active_links,
        'active_links_per_timeframe': percentiles(nnz_per_time),
        'link_total_traffic': percentiles(link_totals),
    }

    return stats, link_totals, nnz_per_time


def analyze_pagerank(pr_vector, graph_data, smoothed_link_totals, smoothed_link_ids):
    """Analyze the PageRank output."""
    print_section("PAGERANK OUTPUT ANALYSIS")

    links = list(graph_data['links'].keys())
    N = len(links)

    pr_values = np.array([pr_vector.get(lid, 0.0) for lid in links])

    print_stat("Total links", N)
    print_stat("Non-zero PR values", int(np.sum(pr_values > 0)))
    print_stat("PR sum", np.sum(pr_values))
    print_stat("PR mean", np.mean(pr_values))
    print_stat("PR median", np.median(pr_values))
    print_stat("PR max", np.max(pr_values))
    print_stat("PR min (non-zero)", np.min(pr_values[pr_values > 0]) if np.any(pr_values > 0) else 0)
    print_stat("PR max / PR mean ratio", np.max(pr_values) / max(np.mean(pr_values), 1e-15))
    print()

    # Concentration
    sorted_pr = np.sort(pr_values)[::-1]
    cumsum = np.cumsum(sorted_pr)
    total = cumsum[-1] if cumsum[-1] > 0 else 1.0
    for pct in [1, 5, 10, 20]:
        n = max(1, int(N * pct / 100))
        share = cumsum[n - 1] / total * 100
        print_stat(f"Top {pct}% links hold", share, "% of PR mass")

    # Gini coefficient
    sorted_vals = np.sort(pr_values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n if np.sum(sorted_vals) > 0 else 0
    print_stat("Gini coefficient", gini)
    print()

    # Top 20 links by PageRank
    top_idx = np.argsort(pr_values)[::-1][:20]
    print("  Top 20 links by PageRank:")
    print(f"    {'Rank':>4}  {'Link ID':>12}  {'PR Score':>12}  {'Length':>8}  {'Speed':>8}  {'Lanes':>5}")
    for rank, idx in enumerate(top_idx, 1):
        lid = links[idx]
        info = graph_data['links'][lid]
        print(f"    {rank:>4}  {lid:>12}  {pr_values[idx]:>12.8f}  {info.get('length', 0):>7.1f}m  {info.get('speed', 0):>7.2f}  {info.get('lanes', 1):>5.1f}")

    # Correlation with link properties
    speeds = np.array([graph_data['links'][lid].get('speed', 11.17) for lid in links])
    lengths_arr = np.array([graph_data['links'][lid].get('length', 100.0) for lid in links])
    lanes_arr = np.array([graph_data['links'][lid].get('lanes', 1.0) for lid in links])
    travel_times_arr = lengths_arr / np.maximum(speeds, 0.1)

    mask = pr_values > 0
    if np.sum(mask) > 100:
        from scipy.stats import spearmanr
        corr_speed, _ = spearmanr(pr_values[mask], speeds[mask])
        corr_length, _ = spearmanr(pr_values[mask], lengths_arr[mask])
        corr_lanes, _ = spearmanr(pr_values[mask], lanes_arr[mask])
        corr_tt, _ = spearmanr(pr_values[mask], travel_times_arr[mask])

        print(f"\n  Spearman rank correlations (PR vs link property):")
        print_stat("PR vs Speed", corr_speed)
        print_stat("PR vs Length", corr_length)
        print_stat("PR vs Lanes", corr_lanes)
        print_stat("PR vs Travel Time", corr_tt)

    stats = {
        'non_zero': int(np.sum(pr_values > 0)),
        'pagerank_values': percentiles(pr_values),
        'gini_coefficient': float(gini),
    }

    return stats, pr_values


def generate_plots(network_data, route_data, pop_data, pr_data):
    """Generate all analysis plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use('ggplot')

    speeds, lengths, lanes, travel_times, out_degrees, in_degrees = network_data

    # 1. Network property distributions (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Road Network Properties', fontsize=16, fontweight='bold')

    axes[0, 0].hist(lengths, bins=80, color='steelblue', edgecolor='black', linewidth=0.3, log=True)
    axes[0, 0].set_xlabel('Link Length (m)')
    axes[0, 0].set_ylabel('Count (log)')
    axes[0, 0].set_title('Link Length Distribution')

    axes[0, 1].hist(speeds * 3.6, bins=50, color='coral', edgecolor='black', linewidth=0.3)
    axes[0, 1].set_xlabel('Speed (km/h)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Speed Distribution')

    axes[1, 0].hist(travel_times, bins=80, color='mediumpurple', edgecolor='black', linewidth=0.3, log=True)
    axes[1, 0].set_xlabel('Travel Time (s)')
    axes[1, 0].set_ylabel('Count (log)')
    axes[1, 0].set_title('Travel Time Distribution')

    axes[1, 1].hist(out_degrees, bins=range(0, int(max(out_degrees)) + 2), color='seagreen', edgecolor='black', linewidth=0.3)
    axes[1, 1].set_xlabel('Out-Degree')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Out-Degree Distribution')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_network_properties.png', dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Route statistics (if available)
    if route_data:
        links_per_route, durations, traversal_values, routes_per_hour = route_data

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Route & Traversal Statistics', fontsize=16, fontweight='bold')

        axes[0, 0].hist(links_per_route, bins=80, color='steelblue', edgecolor='black', linewidth=0.3, log=True)
        axes[0, 0].set_xlabel('Links per Route')
        axes[0, 0].set_ylabel('Count (log)')
        axes[0, 0].set_title('Route Complexity')

        if durations:
            axes[0, 1].hist(durations, bins=80, color='coral', edgecolor='black', linewidth=0.3, log=True)
            axes[0, 1].set_xlabel('Duration (s)')
            axes[0, 1].set_ylabel('Count (log)')
            axes[0, 1].set_title('Route Duration')

        axes[1, 0].hist(traversal_values, bins=80, color='mediumpurple', edgecolor='black', linewidth=0.3, log=True)
        axes[1, 0].set_xlabel('Traversal Count')
        axes[1, 0].set_ylabel('Number of Links (log)')
        axes[1, 0].set_title('Link Traversal Frequency')

        hours = sorted(routes_per_hour.keys())
        counts = [routes_per_hour[h] for h in hours]
        axes[1, 1].bar(hours, counts, color='seagreen', edgecolor='black', linewidth=0.3)
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Routes')
        axes[1, 1].set_title('Temporal Distribution of Routes')
        axes[1, 1].set_xticks(range(0, 24, 3))

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '2_route_statistics.png', dpi=200, bbox_inches='tight')
        plt.close()

    # 3. Popularity analysis
    if pop_data:
        link_totals, nnz_per_time = pop_data

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Popularity Matrix Analysis', fontsize=16, fontweight='bold')

        sorted_totals = np.sort(link_totals[link_totals > 0])[::-1]
        axes[0].plot(range(len(sorted_totals)), sorted_totals, color='steelblue', linewidth=1)
        axes[0].set_xlabel('Link Rank')
        axes[0].set_ylabel('Total Traffic')
        axes[0].set_title('Traffic Concentration (Rank-Ordered)')
        axes[0].set_yscale('log')

        axes[1].hist(nnz_per_time, bins=30, color='coral', edgecolor='black', linewidth=0.3)
        axes[1].set_xlabel('Active Links per Timeframe')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Temporal Activity')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '3_popularity_analysis.png', dpi=200, bbox_inches='tight')
        plt.close()

    # 4. PageRank output
    if pr_data is not None:
        pr_values = pr_data

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('PageRank Output Analysis', fontsize=16, fontweight='bold')

        sorted_pr = np.sort(pr_values[pr_values > 0])[::-1]
        axes[0].plot(range(len(sorted_pr)), sorted_pr, color='steelblue', linewidth=1)
        axes[0].set_xlabel('Link Rank')
        axes[0].set_ylabel('PageRank Score')
        axes[0].set_title('PageRank Distribution (Rank-Ordered)')
        axes[0].set_yscale('log')

        cumsum = np.cumsum(sorted_pr) / np.sum(sorted_pr) * 100
        axes[1].plot(np.arange(len(cumsum)) / len(cumsum) * 100, cumsum, color='coral', linewidth=2)
        axes[1].set_xlabel('% of Links (ranked)')
        axes[1].set_ylabel('Cumulative % of PageRank Mass')
        axes[1].set_title('PageRank Lorenz Curve')
        axes[1].plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect equality')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / '4_pagerank_analysis.png', dpi=200, bbox_inches='tight')
        plt.close()

    print(f"\n  Plots saved to {OUTPUT_DIR}/")


def main():
    all_stats = {}

    # 1. Road network
    print("Loading road network...")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    net_stats, speeds, lengths, lanes, travel_times, out_deg, in_deg = analyze_network(graph_data)
    all_stats['network'] = net_stats

    # 2. Matched routes
    route_plot_data = None
    if config.MATCHED_ROUTES.exists():
        print("Loading matched routes...")
        with open(config.MATCHED_ROUTES) as f:
            matched_json = json.load(f)
        graph_links = list(graph_data['links'].keys())
        route_stats, links_per_route, durations, trav_vals, routes_per_hour = analyze_matched_routes(matched_json, graph_links)
        all_stats['matched_routes'] = route_stats
        route_plot_data = (links_per_route, durations, trav_vals, routes_per_hour)
    else:
        print(f"  [SKIP] {config.MATCHED_ROUTES} not found")

    # 3. Raw popularity
    pop_plot_data = None
    if config.POPULARITY_RAW_NPZ.exists():
        print("Loading raw popularity matrix...")
        matrix, times, link_ids = load_npz(config.POPULARITY_RAW_NPZ)
        pop_stats, link_totals, nnz_per_time = analyze_popularity(matrix, times, link_ids, "RAW")
        all_stats['popularity_raw'] = pop_stats
        pop_plot_data = (link_totals, nnz_per_time)
    else:
        print(f"  [SKIP] {config.POPULARITY_RAW_NPZ} not found")

    # 4. Smoothed popularity
    if config.POPULARITY_NPZ.exists():
        print("Loading smoothed popularity matrix...")
        matrix_s, times_s, link_ids_s = load_npz(config.POPULARITY_NPZ)
        smooth_stats, smooth_totals, smooth_nnz = analyze_popularity(matrix_s, times_s, link_ids_s, "SMOOTHED gamma=0.26")
        all_stats['popularity_smoothed'] = smooth_stats
    else:
        print(f"  [SKIP] {config.POPULARITY_NPZ} not found")
        smooth_totals, link_ids_s = None, None

    # 5. PageRank output
    pr_plot_data = None
    if config.OUTPUT_FILE.exists():
        print("Loading PageRank output...")
        with open(config.OUTPUT_FILE) as f:
            pr_vector = json.load(f)
        pr_stats, pr_values = analyze_pagerank(pr_vector, graph_data, smooth_totals, link_ids_s)
        all_stats['pagerank'] = pr_stats
        pr_plot_data = pr_values
    else:
        print(f"  [SKIP] {config.OUTPUT_FILE} not found")

    # Generate plots
    print_section("GENERATING PLOTS")
    generate_plots(
        (speeds, lengths, lanes, travel_times, out_deg, in_deg),
        route_plot_data,
        pop_plot_data,
        pr_plot_data,
    )

    # Save stats
    with open(STATS_FILE, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Full statistics saved to {STATS_FILE}")


if __name__ == '__main__':
    main()
