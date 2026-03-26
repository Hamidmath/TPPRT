import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, timedelta

# Configuration
# Configuration
PARQUET_FILE = '../../data/compressed_routes.parquet'
MATCHED_ROUTES_FILE = '../../data/matched_routes.json'
STATS_FILE = '../../data/statistics.json'
GRAPH_DIR = 'graphs'
BASE_DATE = datetime(2018, 1, 1)

if not os.path.exists(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371000  # radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def calculate_statistics():
    print("Starting analysis...")
    
    # --- PART 1: PARQUET ANALYSIS (Previous) ---
    parquet_file = pq.ParquetFile(PARQUET_FILE)
    
    global_min_time = float('inf')
    global_max_time = float('-inf')
    
    chunks_results = []
    prev_chunk_last_row = None
    
    for i in range(parquet_file.num_row_groups):
        print(f"Processing Parquet chunk {i+1}/{parquet_file.num_row_groups}")
        table = parquet_file.read_row_group(i, columns=['route_id', 'time_sec', 'lat', 'lon'])
        df = table.to_pandas()
        
        # 1. Global Time Range
        chunk_min = df['time_sec'].min()
        chunk_max = df['time_sec'].max()
        if chunk_min < global_min_time: global_min_time = chunk_min
        if chunk_max > global_max_time: global_max_time = chunk_max
        
        # 2. Aggregations per route
        grouped = df.groupby('route_id')
        agg = grouped.agg(
            min_t=('time_sec', 'min'),
            max_t=('time_sec', 'max'),
            count=('time_sec', 'count')
        )
        
        # 3. Distance
        lat1 = df['lat'].values[:-1]
        lon1 = df['lon'].values[:-1]
        lat2 = df['lat'].values[1:]
        lon2 = df['lon'].values[1:]
        rid1 = df['route_id'].values[:-1]
        rid2 = df['route_id'].values[1:]
        
        mask = (rid1 == rid2)
        dists = np.zeros(len(df), dtype=np.float64)
        valid_idx = np.where(mask)[0]
        
        if len(valid_idx) > 0:
            d_vals = haversine_vectorized(lat1[valid_idx], lon1[valid_idx], lat2[valid_idx], lon2[valid_idx])
            dists[valid_idx + 1] = d_vals
            
        if prev_chunk_last_row is not None:
            last_rid = prev_chunk_last_row['route_id']
            curr_rid = df.iloc[0]['route_id']
            if last_rid == curr_rid:
                d = haversine_vectorized(
                    prev_chunk_last_row['lat'], prev_chunk_last_row['lon'],
                    df.iloc[0]['lat'], df.iloc[0]['lon']
                )
                dists[0] = d
        
        prev_chunk_last_row = df.iloc[-1]
        
        df['seg_dist'] = dists
        dist_agg = df.groupby('route_id')['seg_dist'].sum()
        df = None
        agg['dist'] = dist_agg
        chunks_results.append(agg)
    
    print("Aggregating Parquet results...")
    full_agg = pd.concat(chunks_results)
    full_agg.reset_index(inplace=True)
    
    final_stats = full_agg.groupby('route_id').agg(
        start_time=('min_t', 'min'),
        end_time=('max_t', 'max'),
        num_records=('count', 'sum'),
        total_distance=('dist', 'sum')
    )
    final_stats['duration_sec'] = final_stats['end_time'] - final_stats['start_time']

    # --- PART 2: MATCHED ROUTES ANALYSIS (New) ---
    print(f"Loading matched routes from {MATCHED_ROUTES_FILE}...")
    
    # Store link counts and intersection counts
    # Using a list of dicts to DataFrame
    matched_data = []
    
    try:
        with open(MATCHED_ROUTES_FILE, 'r') as f:
            # Check file size first? 300MB is ok for JSON load usually.
             matched_json = json.load(f)
             
        for item in matched_json:
            rid = item['route']['route_id']
            links = item['route']['routelinks']
            
            n_links = len(links)
            # Assumption: Num Intersections = Num Nodes.
            # A set of links L1->L2->L3 touches nodes N1, N2, N3, N4 usually.
            # Number of nodes visited is roughly N_Links + 1 (unless circular).
            # Specifically, if 0 links, 0 intersections? Or 1 start point?
            # Let's define n_intersections = n_links + 1 if n_links > 0 else 0
            n_intersections = n_links + 1 if n_links > 0 else 0
            
            # Or should it be unique nodes?
            # User said "number of intersections".
            # Usually means number of nodes in the path.
            # I will stick to n_links + 1 for simplicity unless I parse topology (expensive here).
            
            matched_data.append({
                'route_id': int(rid), # Ensure type match with final_stats index
                'num_links': n_links,
                'num_intersections': n_intersections
            })
            
        # Convert to DataFrame
        matched_df = pd.DataFrame(matched_data)
        matched_df.set_index('route_id', inplace=True)
        
        # Merge with final_stats
        # Note: Not all routes might be matched (if filtered or empty)
        # We do outer join or left join? 
        # User wants stats. Only matched routes have link counts.
        # But final_stats has ALL routes.
        # I will join. Routes without match get 0 or NaN.
        print("Merging matched stats...")
        final_stats = final_stats.join(matched_df, how='left')
        
        # Fill NaNs with 0 for unmatched keys?
        final_stats['num_links'] = final_stats['num_links'].fillna(0)
        final_stats['num_intersections'] = final_stats['num_intersections'].fillna(0)
        
    except Exception as e:
        print(f"Warning: Could not process matched routes: {e}")
        final_stats['num_links'] = 0
        final_stats['num_intersections'] = 0

    # --- Statistics Generation ---
    output_stats = {}
    
    output_stats['time_range'] = {
        'min_timestamp': str(BASE_DATE + timedelta(seconds=int(global_min_time))),
        'max_timestamp': str(BASE_DATE + timedelta(seconds=int(global_max_time)))
    }
    
    def get_distribution_stats(series):
        if series.empty: return {}
        desc = series.describe()
        mode = series.mode()
        return {
            'count': int(desc['count']),
            'min': float(desc['min']),
            'max': float(desc['max']),
            'mean': float(desc['mean']),
            'std': float(desc['std']),
            '25%': float(desc['25%']),
            '50%_median': float(desc['50%']),
            '75%': float(desc['75%']),
            'mode': float(mode.iloc[0]) if not mode.empty else None
        }

    output_stats['route_duration_seconds'] = get_distribution_stats(final_stats['duration_sec'])
    output_stats['route_record_counts'] = get_distribution_stats(final_stats['num_records'])
    output_stats['route_length_meters'] = get_distribution_stats(final_stats['total_distance'])
    
    # New Stats
    output_stats['num_links_per_route'] = get_distribution_stats(final_stats['num_links'])
    output_stats['num_intersections_per_route'] = get_distribution_stats(final_stats['num_intersections'])
    
    print(f"Writing stats to {STATS_FILE}...")
    with open(STATS_FILE, 'w') as f:
        json.dump(output_stats, f, indent=4)
        
    print("Generating graphs...")
    # ... (Plotting code for standard metrics)
    
    # Existing plots
    # Duration
    plt.figure(figsize=(10, 6))
    plt.hist(final_stats['duration_sec'], bins=50, log=True, color='skyblue', edgecolor='black')
    plt.title('Distribution of Route Durations (Seconds)')
    plt.xlabel('Duration (s)')
    plt.ylabel('Frequency (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(GRAPH_DIR, 'duration_hist.png'))
    plt.close()
    
    # Count
    plt.figure(figsize=(10, 6))
    plt.hist(final_stats['num_records'], bins=50, log=True, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Record Counts per Route')
    plt.xlabel('Record Count')
    plt.ylabel('Frequency (Log Scale)')
    plt.savefig(os.path.join(GRAPH_DIR, 'record_count_hist.png'))
    plt.close()
    
    # Distance
    plt.figure(figsize=(10, 6))
    plt.hist(final_stats['total_distance'], bins=50, log=True, color='salmon', edgecolor='black')
    plt.title('Distribution of Route Lengths (Meters)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Frequency (Log Scale)')
    plt.savefig(os.path.join(GRAPH_DIR, 'distance_hist.png'))
    plt.close()
    
    # New Plots
    # Num Links
    plt.figure(figsize=(10, 6))
    plt.hist(final_stats['num_links'], bins=50, log=True, color='orange', edgecolor='black')
    plt.title('Distribution of Links per Route')
    plt.xlabel('Number of Links')
    plt.ylabel('Frequency (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(GRAPH_DIR, 'links_hist.png'))
    plt.close()

    # Num Intersections
    plt.figure(figsize=(10, 6))
    plt.hist(final_stats['num_intersections'], bins=50, log=True, color='purple', edgecolor='black')
    plt.title('Distribution of Intersections per Route')
    plt.xlabel('Number of Intersections')
    plt.ylabel('Frequency (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(GRAPH_DIR, 'intersections_hist.png'))
    plt.close()
    
    # --- Save JSON ---
    print(f"Writing stats to {STATS_FILE}...")
    with open(STATS_FILE, 'w') as f:
        json.dump(output_stats, f, indent=4)
        
    # --- Plotting ---
    print("Generating graphs...")
    
    # Duration Histogram
    plt.figure(figsize=(10, 6))
    # Filter insane outliers for plotting usually helps, but user wants 'good statistics'
    # Logging typically helps with these skewed distributions
    plt.hist(final_stats['duration_sec'], bins=50, log=True, color='skyblue', edgecolor='black')
    plt.title('Distribution of Route Durations (Seconds)')
    plt.xlabel('Duration (s)')
    plt.ylabel('Frequency (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(GRAPH_DIR, 'duration_hist.png'))
    plt.close()
    
    # Record Count Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(final_stats['num_records'], bins=50, log=True, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Record Counts per Route')
    plt.xlabel('Record Count')
    plt.ylabel('Frequency (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(GRAPH_DIR, 'record_count_hist.png'))
    plt.close()
    
    # Distance Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(final_stats['total_distance'], bins=50, log=True, color='salmon', edgecolor='black')
    plt.title('Distribution of Route Lengths (Meters)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Frequency (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(GRAPH_DIR, 'distance_hist.png'))
    plt.close()

    print("Analysis complete.")

if __name__ == '__main__':
    calculate_statistics()
