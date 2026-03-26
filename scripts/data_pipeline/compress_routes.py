import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import glob
from datetime import datetime
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# Configuration
INPUT_DIR = 'Combined'
OUTPUT_FILE = '../../data/compressed_routes.parquet'
BASE_DATE = datetime(2018, 1, 1)
BATCH_SIZE = 20  # Files per worker task

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def process_batch(file_paths):
    """
    Process a batch of files.
    Returns: (DataFrame with local route_id 0..N, total_routes)
    """
    results = []
    local_route_counter = 0
    
    for file_path in file_paths:
        try:
            df = pd.read_csv(
                file_path, 
                header=None, 
                names=['did', 'ts_str', 'lat', 'lon'],
                usecols=[0, 1, 2, 3],
                dtype={'did': str, 'ts_str': str, 'lat': float, 'lon': float},
                engine='pyarrow'
            )
            
            # 1. Cleaning
            timestamps = pd.to_datetime(df['ts_str'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            valid_mask = (df['lat'].notna()) & (df['lon'].notna()) & (timestamps.notna())
            df = df[valid_mask].copy()
            timestamps = timestamps[valid_mask]
            
            if df.empty:
                continue
                
            df['ts'] = timestamps

            # 2. Speed Filter (< 2 m/s)
            prev_did = df['did'].shift(1)
            same_device = df['did'] == prev_did
            
            lat1 = df['lat'].shift(1)
            lon1 = df['lon'].shift(1)
            dist = haversine_vectorized(lat1, lon1, df['lat'], df['lon'])
            time_delta = df['ts'].diff().dt.total_seconds()
            
            speed = pd.Series(np.inf, index=df.index)
            valid_calc = same_device & (time_delta > 0)
            if valid_calc.any():
                speed[valid_calc] = dist[valid_calc] / time_delta[valid_calc]
            
            # Handle dupes (dist ~ 0, time = 0)
            same_time = same_device & (time_delta == 0)
            if same_time.any():
                speed[same_time & (dist < 1.0)] = 0.0
                
            filtered_df = df[speed >= 2.0].copy()
            
            if filtered_df.empty:
                continue
                
            # 3. Route Splitting (Gap > 5 min)
            r_did = filtered_df['did']
            r_ts = filtered_df['ts']
            
            did_changed = r_did != r_did.shift(1)
            t_diff = r_ts.diff().dt.total_seconds()
            large_gap = t_diff > 300
            same_dev_split = (r_did == r_did.shift(1))
            
            condition = did_changed | (large_gap & same_dev_split)
            local_groups = condition.cumsum()
            
            # Assign offset relative to this batch's running counter
            filtered_df['route_id'] = local_groups + local_route_counter
            
            # Update counter
            # local_groups is 1-based, max value is num routes found
            num_routes = local_groups.max()
            local_route_counter += num_routes
            
            # Prep output
            filtered_df['time_sec'] = (filtered_df['ts'] - BASE_DATE).dt.total_seconds().astype('uint32')
            results.append(filtered_df[['route_id', 'time_sec', 'lat', 'lon']])
            
        except Exception as e:
            print(f"Error in {file_path}: {e}")
            
    if not results:
        return None, 0
        
    final_df = pd.concat(results)
    return final_df, local_route_counter

def main():
    schema = pa.schema([
        ('route_id', pa.uint32()),
        ('time_sec', pa.uint32()),
        ('lat', pa.float64()),
        ('lon', pa.float64())
    ])
    
    search_path = os.path.join(INPUT_DIR, 'wp-snapped-*.csv')
    csv_files = sorted(glob.glob(search_path))
    print(f"Found {len(csv_files)} files.")
    
    # Create batches
    # Create batches
    batches = [csv_files[i:i + BATCH_SIZE] for i in range(0, len(csv_files), BATCH_SIZE)]
    print(f"Processing {len(batches)} batches with Multiprocessing (FULL DATASET)...")

    writer = pq.ParquetWriter(OUTPUT_FILE, schema, compression='zstd')
    global_route_counter = 0
    
    # Use max_workers=8 (M1/M2/M3 usually have 8+ cores)
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        # We need to preserve order? No, files are independent.
        # BUT Route IDs need to be sequential?
        # If we want purely sequential Route IDs (1, 2, 3...) we should process in order.
        # Executor.map guarantees order of results!
        
        results = executor.map(process_batch, batches)
        
        count = 0
        for batch_df, num_routes in results:
            count += 1
            if batch_df is not None:
                # Adiust Route IDs to be global
                batch_df['route_id'] += global_route_counter
                global_route_counter += num_routes
                
                table = pa.Table.from_pandas(batch_df, schema=schema, preserve_index=False)
                writer.write_table(table)
            
            if count % 5 == 0:
                print(f"Processed batch {count}/{len(batches)}. Total Routes: {global_route_counter}")

    writer.close()
    print(f"Done. Output: {OUTPUT_FILE}")
    print(f"Total Routes: {global_route_counter}")

if __name__ == '__main__':
    # Fix for macOS multiprocessing
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
