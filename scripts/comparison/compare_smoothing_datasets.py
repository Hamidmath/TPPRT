#!/usr/bin/env python3
"""
Run Two-Phase PageRank on 49 random timeframes for 3 datasets.
Uses the exact logic from the core PageRank module.
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core import build_two_phase_matrix, build_teleportation_vector, run_power_iteration, PARAMS
from core import sharpen_teleportation, boost_top_k
import config

DATASETS = [
    ('Raw (no smoothing)',    str(config.POPULARITY_RAW_NPZ)),
    ('Smoothed (gamma=0.10)', str(config.POPULARITY_NPZ)),
    ('Smoothed (gamma=0.26)', str(config.POPULARITY_NPZ_026)),
]

N_TRIALS = 49
SEED = 123

def main():
    # Load graph once
    with open(config.GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)
    links = list(graph_data['links'].keys())
    N = len(links)

    # Build M_2N once
    print(f"Building Two-Phase Matrix (N={N})...", flush=True)
    M_2N = build_two_phase_matrix(graph_data)

    # Get timeframe list from the smoothed_026 dataset
    loader = np.load(str(config.POPULARITY_NPZ_026), allow_pickle=True)
    all_times = list(loader['times'])

    # Pick 49 random timeframes
    rng = np.random.RandomState(SEED)
    trial_indices = rng.choice(len(all_times), size=N_TRIALS, replace=False)
    trial_times = [all_times[i] for i in trial_indices]
    print(f"Selected {N_TRIALS} timeframes (seed={SEED})\n", flush=True)

    # Run all trials
    results = {label: {'full': [], 'top100': []} for label, _ in DATASETS}

    for t_i, target_time in enumerate(trial_times):
        if t_i % 10 == 0:
            print(f"  Trial {t_i+1}/{N_TRIALS} ({target_time})...", flush=True)

        for label, npz_path in DATASETS:
            # Use core's function to build E_N, passing npz_path directly
            E_N = build_teleportation_vector(graph_data, target_time, npz_path=npz_path)

            # Apply the same sharpening and boosting as in main()
            E_tele = sharpen_teleportation(E_N, PARAMS['tau'])
            E_tele = boost_top_k(E_tele, PARAMS['top_k'], PARAMS['top_k_boost'])
            E_2N = np.concatenate([E_tele, np.zeros(N)])

            # Run the same power iteration
            v_2N = run_power_iteration(M_2N, E_2N)

            # Collapse to N dimensions (same as main())
            v_final = v_2N[:N] + v_2N[N:]
            v_final = v_final / np.sum(v_final)

            # Compute MRE (same formula as main())
            mre_full = 0
            for i in range(N):
                mre_full += abs(E_N[i] - v_final[i]) / (E_N[i] + 1e-9)
            mre_full /= N

            top_100_indices = np.argsort(E_N)[::-1][:100]
            t100 = E_N[top_100_indices]
            p100 = v_final[top_100_indices]
            mre_100 = np.mean(np.abs(t100 - p100) / (t100 + 1e-9))

            results[label]['full'].append(mre_full)
            results[label]['top100'].append(mre_100)

    # Print results table
    print("\n" + "=" * 78)
    print(f"RESULTS: Average over {N_TRIALS} timeframes (seed={SEED})")
    print("=" * 78)
    print(f"{'Dataset':<28} {'Full MRE':<14} {'Std':<12} {'Top-100 MRE':<14} {'Std':<12}")
    print("-" * 78)
    for label, _ in DATASETS:
        fm = np.array(results[label]['full'])
        tm = np.array(results[label]['top100'])
        print(f"{label:<28} {fm.mean():<14.4f} {fm.std():<12.4f} {tm.mean():<14.4f} {tm.std():<12.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
