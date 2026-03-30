#!/usr/bin/env python3
"""
Run cross-week predictive PageRank validation.
Averages a specific timeframe from Week 1 and Week 2, feeds it to the model,
and compares the stationary distribution against the ground truth of Week 3.
"""
import json
import sys
import numpy as np
import datetime
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

def main():
    # Define the base timeframes for our experiment
    # For instance: Wednesday at 08:00:00
    # September 2018:
    # 2018-09-01 is Saturday
    # Week 1 Wed: 2018-09-05 08:00:00
    # Week 2 Wed: 2018-09-12 08:00:00
    # Week 3 Wed: 2018-09-19 08:00:00

    t1 = '2018-09-05 08:00:00' # Week 1
    t2 = '2018-09-12 08:00:00' # Week 2
    t3 = '2018-09-19 08:00:00' # Week 3 (Ground Truth target)

    print(f"Predictive Cross-Week Validation\n" + "-"*40)
    print(f"Input Week 1: {t1}")
    print(f"Input Week 2: {t2}")
    print(f"Target Week 3: {t3}\n")

    # Load graph once
    with open(config.GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)
    links = list(graph_data['links'].keys())
    N = len(links)

    # Build M_2N once
    print(f"Building Two-Phase Matrix (N={N})...", flush=True)
    M_2N = build_two_phase_matrix(graph_data)

    results = []

    for label, npz_path in DATASETS:
        print(f"\nProcessing {label}...")

        # Get E_N for Week 1 and Week 2, passing npz_path directly
        E_N_w1 = build_teleportation_vector(graph_data, t1, npz_path=npz_path)
        E_N_w2 = build_teleportation_vector(graph_data, t2, npz_path=npz_path)

        # Get E_N for Week 3 (Our Ground Truth)
        E_N_w3_target = build_teleportation_vector(graph_data, t3, npz_path=npz_path)

        # Average Week 1 and Week 2 to create the predictive input
        E_N_input = (E_N_w1 + E_N_w2) / 2.0

        # Normalize the averaged input (just to be safe)
        if E_N_input.sum() > 0:
            E_N_input = E_N_input / sum(E_N_input)

        # Apply sharpening and boosting to the averaged input
        E_tele = sharpen_teleportation(E_N_input, PARAMS['tau'])
        E_tele = boost_top_k(E_tele, PARAMS['top_k'], PARAMS['top_k_boost'])
        E_2N = np.concatenate([E_tele, np.zeros(N)])

        # Run the power iteration to get the stationary distribution prediction
        v_2N = run_power_iteration(M_2N, E_2N)

        # Collapse to N dimensions
        v_final = v_2N[:N] + v_2N[N:]
        v_final = v_final / np.sum(v_final)

        # Compute MRE against the TARGET (Week 3)
        # Note the denominator is the target truth (E_N_w3_target) + epsilon
        mre_full = 0
        for i in range(N):
            mre_full += abs(E_N_w3_target[i] - v_final[i]) / (E_N_w3_target[i] + 1e-9)
        mre_full /= N

        top_100_indices = np.argsort(E_N_w3_target)[::-1][:100]
        t100 = E_N_w3_target[top_100_indices]
        p100 = v_final[top_100_indices]
        mre_100 = np.mean(np.abs(t100 - p100) / (t100 + 1e-9))

        results.append((label, mre_full, mre_100))

    # Print results table
    print("\n" + "=" * 75)
    print(f"RESULTS: Predictive Cross-Week Power")
    print("=" * 75)
    print(f"{'Dataset':<28} {'Full MRE (vs W3)':<20} {'Top-100 MRE (vs W3)':<20}")
    print("-" * 75)
    for label, mre_f, mre_100 in results:
        print(f"{label:<28} {mre_f:<20.4f} {mre_100:<20.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
