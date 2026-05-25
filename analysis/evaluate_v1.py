"""
Evaluate the V1 / single-matrix two-phase PageRank chain with the calibrated
parameters (beta = 0.124, rho = 0.147) and compare its prediction error
against the empirical between-week variability of the ground-truth signal.

This is the "advisor pass criterion" experiment:
    if   model_error  <=  between-week variability,
    then the model is paper-ready.

V1 single-matrix iteration (no separate damping term):

    v_up_new   = (1 - beta) * P_up^T  v_up   +  rho * E_b * sum(v_down)
    v_down_new = beta * v_up           +  (1 - rho) * P_down^T v_down

Run:
    python3 analysis/evaluate_v1.py --num-frames 49 --seed 42
"""

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from core.pagerank import (
    compute_speed_lane_weights,
    build_phase_matrix,
)
from core.io import load_popularity_npz

logging.basicConfig(level=logging.WARNING)


BETA = 0.124
RHO = 0.147
TOL = 1e-7
MAX_ITERS = 300


def build_phase_kernels(graph_data):
    up_weights = compute_speed_lane_weights(graph_data, is_up_phase=True)
    down_weights = compute_speed_lane_weights(graph_data, is_up_phase=False)
    P_up = build_phase_matrix(graph_data, up_weights)
    P_down = build_phase_matrix(graph_data, down_weights)
    return P_up.tocsr(), P_down.tocsr()


def v1_power_iteration(P_up_T, P_down_T, E_b, beta, rho, tol, max_iter):
    """Run V1 single-matrix two-phase power iteration on N-block representation.

    Inputs P_up_T, P_down_T are precomputed transposes (CSR) for fast matvec.
    Returns the joint 2N steady state v = [v_up; v_down] with sum(v) == 1.
    """
    N = E_b.shape[0]
    v_up = E_b.copy()
    v_down = np.zeros(N)

    for _ in range(max_iter):
        s_down = float(v_down.sum())
        v_up_new = (1.0 - beta) * (P_up_T @ v_up) + rho * E_b * s_down
        v_down_new = beta * v_up + (1.0 - rho) * (P_down_T @ v_down)
        # Renormalise to unit total mass to keep iterates bounded under accumulated FP error.
        s = float(v_up_new.sum() + v_down_new.sum())
        if s > 0:
            v_up_new /= s
            v_down_new /= s
        diff = float(np.abs(v_up_new - v_up).sum() + np.abs(v_down_new - v_down).sum())
        v_up, v_down = v_up_new, v_down_new
        if diff < tol:
            break
    return v_up, v_down


def extract_E_N(matrix, t_idx, pop_link_ids, lid_to_idx, N):
    proj = np.fromiter(
        (lid_to_idx.get(lid, -1) for lid in pop_link_ids),
        dtype=np.int64,
        count=len(pop_link_ids),
    )
    row_vals = matrix.getrow(t_idx).toarray().ravel()
    E_N = np.zeros(N)
    valid = proj >= 0
    np.add.at(E_N, proj[valid], row_vals[valid])
    s = E_N.sum()
    if s > 0:
        E_N /= s
    else:
        E_N = np.ones(N) / N
    return E_N


def metrics(truth, pred, top_k=100):
    rel = np.abs(truth - pred) / (truth + 1e-9)
    overall_mre = float(np.mean(rel))
    top_idx = np.argsort(truth)[::-1][:top_k]
    top100_mre = float(np.mean(rel[top_idx]))

    abs_err = np.abs(truth - pred)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(abs_err ** 2)))

    mask = truth > 0
    if mask.sum() > 100:
        pearson_r = float(pearsonr(truth[mask], pred[mask])[0])
        spearman_r = float(spearmanr(truth[mask], pred[mask])[0])
    else:
        pearson_r = 0.0
        spearman_r = 0.0

    pred_top = set(np.argsort(pred)[::-1][:top_k])
    true_top = set(top_idx)
    rank_overlap = len(pred_top & true_top)

    eps = 1e-12
    p = np.maximum(truth, eps)
    q = np.maximum(pred, eps)
    p /= p.sum()
    q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))

    return {
        "overall_mre": overall_mre,
        "top100_mre": top100_mre,
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "rank_overlap_top100": int(rank_overlap),
        "kl_divergence": kl,
    }


def parse_dt(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def find_same_slot_siblings(times, target_dt, max_per_target=8):
    """Return list of timeframe-strings that share weekday + hour:minute with target_dt
    but differ in calendar week. Capped at max_per_target."""
    siblings = []
    for t in times:
        dt = parse_dt(t)
        if dt == target_dt:
            continue
        if dt.weekday() == target_dt.weekday() and dt.hour == target_dt.hour and dt.minute == target_dt.minute:
            siblings.append(t)
            if len(siblings) >= max_per_target:
                break
    return siblings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--siblings-per-frame", type=int, default=8)
    args = parser.parse_args()

    print("Loading data ...")
    with open(config.GRAPH_FILE, "r") as f:
        graph_data = json.load(f)
    links = list(graph_data["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    bundle = load_popularity_npz(str(config.POPULARITY_NPZ))
    matrix = bundle["matrix"]
    times = bundle["times"]
    pop_link_ids = bundle["link_ids"]

    print(f"  N = {N} links, {len(times)} timeframes")

    print("Building phase kernels P_up, P_down ...")
    P_up, P_down = build_phase_kernels(graph_data)
    P_up_T = P_up.T.tocsr()
    P_down_T = P_down.T.tocsr()

    random.seed(args.seed)
    sample = random.sample(times, min(args.num_frames, len(times)))

    print(f"\nV1 single-matrix two-phase chain")
    print(f"  beta = {BETA}, rho = {RHO}")
    print(f"  iteration: v_up_new = (1-beta) P_up^T v_up + rho E_b sum(v_down)")
    print(f"             v_down_new = beta v_up + (1-rho) P_down^T v_down")
    print()

    print(f"{'#':>3} | {'Timeframe':>20} | {'Top100':>8} | {'Overall':>8} | {'Pearson':>8} | {'Rank':>5} | {'sec':>5}")
    print("-" * 75)

    model_metrics = []
    sibling_metrics = []
    t_start = time.time()

    for i, t_str in enumerate(sample):
        t_idx = times.index(t_str)
        E_N = extract_E_N(matrix, t_idx, pop_link_ids, lid_to_idx, N)

        t0 = time.time()
        v_up, v_down = v1_power_iteration(P_up_T, P_down_T, E_N, BETA, RHO, TOL, MAX_ITERS)
        v_total = v_up + v_down
        v_total /= v_total.sum()
        elapsed = time.time() - t0

        m = metrics(E_N, v_total, top_k=args.top_k)
        m["timeframe"] = t_str
        m["seconds"] = elapsed
        model_metrics.append(m)

        # Between-week variability for this slot:
        target_dt = parse_dt(t_str)
        sib_strs = find_same_slot_siblings(times, target_dt, max_per_target=args.siblings_per_frame)
        sib_mre = []
        for sib in sib_strs:
            s_idx = times.index(sib)
            E_sib = extract_E_N(matrix, s_idx, pop_link_ids, lid_to_idx, N)
            sm = metrics(E_N, E_sib, top_k=args.top_k)
            sm["target"] = t_str
            sm["sibling"] = sib
            sibling_metrics.append(sm)
            sib_mre.append(sm["top100_mre"])
        sib_top100_mean = float(np.mean(sib_mre)) if sib_mre else float("nan")
        m["sibling_top100_mre_mean"] = sib_top100_mean
        m["num_siblings"] = len(sib_strs)

        print(f"{i+1:3d} | {t_str:>20} | {m['top100_mre']:8.4f} | {m['overall_mre']:8.4f} | {m['pearson_r']:8.4f} | {m['rank_overlap_top100']:>3}/100 | {elapsed:4.1f}s   "
              f"[siblings n={len(sib_strs):>2}, top100_mre_mean={sib_top100_mean:.4f}]")

    print()
    print(f"Total wall-clock: {time.time() - t_start:.1f} s")

    # Aggregate.
    def agg(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
        }

    model_top100 = [m["top100_mre"] for m in model_metrics]
    model_overall = [m["overall_mre"] for m in model_metrics]
    model_kl = [m["kl_divergence"] for m in model_metrics]
    model_pearson = [m["pearson_r"] for m in model_metrics]
    model_rankov = [m["rank_overlap_top100"] for m in model_metrics]

    sib_top100 = [m["top100_mre"] for m in sibling_metrics]
    sib_overall = [m["overall_mre"] for m in sibling_metrics]
    sib_kl = [m["kl_divergence"] for m in sibling_metrics]
    sib_pearson = [m["pearson_r"] for m in sibling_metrics]

    print()
    print("=" * 70)
    print("  AGGREGATE: V1 model vs ground truth  (per-timeframe predictions)")
    print("=" * 70)
    for name, arr in [
        ("Top-100 MRE",   model_top100),
        ("Overall MRE",   model_overall),
        ("KL divergence", model_kl),
        ("Pearson r",     model_pearson),
        ("Top-100 rank overlap", model_rankov),
    ]:
        a = agg(arr)
        print(f"  {name:<32} mean={a['mean']:8.4f}  std={a['std']:8.4f}  median={a['median']:8.4f}  range=[{a['min']:.4f}, {a['max']:.4f}]")

    print()
    print("=" * 70)
    print("  AGGREGATE: between-week variability  (E_N(t) vs E_N(same-slot, other week))")
    print("=" * 70)
    print(f"  number of (target, sibling) pairs: {len(sibling_metrics)}")
    for name, arr in [
        ("Top-100 MRE",   sib_top100),
        ("Overall MRE",   sib_overall),
        ("KL divergence", sib_kl),
        ("Pearson r",     sib_pearson),
    ]:
        if not arr:
            continue
        a = agg(arr)
        print(f"  {name:<32} mean={a['mean']:8.4f}  std={a['std']:8.4f}  median={a['median']:8.4f}  range=[{a['min']:.4f}, {a['max']:.4f}]")

    # Pass-criterion verdict.
    print()
    print("=" * 70)
    print("  ADVISOR PASS CRITERION")
    print("=" * 70)
    m_top = float(np.mean(model_top100))
    s_top = float(np.mean(sib_top100)) if sib_top100 else float("nan")
    m_ovr = float(np.mean(model_overall))
    s_ovr = float(np.mean(sib_overall)) if sib_overall else float("nan")
    print(f"  Top-100 MRE:   model = {m_top:.4f}   between-week = {s_top:.4f}   {'PASS' if m_top <= s_top else 'FAIL'}")
    print(f"  Overall MRE:   model = {m_ovr:.4f}   between-week = {s_ovr:.4f}   {'PASS' if m_ovr <= s_ovr else 'FAIL'}")
    print()
    print("  Interpretation:")
    print("    PASS = model prediction error is no worse than the natural week-to-week")
    print("           variability of the empirical signal at the same (weekday, hour).")
    print("    FAIL = model is more wrong than just using last week's data at the same slot.")

    # Per-frame side-by-side bar chart.
    out_dir = ROOT / "results" / "route_distribution_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Boxplot of top-100 MRE: model vs sibling.
    ax = axes[0]
    data = [model_top100, sib_top100]
    bp = ax.boxplot(data, labels=["V1 model\nvs truth", "between-week\nE_N vs E_N'"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#4c72b0")
    bp["boxes"][1].set_facecolor("#dd8452")
    ax.set_ylabel("Top-100 MRE")
    ax.set_title("Top-100 MRE: model error vs week-to-week variability")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(np.mean(model_top100), color="#4c72b0", linestyle="--", alpha=0.7,
               label=f"model mean = {np.mean(model_top100):.4f}")
    ax.axhline(np.mean(sib_top100), color="#dd8452", linestyle="--", alpha=0.7,
               label=f"between-week mean = {np.mean(sib_top100):.4f}")
    ax.legend(fontsize=8)

    ax = axes[1]
    data = [model_overall, sib_overall]
    bp = ax.boxplot(data, labels=["V1 model\nvs truth", "between-week\nE_N vs E_N'"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#4c72b0")
    bp["boxes"][1].set_facecolor("#dd8452")
    ax.set_ylabel("Overall MRE")
    ax.set_title("Overall MRE: model error vs week-to-week variability")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(np.mean(model_overall), color="#4c72b0", linestyle="--", alpha=0.7,
               label=f"model mean = {np.mean(model_overall):.4f}")
    ax.axhline(np.mean(sib_overall), color="#dd8452", linestyle="--", alpha=0.7,
               label=f"between-week mean = {np.mean(sib_overall):.4f}")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "v1_model_vs_betweenweek.png", dpi=140, bbox_inches="tight")
    plt.savefig(out_dir / "v1_model_vs_betweenweek.pdf", bbox_inches="tight")
    plt.close()

    # Save aggregate JSON.
    out = {
        "configuration": {
            "beta": BETA,
            "rho": RHO,
            "num_frames": len(model_metrics),
            "seed": args.seed,
            "tol": TOL,
            "max_iters": MAX_ITERS,
            "siblings_per_frame": args.siblings_per_frame,
        },
        "model_aggregate": {
            "top100_mre":   agg(model_top100),
            "overall_mre":  agg(model_overall),
            "kl":           agg(model_kl),
            "pearson_r":    agg(model_pearson),
            "rank_overlap_top100": agg(model_rankov),
        },
        "betweenweek_aggregate": {
            "top100_mre":   agg(sib_top100) if sib_top100 else None,
            "overall_mre":  agg(sib_overall) if sib_overall else None,
            "kl":           agg(sib_kl) if sib_kl else None,
            "pearson_r":    agg(sib_pearson) if sib_pearson else None,
            "num_pairs":    len(sibling_metrics),
        },
        "verdict": {
            "top100_pass": float(np.mean(model_top100)) <= float(np.mean(sib_top100)) if sib_top100 else None,
            "overall_pass": float(np.mean(model_overall)) <= float(np.mean(sib_overall)) if sib_overall else None,
        },
        "per_frame": model_metrics,
    }
    out_json = out_dir / "v1_evaluation.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_json}")
    print(f"Saved {out_dir / 'v1_model_vs_betweenweek.png'}")


if __name__ == "__main__":
    main()
