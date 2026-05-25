"""
Cross-week forecast experiment for the V1 / single-matrix two-phase chain.

For each (target_slot t, sibling_slot t') with same weekday + time-of-day but
different week, do three comparisons against the ground truth E_N(t):

  A. naive: just use E_N(t')              -> baseline 1 (raw last-week signal)
  B. model_t : run model with E_N(t)      -> baseline 2 (same-slot self-consistency)
  C. model_t': run model with E_N(t')     -> the actual forecast (predict t from t')

Also compute the "averaged sibling" baseline:

  D. naive_avg: mean_{t'}(E_N(t'))         -> a smoother last-week reference
  E. model_avg: model started from naive_avg

Reports MRE Top-100 and Overall for each. The professor's gate is now:

       MRE(model_t', E_N(t))   <=   MRE(E_N(t'), E_N(t)).

Run:  python3 analysis/evaluate_v1_forecast.py --num-frames 49 --seed 42
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from core.pagerank import compute_speed_lane_weights, build_phase_matrix
from core.io import load_popularity_npz

BETA = 0.124
RHO = 0.147
TOL = 1e-7
MAX_ITERS = 300


def build_phase_kernels(graph_data):
    up_w = compute_speed_lane_weights(graph_data, is_up_phase=True)
    dn_w = compute_speed_lane_weights(graph_data, is_up_phase=False)
    P_up = build_phase_matrix(graph_data, up_w).tocsr()
    P_dn = build_phase_matrix(graph_data, dn_w).tocsr()
    return P_up.T.tocsr(), P_dn.T.tocsr()


def v1_iter(P_up_T, P_dn_T, E_b, beta, rho, tol, max_iter):
    N = E_b.shape[0]
    v_up = E_b.copy()
    v_dn = np.zeros(N)
    for _ in range(max_iter):
        s_dn = float(v_dn.sum())
        v_up_new = (1.0 - beta) * (P_up_T @ v_up) + rho * E_b * s_dn
        v_dn_new = beta * v_up + (1.0 - rho) * (P_dn_T @ v_dn)
        s = float(v_up_new.sum() + v_dn_new.sum())
        if s > 0:
            v_up_new /= s
            v_dn_new /= s
        diff = float(np.abs(v_up_new - v_up).sum() + np.abs(v_dn_new - v_dn).sum())
        v_up, v_dn = v_up_new, v_dn_new
        if diff < tol:
            break
    return v_up, v_dn


def extract_E_N(matrix, t_idx, pop_link_ids, lid_to_idx, N):
    proj = np.fromiter(
        (lid_to_idx.get(lid, -1) for lid in pop_link_ids),
        dtype=np.int64,
        count=len(pop_link_ids),
    )
    row = matrix.getrow(t_idx).toarray().ravel()
    E = np.zeros(N)
    valid = proj >= 0
    np.add.at(E, proj[valid], row[valid])
    s = E.sum()
    if s > 0:
        E /= s
    else:
        E = np.ones(N) / N
    return E


def mre(truth, pred, top_k=100):
    rel = np.abs(truth - pred) / (truth + 1e-9)
    overall = float(np.mean(rel))
    top = np.argsort(truth)[::-1][:top_k]
    top_mre = float(np.mean(rel[top]))
    return overall, top_mre


def parse_dt(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def find_siblings(times_set, target_dt, max_per=8):
    out = []
    for t in times_set:
        if t == target_dt.strftime("%Y-%m-%d %H:%M:%S"):
            continue
        dt = parse_dt(t)
        if dt.weekday() == target_dt.weekday() and dt.hour == target_dt.hour and dt.minute == target_dt.minute:
            out.append(t)
            if len(out) >= max_per:
                break
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--siblings-per-frame", type=int, default=6)
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

    print("Building phase kernels ...")
    P_up_T, P_dn_T = build_phase_kernels(graph_data)

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(times), size=min(args.num_frames, len(times)), replace=False)
    sample_times = [times[i] for i in sample_idx]

    print(f"\nV1 single-matrix two-phase chain  (beta={BETA}, rho={RHO})")
    print(f"Forecast experiment: predict slot t from sibling slot t' (different week, same weekday+time-of-day)")
    print()

    # Aggregators.
    naive_top, naive_ovr = [], []        # raw E_N(t') vs E_N(t)
    model_t_top, model_t_ovr = [], []    # model started from E_N(t) vs E_N(t)
    forecast_top, forecast_ovr = [], []  # model started from E_N(t') vs E_N(t)
    naive_avg_top, naive_avg_ovr = [], []     # mean of siblings vs E_N(t)
    model_avg_top, model_avg_ovr = [], []     # model started from sibling-mean vs E_N(t)

    per_frame = []
    t_start = time.time()

    for fi, t_str in enumerate(sample_times):
        t_idx = times.index(t_str)
        E_t = extract_E_N(matrix, t_idx, pop_link_ids, lid_to_idx, N)

        # Same-slot model run (B).
        v_up, v_dn = v1_iter(P_up_T, P_dn_T, E_t, BETA, RHO, TOL, MAX_ITERS)
        v_self = v_up + v_dn
        v_self /= v_self.sum()
        ovr_b, top_b = mre(E_t, v_self, top_k=args.top_k)
        model_t_top.append(top_b)
        model_t_ovr.append(ovr_b)

        # Siblings.
        target_dt = parse_dt(t_str)
        sibs = find_siblings(times, target_dt, max_per=args.siblings_per_frame)
        if not sibs:
            continue

        E_sib_list = []
        f_top_list, f_ovr_list = [], []
        n_top_list, n_ovr_list = [], []
        for sib in sibs:
            s_idx = times.index(sib)
            E_sib = extract_E_N(matrix, s_idx, pop_link_ids, lid_to_idx, N)
            E_sib_list.append(E_sib)

            # Naive forecast (A): just use the sibling.
            ovr_a, top_a = mre(E_t, E_sib, top_k=args.top_k)
            n_top_list.append(top_a); n_ovr_list.append(ovr_a)
            naive_top.append(top_a); naive_ovr.append(ovr_a)

            # Model forecast (C): run model with sibling as input, compare to E_t.
            v_up_s, v_dn_s = v1_iter(P_up_T, P_dn_T, E_sib, BETA, RHO, TOL, MAX_ITERS)
            v_fc = v_up_s + v_dn_s
            v_fc /= v_fc.sum()
            ovr_c, top_c = mre(E_t, v_fc, top_k=args.top_k)
            f_top_list.append(top_c); f_ovr_list.append(ovr_c)
            forecast_top.append(top_c); forecast_ovr.append(ovr_c)

        # Averaged-sibling baselines (D, E).
        E_sib_mean = np.mean(np.stack(E_sib_list), axis=0)
        E_sib_mean /= E_sib_mean.sum()
        ovr_d, top_d = mre(E_t, E_sib_mean, top_k=args.top_k)
        naive_avg_top.append(top_d); naive_avg_ovr.append(ovr_d)
        v_up_a, v_dn_a = v1_iter(P_up_T, P_dn_T, E_sib_mean, BETA, RHO, TOL, MAX_ITERS)
        v_avg = v_up_a + v_dn_a
        v_avg /= v_avg.sum()
        ovr_e, top_e = mre(E_t, v_avg, top_k=args.top_k)
        model_avg_top.append(top_e); model_avg_ovr.append(ovr_e)

        per_frame.append({
            "timeframe": t_str,
            "n_siblings": len(sibs),
            "model_t_top100": top_b,
            "model_t_overall": ovr_b,
            "naive_top100_mean": float(np.mean(n_top_list)),
            "naive_overall_mean": float(np.mean(n_ovr_list)),
            "forecast_top100_mean": float(np.mean(f_top_list)),
            "forecast_overall_mean": float(np.mean(f_ovr_list)),
            "naive_avg_top100": top_d,
            "naive_avg_overall": ovr_d,
            "model_avg_top100": top_e,
            "model_avg_overall": ovr_e,
        })

        print(f"  {fi+1:>3}/{len(sample_times)}  {t_str}  sib={len(sibs):>2}  "
              f"naive_top100={np.mean(n_top_list):.4f}  forecast_top100={np.mean(f_top_list):.4f}  "
              f"sib_avg={top_d:.4f}  model_sib_avg={top_e:.4f}")

    print()
    print(f"Total wall-clock: {time.time() - t_start:.1f} s")

    def agg(arr):
        if not arr:
            return None
        return dict(mean=float(np.mean(arr)),
                    std=float(np.std(arr)),
                    median=float(np.median(arr)),
                    minv=float(np.min(arr)),
                    maxv=float(np.max(arr)))

    print()
    print("=" * 86)
    print("  AGGREGATE: Top-100 MRE vs E_N(t)")
    print("=" * 86)
    rows = [
        ("naive: E_N(t') alone (raw last-week)",                 naive_top),
        ("forecast: model started from E_N(t')",                 forecast_top),
        ("naive_avg: mean_{t'} E_N(t')  (averaged last weeks)",  naive_avg_top),
        ("model_avg: model started from mean_{t'} E_N(t')",      model_avg_top),
        ("model_t: model started from E_N(t)  (same-slot)",      model_t_top),
    ]
    for name, arr in rows:
        a = agg(arr)
        if a:
            print(f"  {name:<58}  mean={a['mean']:8.4f}  std={a['std']:8.4f}  median={a['median']:8.4f}")

    print()
    print("=" * 86)
    print("  AGGREGATE: Overall MRE vs E_N(t)")
    print("=" * 86)
    rows = [
        ("naive: E_N(t') alone",                                  naive_ovr),
        ("forecast: model started from E_N(t')",                  forecast_ovr),
        ("naive_avg: mean_{t'} E_N(t')",                          naive_avg_ovr),
        ("model_avg: model started from mean_{t'} E_N(t')",       model_avg_ovr),
        ("model_t: model started from E_N(t)  (same-slot)",       model_t_ovr),
    ]
    for name, arr in rows:
        a = agg(arr)
        if a:
            print(f"  {name:<58}  mean={a['mean']:8.4f}  std={a['std']:8.4f}  median={a['median']:8.4f}")

    # Verdict.
    n_top_m = float(np.mean(naive_top))
    f_top_m = float(np.mean(forecast_top))
    na_top_m = float(np.mean(naive_avg_top))
    ma_top_m = float(np.mean(model_avg_top))
    n_ovr_m = float(np.mean(naive_ovr))
    f_ovr_m = float(np.mean(forecast_ovr))
    na_ovr_m = float(np.mean(naive_avg_ovr))
    ma_ovr_m = float(np.mean(model_avg_ovr))

    print()
    print("=" * 86)
    print("  VERDICTS")
    print("=" * 86)
    print(f"  Forecast vs naive (per-sibling pair):")
    print(f"    Top-100:  forecast={f_top_m:.4f}  naive={n_top_m:.4f}   "
          f"{'PASS' if f_top_m <= n_top_m else 'FAIL'}  (improvement = {(n_top_m - f_top_m)/n_top_m*100:+.1f}%)")
    print(f"    Overall:  forecast={f_ovr_m:.4f}  naive={n_ovr_m:.4f}   "
          f"{'PASS' if f_ovr_m <= n_ovr_m else 'FAIL'}  (improvement = {(n_ovr_m - f_ovr_m)/n_ovr_m*100:+.1f}%)")
    print()
    print(f"  Sibling-averaged baseline (D vs E):")
    print(f"    Top-100:  model_avg={ma_top_m:.4f}  naive_avg={na_top_m:.4f}   "
          f"{'PASS' if ma_top_m <= na_top_m else 'FAIL'}  (improvement = {(na_top_m - ma_top_m)/na_top_m*100:+.1f}%)")
    print(f"    Overall:  model_avg={ma_ovr_m:.4f}  naive_avg={na_ovr_m:.4f}   "
          f"{'PASS' if ma_ovr_m <= na_ovr_m else 'FAIL'}  (improvement = {(na_ovr_m - ma_ovr_m)/na_ovr_m*100:+.1f}%)")

    # Plot.
    out_dir = ROOT / "results" / "route_distribution_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = ["naive\nE_N(t')", "forecast\nmodel(E_N(t'))", "naive_avg\nmean E_N(t')", "model_avg\nmodel(mean)", "same-slot\nmodel(E_N(t))"]
    colors = ["#dd8452", "#4c72b0", "#dd8452", "#4c72b0", "#55a868"]

    ax = axes[0]
    data = [naive_top, forecast_top, naive_avg_top, model_avg_top, model_t_top]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
    ax.set_ylabel("Top-100 MRE")
    ax.set_title("Top-100 MRE vs ground truth E_N(t)")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    data = [naive_ovr, forecast_ovr, naive_avg_ovr, model_avg_ovr, model_t_ovr]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
    ax.set_ylabel("Overall MRE")
    ax.set_title("Overall MRE vs ground truth E_N(t)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "v1_forecast_baselines.png", dpi=140, bbox_inches="tight")
    plt.savefig(out_dir / "v1_forecast_baselines.pdf", bbox_inches="tight")
    plt.close()

    # Save aggregates.
    out = {
        "configuration": dict(beta=BETA, rho=RHO, num_frames=len(sample_times),
                              seed=args.seed, siblings_per_frame=args.siblings_per_frame,
                              tol=TOL, max_iters=MAX_ITERS),
        "aggregate_top100": {
            "naive":      agg(naive_top),
            "forecast":   agg(forecast_top),
            "naive_avg":  agg(naive_avg_top),
            "model_avg":  agg(model_avg_top),
            "model_t":    agg(model_t_top),
        },
        "aggregate_overall": {
            "naive":      agg(naive_ovr),
            "forecast":   agg(forecast_ovr),
            "naive_avg":  agg(naive_avg_ovr),
            "model_avg":  agg(model_avg_ovr),
            "model_t":    agg(model_t_ovr),
        },
        "verdicts": {
            "forecast_vs_naive_top100":      f_top_m <= n_top_m,
            "forecast_vs_naive_overall":     f_ovr_m <= n_ovr_m,
            "model_avg_vs_naive_avg_top100": ma_top_m <= na_top_m,
            "model_avg_vs_naive_avg_overall": ma_ovr_m <= na_ovr_m,
        },
        "per_frame": per_frame,
    }
    out_path = out_dir / "v1_forecast.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")
    print(f"Saved {out_dir / 'v1_forecast_baselines.png'}")


if __name__ == "__main__":
    main()
