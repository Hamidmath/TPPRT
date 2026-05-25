"""Forecast experiment for the Utah vs Washington football game,
Sept 15, 2018, 8:00 PM MDT kickoff at Rice-Eccles Stadium.

Two experiments:

(1) Forecast with default parameters.
    Input prior  : empirical E_b from the previous Saturday's same bin
                   (Sept 8 at hh:mm).
    Output       : v_b = TwoPhase(E_b^Sept8; beta=0.102, rho=0.101).
    Target       : empirical E_b on Sept 15 at the same bin.
    Error        : Top-100 MRE and Overall MRE.
    Control pair : Sept 1 -> Sept 8 (no event on either side).
    Expected     : event-day MRE > control MRE.

(2) Grid-tune (beta, rho) on the Sept 8 -> Sept 15 pair and report
    the improvement over default.
"""
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz
from core.pagerank import build_phase_kernels, power_iteration

OUT = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project/results/event_sept15")
OUT.mkdir(parents=True, exist_ok=True)

BETA_DEF = 0.102
RHO_DEF = 0.101
TOL = 1e-6
MAX_ITER = 200

# Event window: Sept 15, 17:00 to 23:55 (84 five-minute bins).
# Kickoff at 20:00 MDT. Window spans pre-event (17:00-20:00), game
# (20:00-23:00), and start of departure (23:00-23:55).
WINDOW_START = "17:00:00"
WINDOW_END   = "23:55:00"

# Rice-Eccles Stadium location (approx) and cluster radius
STADIUM_LAT = 40.7596
STADIUM_LON = -111.8485
CLUSTER_RADIUS_KM = 2.0


def build_stadium_cluster_mask(graph_data, links_order):
    """Parse slc_network.xml to get node lat/lon. For each MATSim link in
    `links_order`, compute its midpoint and return a boolean mask of length
    N marking links whose midpoint is within CLUSTER_RADIUS_KM of the
    stadium."""
    xml_path = str(config.NETWORK_XML)
    print(f"[cluster] parsing {xml_path} ...")
    nodes = {}  # node_id (str) -> (lon, lat)
    link_endpoints = {}  # link_id (str) -> (from_id, to_id)
    node_re = re.compile(r'<node\s+id="(\d+)"\s+x="(-?[\d.]+)"\s+y="(-?[\d.]+)"')
    link_re = re.compile(r'<link\s+id="(\d+)"\s+from="(\d+)"\s+to="(\d+)"')
    with open(xml_path) as f:
        for line in f:
            m = node_re.search(line)
            if m:
                nodes[m.group(1)] = (float(m.group(2)), float(m.group(3)))
                continue
            m = link_re.search(line)
            if m:
                link_endpoints[m.group(1)] = (m.group(2), m.group(3))
    print(f"  parsed {len(nodes):,} nodes and {len(link_endpoints):,} links")

    # Equirectangular approximation, valid for ~2 km near 40.76 N
    lat_ref = np.deg2rad(STADIUM_LAT)
    cos_lat = np.cos(lat_ref)
    R_km = 6371.0
    N = len(links_order)
    in_cluster = np.zeros(N, dtype=bool)
    n_found = 0
    n_missing = 0
    for i, lid in enumerate(links_order):
        ep = link_endpoints.get(str(lid))
        if ep is None or ep[0] not in nodes or ep[1] not in nodes:
            n_missing += 1
            continue
        lon_a, lat_a = nodes[ep[0]]
        lon_b, lat_b = nodes[ep[1]]
        mid_lon = 0.5 * (lon_a + lon_b)
        mid_lat = 0.5 * (lat_a + lat_b)
        dlat_km = (mid_lat - STADIUM_LAT) * np.pi / 180.0 * R_km
        dlon_km = (mid_lon - STADIUM_LON) * np.pi / 180.0 * R_km * cos_lat
        if dlat_km * dlat_km + dlon_km * dlon_km <= CLUSTER_RADIUS_KM ** 2:
            in_cluster[i] = True
            n_found += 1
    print(f"  links in cluster (<= {CLUSTER_RADIUS_KM} km from "
          f"Rice-Eccles): {n_found:,}")
    print(f"  links missing node lookups: {n_missing}")
    return in_cluster

# Grid for the tuning experiment
GRID_BETA = [0.05, 0.075, 0.102, 0.15, 0.25, 0.40, 0.60, 0.80]
GRID_RHO  = [0.05, 0.075, 0.101, 0.15, 0.25, 0.40, 0.60, 0.80]


def bins_for_day(day_str, start_hms=WINDOW_START, end_hms=WINDOW_END):
    """Return list of bin time strings for the event window on a given day."""
    start = datetime.strptime(f"{day_str} {start_hms}", "%Y-%m-%d %H:%M:%S")
    end   = datetime.strptime(f"{day_str} {end_hms}",   "%Y-%m-%d %H:%M:%S")
    out = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d %H:%M:%S"))
        cur += timedelta(minutes=5)
    return out


def load_corpus():
    print("[load] graph + popularity ...")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    print("[load] building phase kernels (one-time) ...")
    t0 = time.time()
    P_up_cs, P_down_cs = build_phase_kernels(graph_data)
    print(f"  kernels built in {time.time()-t0:.0f}s")

    cluster_mask = build_stadium_cluster_mask(graph_data, links)

    b = load_popularity_npz(str(config.DATA_DIR / "popularity_results_osm.npz"))
    matrix = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_link_ids = b["link_ids"]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_link_ids], dtype=np.int64)
    valid = proj >= 0
    time_index = {t: i for i, t in enumerate(times)}

    def get_E(time_str):
        t_idx = time_index.get(time_str)
        if t_idx is None:
            return None
        row = matrix.getrow(t_idx).toarray().ravel()
        E_N = np.zeros(N)
        np.add.at(E_N, proj[valid], row[valid])
        s = E_N.sum()
        if s <= 0:
            return None
        return E_N / s
    return P_up_cs, P_down_cs, get_E, N, cluster_mask


def forecast_pair(P_up_cs, P_down_cs, get_E, prior_day, target_day,
                  beta, rho, cluster_mask=None, tol=TOL, max_iter=MAX_ITER):
    """For each event-window bin, forecast target from prior_day via the
    chain run with (beta, rho). Returns per-bin Top-100 MRE, Overall MRE,
    and Cluster MRE (restricted to cluster_mask if provided)."""
    prior_bins  = bins_for_day(prior_day)
    target_bins = bins_for_day(target_day)
    top100, overall, cluster_mre, kept = [], [], [], []
    for bp, bt in zip(prior_bins, target_bins):
        E_in = get_E(bp)
        E_tg = get_E(bt)
        if E_in is None or E_tg is None:
            continue
        v_up, v_down, _ = power_iteration(P_up_cs, P_down_cs, E_in,
                                           beta, rho, tol, max_iter)
        v = v_up + v_down
        s = v.sum()
        if s > 0:
            v /= s
        abs_err = np.abs(E_tg - v)
        eps = 1e-6
        top_idx = np.argsort(E_tg)[::-1][:100]
        t100 = float(np.mean(abs_err[top_idx] / (E_tg[top_idx] + eps)))
        ovr  = float(np.mean(abs_err / (E_tg + eps)))
        top100.append(t100); overall.append(ovr); kept.append(bt)
        if cluster_mask is not None and cluster_mask.any():
            cm = float(np.mean(abs_err[cluster_mask] /
                                (E_tg[cluster_mask] + eps)))
        else:
            cm = float("nan")
        cluster_mre.append(cm)
    return (np.array(top100), np.array(overall),
            np.array(cluster_mre), kept)


def main():
    t_global = time.time()
    P_up_cs, P_down_cs, get_E, N, cluster_mask = load_corpus()
    print(f"  N = {N:,}, cluster size = {int(cluster_mask.sum()):,}")

    results = {}

    # -- Experiment 1 ---------------------------------------------------
    print()
    print("=" * 78)
    print("EXPERIMENT 1: forecast with default parameters")
    print(f"  beta = {BETA_DEF}, rho = {RHO_DEF}")
    print("=" * 78)
    print()
    print("Control pair  : Sept 1 -> Sept 8   (no event either side)")
    t100_ctrl, ovr_ctrl, cl_ctrl, _ = forecast_pair(
        P_up_cs, P_down_cs, get_E,
        prior_day="2018-09-01", target_day="2018-09-08",
        beta=BETA_DEF, rho=RHO_DEF, cluster_mask=cluster_mask,
    )
    print(f"  N bins kept       : {len(t100_ctrl)}")
    print(f"  Top-100 MRE       : mean={t100_ctrl.mean():.4f}  median={np.median(t100_ctrl):.4f}")
    print(f"  Overall MRE       : mean={ovr_ctrl.mean():.4f}  median={np.median(ovr_ctrl):.4f}")
    print(f"  Stadium MRE       : mean={cl_ctrl.mean():.4f}  median={np.median(cl_ctrl):.4f}")
    print()
    print("Event pair    : Sept 8 -> Sept 15  (Sept 15 is the football game)")
    t100_evt, ovr_evt, cl_evt, kept_event = forecast_pair(
        P_up_cs, P_down_cs, get_E,
        prior_day="2018-09-08", target_day="2018-09-15",
        beta=BETA_DEF, rho=RHO_DEF, cluster_mask=cluster_mask,
    )
    print(f"  N bins kept       : {len(t100_evt)}")
    print(f"  Top-100 MRE       : mean={t100_evt.mean():.4f}  median={np.median(t100_evt):.4f}")
    print(f"  Overall MRE       : mean={ovr_evt.mean():.4f}  median={np.median(ovr_evt):.4f}")
    print(f"  Stadium MRE       : mean={cl_evt.mean():.4f}  median={np.median(cl_evt):.4f}")
    print()
    if t100_ctrl.mean() > 0:
        print(f"  Top-100 ratio (event / control): {t100_evt.mean()/t100_ctrl.mean():.2f}x")
    if ovr_ctrl.mean() > 0:
        print(f"  Overall ratio (event / control): {ovr_evt.mean()/ovr_ctrl.mean():.2f}x")
    if cl_ctrl.mean() > 0:
        print(f"  Stadium ratio (event / control): {cl_evt.mean()/cl_ctrl.mean():.2f}x")

    results["cluster_size"] = int(cluster_mask.sum())
    results["stadium"] = {"lat": STADIUM_LAT, "lon": STADIUM_LON,
                          "radius_km": CLUSTER_RADIUS_KM}
    results["experiment_1_default_params"] = {
        "beta": BETA_DEF, "rho": RHO_DEF,
        "control_pair": "Sept 1 -> Sept 8",
        "control_top100_per_bin": t100_ctrl.tolist(),
        "control_overall_per_bin": ovr_ctrl.tolist(),
        "control_cluster_per_bin": cl_ctrl.tolist(),
        "control_bins": bins_for_day("2018-09-08"),
        "control_top100_mean": float(t100_ctrl.mean()),
        "control_overall_mean": float(ovr_ctrl.mean()),
        "control_cluster_mean": float(cl_ctrl.mean()),
        "event_pair": "Sept 8 -> Sept 15",
        "event_top100_per_bin": t100_evt.tolist(),
        "event_overall_per_bin": ovr_evt.tolist(),
        "event_cluster_per_bin": cl_evt.tolist(),
        "event_bins": kept_event,
        "event_top100_mean": float(t100_evt.mean()),
        "event_overall_mean": float(ovr_evt.mean()),
        "event_cluster_mean": float(cl_evt.mean()),
    }

    # -- Experiment 2 ---------------------------------------------------
    print()
    print("=" * 78)
    print("EXPERIMENT 2: tune (beta, rho) on the Sept 8 -> Sept 15 pair")
    print("=" * 78)
    print()
    print(f"  grid: beta in {GRID_BETA}")
    print(f"        rho  in {GRID_RHO}")
    print(f"  total: {len(GRID_BETA)*len(GRID_RHO)} combinations on "
          f"{len(t100_evt)} event-day bins")
    print()
    grid = []
    best = {"top100":  (None, None, float("inf")),
            "overall": (None, None, float("inf")),
            "cluster": (None, None, float("inf"))}
    t_grid = time.time()
    for beta in GRID_BETA:
        for rho in GRID_RHO:
            t_top, t_ovr, t_cl, _ = forecast_pair(
                P_up_cs, P_down_cs, get_E,
                prior_day="2018-09-08", target_day="2018-09-15",
                beta=beta, rho=rho, cluster_mask=cluster_mask,
            )
            mt = float(t_top.mean()); mo = float(t_ovr.mean()); mc = float(t_cl.mean())
            grid.append({"beta": beta, "rho": rho,
                         "top100_mean": mt, "overall_mean": mo,
                         "cluster_mean": mc})
            tag = ""
            if mt < best["top100"][2]:
                best["top100"] = (beta, rho, mt); tag += " <- best Top-100"
            if mo < best["overall"][2]:
                best["overall"] = (beta, rho, mo); tag += " <- best Overall"
            if mc < best["cluster"][2]:
                best["cluster"] = (beta, rho, mc); tag += " <- best Stadium"
            print(f"  beta={beta:.3f}  rho={rho:.3f}  "
                  f"Top-100={mt:.4f}  Overall={mo:.4f}  "
                  f"Stadium={mc:.4f}{tag}", flush=True)
    print()
    print(f"  Best by Top-100 : beta={best['top100'][0]:.3f}  rho={best['top100'][1]:.3f}  MRE={best['top100'][2]:.4f}")
    print(f"  Best by Overall : beta={best['overall'][0]:.3f}  rho={best['overall'][1]:.3f}  MRE={best['overall'][2]:.4f}")
    print(f"  Best by Stadium : beta={best['cluster'][0]:.3f}  rho={best['cluster'][1]:.3f}  MRE={best['cluster'][2]:.4f}")
    print()
    print(f"  Default Top-100 MRE on event: {t100_evt.mean():.4f}  ->  Tuned: {best['top100'][2]:.4f}")
    print(f"  Default Overall MRE on event: {ovr_evt.mean():.4f}  ->  Tuned: {best['overall'][2]:.4f}")
    print(f"  Default Stadium MRE on event: {cl_evt.mean():.4f}  ->  Tuned: {best['cluster'][2]:.4f}")
    print(f"\n  grid time = {time.time()-t_grid:.0f}s")

    results["experiment_2_tuned"] = {
        "grid": grid,
        "best_top100": {"beta": best["top100"][0], "rho": best["top100"][1],
                        "mre": best["top100"][2]},
        "best_overall": {"beta": best["overall"][0], "rho": best["overall"][1],
                         "mre": best["overall"][2]},
        "best_cluster": {"beta": best["cluster"][0], "rho": best["cluster"][1],
                         "mre": best["cluster"][2]},
    }

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] saved {OUT / 'results.json'} in {time.time()-t_global:.0f}s")


if __name__ == "__main__":
    main()
