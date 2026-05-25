"""Event-day forecast experiment, diffused-prior version.

Differences from event_sept15_forecast.py:

- Diffusion smoothing applied to every bin's prior before any
  comparison: tilde_c = (1 - gamma) (C + alpha) + gamma * P (C + alpha),
  then normalized. gamma = 0.2, alpha = 0.01 (per user spec).
- Window: 19:00 to 23:55 on the target day (60 five-minute bins =
  1 hour before kickoff + 3-hour game + 1 hour after typical end).
- Three MREs reported as before: Top-100, Overall (full N), Stadium
  (links within 2 km of Rice-Eccles).
- Forecast: chain(diffused E_b^prior_day) compared to diffused
  E_b^target_day. Default chain params beta = 0.102, rho = 0.101.
- Control pair: Sept 1 -> Sept 8 (no event).
- Event pair  : Sept 8 -> Sept 15.
- Tuning grid : reduced to 4x4 (beta, rho).
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
from core import pagerank
from core.pagerank import build_phase_kernels, power_iteration

OUT = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project/results/event_sept15")
OUT.mkdir(parents=True, exist_ok=True)

BETA_DEF = 0.102
RHO_DEF  = 0.101
GAMMA    = 0.20
ALPHA    = 0.01
TOL = 1e-6
MAX_ITER = 200

# Window: 7 PM to 8 PM (one hour pre-kickoff, 12 five-minute bins).
WINDOW_START = "19:00:00"
WINDOW_END   = "19:55:00"

STADIUM_LAT = 40.7596
STADIUM_LON = -111.8485
# Radius for the corridor identification candidate set. Override via env var
# RADIUS_KM=3.0 to reproduce the wider cluster.
import os as _os
CLUSTER_RADIUS_KM = float(_os.environ.get("RADIUS_KM", 1.0))
CORRIDOR_TOP_K = 100        # keep the top-100 most-trafficked links in the ring

# Saturday evening bins used to RANK links by typical traffic. Uses two
# clean baseline Saturdays (Sept 1 and Sept 22), 17:00 to 23:55, so the
# ranking is independent of the event day (Sept 15) and of Sept 8 which
# is the forecast prior in the event pair.
CORRIDOR_RANK_DAYS  = ["2018-09-01", "2018-09-22"]
CORRIDOR_RANK_START = "17:00:00"
CORRIDOR_RANK_END   = "23:55:00"

GRID_BETA = [0.05, 0.102, 0.25, 0.5]
GRID_RHO  = [0.05, 0.101, 0.25, 0.5]
# alpha grid for Experiment 3 (4D joint tune).
# alpha_s, alpha_l = 0 reproduces uniform weights (Experiment 2 result).
GRID_ALPHA_S = [-0.5, 0.0, 0.5]
GRID_ALPHA_L = [-0.5, 0.0, 0.5]


def bins_for_day(day_str):
    start = datetime.strptime(f"{day_str} {WINDOW_START}", "%Y-%m-%d %H:%M:%S")
    end   = datetime.strptime(f"{day_str} {WINDOW_END}",   "%Y-%m-%d %H:%M:%S")
    out = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d %H:%M:%S"))
        cur += timedelta(minutes=5)
    return out


def build_diffusion_P(graph_data):
    """Row-stochastic uniform out-adjacency, no self-loops."""
    links = list(graph_data['links'].keys())
    N = len(links)
    id_to_idx = {lid: i for i, lid in enumerate(links)}
    adj = graph_data.get('adjacency', {})
    row, col, data = [], [], []
    for i, lid in enumerate(links):
        out_links = adj.get(lid, [])
        succ = [id_to_idx[ol] for ol in out_links if ol in id_to_idx]
        if not succ:
            continue
        w = 1.0 / len(succ)
        for j in succ:
            row.append(i); col.append(j); data.append(w)
    return csr_matrix((data, (row, col)), shape=(N, N), dtype=np.float64)


def build_geometric_ring_mask(links_order):
    """Boolean mask of links whose midpoint is within CLUSTER_RADIUS_KM
    of Rice-Eccles. Used as a candidate set; we then filter to the top-K
    most-trafficked."""
    xml_path = str(config.NETWORK_XML)
    nodes = {}
    link_endpoints = {}
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
    cos_lat = np.cos(np.deg2rad(STADIUM_LAT))
    R_km = 6371.0
    N = len(links_order)
    mask = np.zeros(N, dtype=bool)
    for i, lid in enumerate(links_order):
        ep = link_endpoints.get(str(lid))
        if ep is None or ep[0] not in nodes or ep[1] not in nodes:
            continue
        lon_a, lat_a = nodes[ep[0]]
        lon_b, lat_b = nodes[ep[1]]
        mid_lon = 0.5 * (lon_a + lon_b)
        mid_lat = 0.5 * (lat_a + lat_b)
        dlat_km = (mid_lat - STADIUM_LAT) * np.pi / 180.0 * R_km
        dlon_km = (mid_lon - STADIUM_LON) * np.pi / 180.0 * R_km * cos_lat
        if dlat_km ** 2 + dlon_km ** 2 <= CLUSTER_RADIUS_KM ** 2:
            mask[i] = True
    return mask


def build_corridor_cluster_mask(links_order, get_E_diffused):
    """Restrict the geometric ring to the top-K links by mean diffused
    popularity over baseline-Saturday evening bins. That picks out the
    actual access corridors (arterials) and drops residential streets."""
    ring = build_geometric_ring_mask(links_order)
    n_ring = int(ring.sum())
    print(f"[cluster] {n_ring:,} links within {CLUSTER_RADIUS_KM} km of Rice-Eccles "
          f"(geometric ring, candidate set)")
    # Accumulate mean E_b over baseline-Saturday evening bins
    N = len(links_order)
    accum = np.zeros(N, dtype=np.float64)
    nbins = 0
    for day in CORRIDOR_RANK_DAYS:
        cur = datetime.strptime(f"{day} {CORRIDOR_RANK_START}",
                                 "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(f"{day} {CORRIDOR_RANK_END}",
                                 "%Y-%m-%d %H:%M:%S")
        while cur <= end:
            E = get_E_diffused(cur.strftime("%Y-%m-%d %H:%M:%S"))
            if E is not None:
                accum += E
                nbins += 1
            cur += timedelta(minutes=5)
    print(f"  ranked over {nbins} baseline bins "
          f"({CORRIDOR_RANK_DAYS}, {CORRIDOR_RANK_START}-{CORRIDOR_RANK_END})")
    accum /= max(nbins, 1)
    # Zero out links outside the ring, then take top-K
    accum_in_ring = np.where(ring, accum, -1.0)
    top_idx = np.argsort(accum_in_ring)[::-1][:CORRIDOR_TOP_K]
    mask = np.zeros(N, dtype=bool)
    mask[top_idx] = True
    # Sanity: confirm all selected links are inside the ring
    assert ring[mask].all(), "corridor selection leaked outside the ring"
    print(f"[cluster] kept top {int(mask.sum())} links by mean baseline "
          f"popularity (these are the access corridors)")
    return mask


def load_corpus():
    print("[load] graph + popularity ...")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    print("[load] building phase kernels ...")
    t0 = time.time()
    P_up_cs, P_down_cs = build_phase_kernels(graph_data)
    print(f"  chain kernels: {time.time()-t0:.0f}s")

    print("[load] building diffusion P (uniform row-stochastic) ...")
    t0 = time.time()
    P_diff = build_diffusion_P(graph_data)
    print(f"  diffusion P  : {time.time()-t0:.0f}s")

    b = load_popularity_npz(str(config.DATA_DIR / "popularity_results_osm.npz"))
    matrix = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_link_ids = b["link_ids"]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_link_ids], dtype=np.int64)
    valid = proj >= 0
    time_index = {t: i for i, t in enumerate(times)}

    def get_E_diffused(time_str):
        """Return diffused-and-normalized E_b for the given bin time.
        Returns None if the bin is absent from the matrix."""
        t_idx = time_index.get(time_str)
        if t_idx is None:
            return None
        row = matrix.getrow(t_idx).toarray().ravel().astype(np.float64)
        C = np.zeros(N, dtype=np.float64)
        np.add.at(C, proj[valid], row[valid])
        # Laplace + one-step diffusion
        c = C + ALPHA
        c_diff = (1.0 - GAMMA) * c + GAMMA * (P_diff @ c)
        s = c_diff.sum()
        if s <= 0:
            return None
        return c_diff / s

    cluster_mask = build_corridor_cluster_mask(links, get_E_diffused)
    return P_up_cs, P_down_cs, get_E_diffused, N, cluster_mask


def forecast_pair(P_up_cs, P_down_cs, get_E, prior_day, target_day,
                  beta, rho, cluster_mask):
    """For each bin in the window:
       - Load diffused E_b for prior_day and target_day.
       - Run chain on prior_day's diffused E_b.
       - Compute four MREs per bin:
           * d2d_overall  = data-to-data, all 99,716 links
                            MRE(E_b^target, E_b^prior)
           * d2m_overall  = data-to-model, all 99,716 links
                            MRE(E_b^target, chain(E_b^prior))
           * d2d_cluster  = data-to-data on the 100 corridor links
           * d2m_cluster  = data-to-model on the 100 corridor links
    """
    prior_bins  = bins_for_day(prior_day)
    target_bins = bins_for_day(target_day)
    d2d_ov, d2m_ov, d2d_cl, d2m_cl, kept = [], [], [], [], []
    eps = 1e-6
    for bp, bt in zip(prior_bins, target_bins):
        E_in = get_E(bp)
        E_tg = get_E(bt)
        if E_in is None or E_tg is None:
            continue
        v_up, v_down, _ = power_iteration(P_up_cs, P_down_cs, E_in,
                                           beta, rho, TOL, MAX_ITER)
        v = v_up + v_down
        s = v.sum()
        if s > 0:
            v /= s
        denom = E_tg + eps
        err_d2d = np.abs(E_tg - E_in) / denom
        err_d2m = np.abs(E_tg - v) / denom
        d2d_ov.append(float(err_d2d.mean()))
        d2m_ov.append(float(err_d2m.mean()))
        if cluster_mask.any():
            d2d_cl.append(float(err_d2d[cluster_mask].mean()))
            d2m_cl.append(float(err_d2m[cluster_mask].mean()))
        else:
            d2d_cl.append(float("nan"))
            d2m_cl.append(float("nan"))
        kept.append(bt)
    return (np.array(d2d_ov), np.array(d2m_ov),
            np.array(d2d_cl), np.array(d2m_cl), kept)


def main():
    t_global = time.time()
    P_up_cs, P_down_cs, get_E, N, cluster_mask = load_corpus()
    print(f"  N = {N:,}, cluster size = {int(cluster_mask.sum()):,}")
    print(f"  diffusion params: gamma = {GAMMA}, alpha = {ALPHA}")
    print(f"  window         : {WINDOW_START} -- {WINDOW_END} on the target day")
    print()

    results = {"gamma": GAMMA, "alpha": ALPHA,
               "window_start": WINDOW_START, "window_end": WINDOW_END,
               "cluster_size": int(cluster_mask.sum())}

    # ----- EXPERIMENT 1 -----
    print("=" * 78)
    print("EXPERIMENT 1: forecast with default (beta, rho) on diffused priors")
    print(f"  beta = {BETA_DEF}, rho = {RHO_DEF}")
    print("=" * 78)
    print()

    def run_pair(prior_day, target_day, label):
        d2d_ov, d2m_ov, d2d_cl, d2m_cl, kept = forecast_pair(
            P_up_cs, P_down_cs, get_E,
            prior_day=prior_day, target_day=target_day,
            beta=BETA_DEF, rho=RHO_DEF, cluster_mask=cluster_mask,
        )
        print(f"{label}  ({prior_day} -> {target_day})")
        print(f"  N bins                            : {len(d2d_ov)}")
        print(f"  data->data  MRE all-links         : mean={d2d_ov.mean():.4f}  median={np.median(d2d_ov):.4f}")
        print(f"  data->model MRE all-links         : mean={d2m_ov.mean():.4f}  median={np.median(d2m_ov):.4f}")
        print(f"  data->data  MRE corridors (100)   : mean={d2d_cl.mean():.4f}  median={np.median(d2d_cl):.4f}")
        print(f"  data->model MRE corridors (100)   : mean={d2m_cl.mean():.4f}  median={np.median(d2m_cl):.4f}")
        print()
        return {
            "pair": f"{prior_day} -> {target_day}",
            "n_bins": len(d2d_ov),
            "bins": kept,
            "d2d_overall_per_bin": d2d_ov.tolist(),
            "d2m_overall_per_bin": d2m_ov.tolist(),
            "d2d_cluster_per_bin": d2d_cl.tolist(),
            "d2m_cluster_per_bin": d2m_cl.tolist(),
            "d2d_overall_mean":    float(d2d_ov.mean()),
            "d2m_overall_mean":    float(d2m_ov.mean()),
            "d2d_cluster_mean":    float(d2d_cl.mean()),
            "d2m_cluster_mean":    float(d2m_cl.mean()),
        }

    control = run_pair("2018-09-01", "2018-09-08",
                        "Control pair (no event)")
    event   = run_pair("2018-09-08", "2018-09-15",
                        "Event pair (Sept 15 game)")
    print("Ratios (event / control) on the same metric:")
    print(f"  data->data  all-links     : {event['d2d_overall_mean']/control['d2d_overall_mean']:.2f}x")
    print(f"  data->model all-links     : {event['d2m_overall_mean']/control['d2m_overall_mean']:.2f}x")
    print(f"  data->data  corridors     : {event['d2d_cluster_mean']/control['d2d_cluster_mean']:.2f}x")
    print(f"  data->model corridors     : {event['d2m_cluster_mean']/control['d2m_cluster_mean']:.2f}x")
    print()

    results["experiment_1_default"] = {
        "beta": BETA_DEF, "rho": RHO_DEF,
        "control": control,
        "event":   event,
    }
    # remember event arrays for experiment 2 tuning summary
    evt_d2m_ov = np.array(event["d2m_overall_per_bin"])
    evt_d2m_cl = np.array(event["d2m_cluster_per_bin"])

    # ----- EXPERIMENT 2 -----
    print("=" * 78)
    print("EXPERIMENT 2: tune (beta, rho) on Sept 8 -> Sept 15 event window")
    print("=" * 78)
    print()
    print(f"  grid: beta in {GRID_BETA}")
    print(f"        rho  in {GRID_RHO}")
    print()
    grid = []
    best = {"overall": (None, None, float("inf")),
            "cluster": (None, None, float("inf"))}
    t_grid = time.time()
    for beta in GRID_BETA:
        for rho in GRID_RHO:
            _, d2m_ov, _, d2m_cl, _ = forecast_pair(
                P_up_cs, P_down_cs, get_E,
                prior_day="2018-09-08", target_day="2018-09-15",
                beta=beta, rho=rho, cluster_mask=cluster_mask,
            )
            mo = float(d2m_ov.mean()); mc = float(d2m_cl.mean())
            grid.append({"beta": beta, "rho": rho,
                         "d2m_overall_mean": mo,
                         "d2m_cluster_mean": mc})
            tag = ""
            if mo < best["overall"][2]:
                best["overall"] = (beta, rho, mo); tag += " <- best all-links"
            if mc < best["cluster"][2]:
                best["cluster"] = (beta, rho, mc); tag += " <- best corridors"
            print(f"  beta={beta:.3f}  rho={rho:.3f}  "
                  f"all-links={mo:.4f}  corridors={mc:.4f}{tag}", flush=True)
    print()
    print(f"  Best all-links : beta={best['overall'][0]} rho={best['overall'][1]} MRE={best['overall'][2]:.4f}")
    print(f"  Best corridors : beta={best['cluster'][0]} rho={best['cluster'][1]} MRE={best['cluster'][2]:.4f}")
    print()
    print(f"  Default all-links MRE = {evt_d2m_ov.mean():.4f}  ->  Tuned = {best['overall'][2]:.4f}  "
          f"(reduction {(1 - best['overall'][2]/evt_d2m_ov.mean())*100:.1f}%)")
    print(f"  Default corridors MRE = {evt_d2m_cl.mean():.4f}  ->  Tuned = {best['cluster'][2]:.4f}  "
          f"(reduction {(1 - best['cluster'][2]/evt_d2m_cl.mean())*100:.1f}%)")
    print(f"\n  grid time = {time.time()-t_grid:.0f}s")

    results["experiment_2_tuned"] = {
        "grid": grid,
        "best_overall": {"beta": best["overall"][0], "rho": best["overall"][1],
                         "mre": best["overall"][2]},
        "best_cluster": {"beta": best["cluster"][0], "rho": best["cluster"][1],
                         "mre": best["cluster"][2]},
    }

    # ----- EXPERIMENT 3: full 4D tune ----------------------------------
    print()
    print("=" * 78)
    print("EXPERIMENT 3: tune (alpha_s, alpha_l, beta, rho) on event pair")
    print("=" * 78)
    print(f"  alpha_s in {GRID_ALPHA_S}")
    print(f"  alpha_l in {GRID_ALPHA_L}")
    print(f"  beta    in {GRID_BETA}")
    print(f"  rho     in {GRID_RHO}")
    n_combos = (len(GRID_ALPHA_S) * len(GRID_ALPHA_L) *
                len(GRID_BETA) * len(GRID_RHO))
    print(f"  total combos: {n_combos}\n")

    grid4d = []
    best4d = {"overall": (None, None, None, None, float("inf")),
              "cluster": (None, None, None, None, float("inf"))}

    # Need to read graph_data again to rebuild kernels per (a_s, a_l)
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)

    t_g = time.time()
    for a_s in GRID_ALPHA_S:
        for a_l in GRID_ALPHA_L:
            pagerank.PARAMS['alpha_s'] = a_s
            pagerank.PARAMS['alpha_l'] = a_l
            P_up_cs, P_down_cs = build_phase_kernels(graph_data)
            for beta in GRID_BETA:
                for rho in GRID_RHO:
                    _, d2m_ov, _, d2m_cl, _ = forecast_pair(
                        P_up_cs, P_down_cs, get_E,
                        prior_day="2018-09-08", target_day="2018-09-15",
                        beta=beta, rho=rho, cluster_mask=cluster_mask,
                    )
                    mo = float(d2m_ov.mean())
                    mc = float(d2m_cl.mean())
                    grid4d.append({"alpha_s": a_s, "alpha_l": a_l,
                                    "beta": beta, "rho": rho,
                                    "d2m_overall_mean": mo,
                                    "d2m_cluster_mean": mc})
                    tag = ""
                    if mo < best4d["overall"][4]:
                        best4d["overall"] = (a_s, a_l, beta, rho, mo); tag += " <- best all-links"
                    if mc < best4d["cluster"][4]:
                        best4d["cluster"] = (a_s, a_l, beta, rho, mc); tag += " <- best corridors"
                    print(f"  a_s={a_s:+.2f} a_l={a_l:+.2f}  "
                          f"b={beta:.3f} r={rho:.3f}  "
                          f"all={mo:.4f} corr={mc:.4f}{tag}", flush=True)
    print()
    bo = best4d["overall"]
    bc = best4d["cluster"]
    print(f"  Best all-links : alpha_s={bo[0]:+.2f} alpha_l={bo[1]:+.2f} "
          f"beta={bo[2]} rho={bo[3]}  MRE={bo[4]:.4f}")
    print(f"  Best corridors : alpha_s={bc[0]:+.2f} alpha_l={bc[1]:+.2f} "
          f"beta={bc[2]} rho={bc[3]}  MRE={bc[4]:.4f}")
    print()
    print(f"  Default (all zeros, beta=0.102 rho=0.101):")
    print(f"    all-links MRE = {evt_d2m_ov.mean():.4f}")
    print(f"    corridors MRE = {evt_d2m_cl.mean():.4f}")
    print(f"  Tuned 4D best:")
    print(f"    all-links MRE = {bo[4]:.4f}  "
          f"(reduction {(1 - bo[4]/evt_d2m_ov.mean())*100:.1f}%)")
    print(f"    corridors MRE = {bc[4]:.4f}  "
          f"(reduction {(1 - bc[4]/evt_d2m_cl.mean())*100:.1f}%)")
    print(f"\n  grid time = {time.time()-t_g:.0f}s")

    results["experiment_3_tuned_4d"] = {
        "grid_alpha_s": GRID_ALPHA_S,
        "grid_alpha_l": GRID_ALPHA_L,
        "grid_beta": GRID_BETA,
        "grid_rho": GRID_RHO,
        "grid": grid4d,
        "best_overall": {"alpha_s": bo[0], "alpha_l": bo[1],
                          "beta": bo[2], "rho": bo[3], "mre": bo[4]},
        "best_cluster": {"alpha_s": bc[0], "alpha_l": bc[1],
                          "beta": bc[2], "rho": bc[3], "mre": bc[4]},
    }

    out_name = f"results_diffused_{int(CLUSTER_RADIUS_KM)}km.json"
    with open(OUT / out_name, "w") as f:
        json.dump(results, f, indent=2)
    # also keep a canonical "results_diffused.json" pointing at the latest run
    with open(OUT / "results_diffused.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {time.time()-t_global:.0f}s. saved {OUT/out_name}")


if __name__ == "__main__":
    main()
