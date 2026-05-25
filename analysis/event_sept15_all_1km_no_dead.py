"""Focused rerun: only the `all_1km` cluster, with proper dead-end filter.

A "dead-end link" is one whose source or destination node has only one
unique undirected neighbour in the road graph (pendant node). This
catches cul-de-sacs, driveway stubs and graph-boundary artifacts.

The other clusters (top25/50/100, neighbors_3hop) are unaffected
because their construction either drops dead-ends naturally
(top-K-by-traffic) or operates on a different topology rule
(neighbours). Their numbers in results_cluster_compare.json stand.
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

BETA_DEF, RHO_DEF = 0.102, 0.101
GAMMA, ALPHA = 0.20, 0.01
TOL, MAX_ITER = 1e-6, 200

STADIUM_LAT, STADIUM_LON = 40.7596, -111.8485
CLUSTER_RADIUS_KM = 1.0
WINDOW_START, WINDOW_END = "19:00:00", "19:55:00"
CORRIDOR_RANK_DAYS  = ["2018-09-01", "2018-09-22"]
CORRIDOR_RANK_START = "17:00:00"
CORRIDOR_RANK_END   = "23:55:00"

GRID_BETA = [0.05, 0.102, 0.25, 0.5]
GRID_RHO  = [0.05, 0.101, 0.25, 0.5]
GRID_ALPHA_S = [-0.5, 0.0, 0.5]
GRID_ALPHA_L = [-0.5, 0.0, 0.5]


def bins_for_day(day_str):
    start = datetime.strptime(f"{day_str} {WINDOW_START}", "%Y-%m-%d %H:%M:%S")
    end   = datetime.strptime(f"{day_str} {WINDOW_END}",   "%Y-%m-%d %H:%M:%S")
    out, cur = [], start
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d %H:%M:%S"))
        cur += timedelta(minutes=5)
    return out


def parse_matsim_xml(xml_path):
    link_endpoints = {}
    nodes = {}
    n_re = re.compile(r'<node\s+id="(\d+)"\s+x="(-?[\d.]+)"\s+y="(-?[\d.]+)"')
    l_re = re.compile(r'<link\s+id="(\d+)"\s+from="(\d+)"\s+to="(\d+)"')
    with open(xml_path) as f:
        for line in f:
            m = n_re.search(line)
            if m:
                nodes[m.group(1)] = (float(m.group(2)), float(m.group(3))); continue
            m = l_re.search(line)
            if m:
                link_endpoints[m.group(1)] = (m.group(2), m.group(3))
    return nodes, link_endpoints


def build_diffusion_P(graph_data, links_order):
    id_to_idx = {lid: i for i, lid in enumerate(links_order)}
    adj = graph_data.get('adjacency', {})
    N = len(links_order)
    row, col, data = [], [], []
    for i, lid in enumerate(links_order):
        succ = [id_to_idx[ol] for ol in adj.get(lid, []) if ol in id_to_idx]
        if not succ: continue
        w = 1.0 / len(succ)
        for j in succ:
            row.append(i); col.append(j); data.append(w)
    return csr_matrix((data, (row, col)), shape=(N, N), dtype=np.float64)


def pendant_dead_end_mask(links_order, link_endpoints):
    """A link is a dead-end if either endpoint touches a pendant node
    (node with only one unique undirected neighbour)."""
    node_nbrs = {}
    for lid in links_order:
        ep = link_endpoints.get(str(lid))
        if ep is None: continue
        fn, tn = ep
        node_nbrs.setdefault(fn, set()).add(tn)
        node_nbrs.setdefault(tn, set()).add(fn)
    pendant = {n for n, nbrs in node_nbrs.items() if len(nbrs) <= 1}
    N = len(links_order)
    m = np.zeros(N, dtype=bool)
    for i, lid in enumerate(links_order):
        ep = link_endpoints.get(str(lid))
        if ep is None: continue
        fn, tn = ep
        if fn in pendant or tn in pendant:
            m[i] = True
    return m, len(pendant)


def km_dist(mid, lat0, lon0):
    R_km = 6371.0
    cos_lat = np.cos(np.deg2rad(lat0))
    dlat_km = (mid[:, 1] - lat0) * np.pi / 180.0 * R_km
    dlon_km = (mid[:, 0] - lon0) * np.pi / 180.0 * R_km * cos_lat
    return np.sqrt(dlat_km ** 2 + dlon_km ** 2)


def main():
    t0 = time.time()
    print("[load] graph ...")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    print("[load] phase kernels ...")
    P_up_cs, P_down_cs = build_phase_kernels(graph_data)
    P_diff = build_diffusion_P(graph_data, links)

    print("[load] popularity ...")
    b = load_popularity_npz(str(config.DATA_DIR / "popularity_results_osm.npz"))
    matrix = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_link_ids = b["link_ids"]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_link_ids], dtype=np.int64)
    valid = proj >= 0
    time_index = {t: i for i, t in enumerate(times)}

    def get_E(time_str):
        ti = time_index.get(time_str)
        if ti is None: return None
        row = matrix.getrow(ti).toarray().ravel().astype(np.float64)
        C = np.zeros(N, dtype=np.float64)
        np.add.at(C, proj[valid], row[valid])
        c = C + ALPHA
        c_diff = (1.0 - GAMMA) * c + GAMMA * (P_diff @ c)
        s = c_diff.sum()
        return c_diff / s if s > 0 else None

    print("[graph] parsing XML, finding pendant dead-ends ...")
    _, link_endpoints = parse_matsim_xml(str(config.NETWORK_XML))
    dead_mask, n_pendant = pendant_dead_end_mask(links, link_endpoints)
    n_dead = int(dead_mask.sum())
    print(f"  pendant nodes: {n_pendant:,}; dead-end links: {n_dead:,} "
          f"({100.0*n_dead/N:.2f}%)")

    # Build all_1km cluster (with dead-end filter)
    mid = np.full((N, 2), np.nan, dtype=np.float64)
    for i, lid in enumerate(links):
        ep = link_endpoints.get(str(lid))
        if ep is None: continue
        # nodes were not returned; we don't need them once we have endpoints
    # We need node lat/lon for midpoints. Re-parse for nodes.
    nodes, _ = parse_matsim_xml(str(config.NETWORK_XML))
    for i, lid in enumerate(links):
        ep = link_endpoints.get(str(lid))
        if ep is None: continue
        fn, tn = ep
        if fn not in nodes or tn not in nodes: continue
        a = nodes[fn]; b_ = nodes[tn]
        mid[i] = [0.5*(a[0]+b_[0]), 0.5*(a[1]+b_[1])]

    d_km = km_dist(mid, STADIUM_LAT, STADIUM_LON)
    ring_raw  = d_km <= CLUSTER_RADIUS_KM
    ring_no_dead = ring_raw & (~dead_mask)
    n_ring  = int(ring_raw.sum())
    n_clean = int(ring_no_dead.sum())
    n_dropped = n_ring - n_clean
    print(f"[cluster] 1 km ring raw       : {n_ring} links")
    print(f"[cluster] 1 km ring no-dead   : {n_clean} links "
          f"({n_dropped} dead-end links dropped)")

    eps = 1e-6
    bins_prior_ctrl = bins_for_day("2018-09-01")
    bins_target_ctrl = bins_for_day("2018-09-08")
    bins_prior_evt  = bins_for_day("2018-09-08")
    bins_target_evt = bins_for_day("2018-09-15")

    def run_pair(prior_bins, target_bins, beta, rho):
        d2d, d2m = [], []
        for bp, bt in zip(prior_bins, target_bins):
            E_in = get_E(bp); E_tg = get_E(bt)
            if E_in is None or E_tg is None: continue
            v_up, v_down, _ = power_iteration(P_up_cs, P_down_cs, E_in,
                                               beta, rho, TOL, MAX_ITER)
            v = v_up + v_down; s = v.sum()
            if s > 0: v /= s
            denom = E_tg + eps
            err_d2d = np.abs(E_tg - E_in) / denom
            err_d2m = np.abs(E_tg - v)    / denom
            d2d.append(float(err_d2d[ring_no_dead].mean()))
            d2m.append(float(err_d2m[ring_no_dead].mean()))
        return float(np.mean(d2d)), float(np.mean(d2m))

    print("\n--- EXPERIMENT 1: default (beta=0.102, rho=0.101) ---")
    ctrl_d2d, ctrl_d2m = run_pair(bins_prior_ctrl, bins_target_ctrl, BETA_DEF, RHO_DEF)
    evt_d2d,  evt_d2m  = run_pair(bins_prior_evt,  bins_target_evt,  BETA_DEF, RHO_DEF)
    print(f"  control (Sept 1 -> 8) : d2d={ctrl_d2d:.4f}  d2m={ctrl_d2m:.4f}")
    print(f"  event   (Sept 8 -> 15): d2d={evt_d2d:.4f}  d2m={evt_d2m:.4f}")

    print("\n--- EXPERIMENT 3: 4D tune (event pair only, all_1km no-dead) ---")
    best = dict(mre=float("inf"), params=None)
    t_g = time.time()
    done = 0
    for a_s in GRID_ALPHA_S:
        for a_l in GRID_ALPHA_L:
            pagerank.PARAMS["alpha_s"] = a_s
            pagerank.PARAMS["alpha_l"] = a_l
            Pu, Pd = build_phase_kernels(graph_data)
            for beta in GRID_BETA:
                for rho in GRID_RHO:
                    d2d_b, d2m = [], []
                    for bp, bt in zip(bins_prior_evt, bins_target_evt):
                        E_in = get_E(bp); E_tg = get_E(bt)
                        if E_in is None or E_tg is None: continue
                        vu, vd, _ = power_iteration(Pu, Pd, E_in, beta, rho, TOL, MAX_ITER)
                        v = vu + vd; s = v.sum()
                        if s > 0: v /= s
                        err = np.abs(E_tg - v) / (E_tg + eps)
                        d2m.append(float(err[ring_no_dead].mean()))
                    m = float(np.mean(d2m))
                    if m < best["mre"]:
                        best = dict(mre=m, params=dict(alpha_s=a_s, alpha_l=a_l,
                                                          beta=beta, rho=rho))
                    done += 1
                    if done % 18 == 0:
                        print(f"  {done}/144  t={time.time()-t_g:.0f}s  best={best['mre']:.4f}",
                              flush=True)
    print(f"\nBest 4D: {best['params']}  MRE = {best['mre']:.4f}")
    red = (1 - best["mre"] / evt_d2m) * 100.0
    print(f"\nSummary on all_1km (305 - {n_dropped} = {n_clean} non-dead-end links):")
    print(f"  d2d (event)         = {evt_d2d:.4f}")
    print(f"  d2m default         = {evt_d2m:.4f}")
    print(f"  d2m tuned (4D best) = {best['mre']:.4f}")
    print(f"  reduction           = {red:.1f}%")
    print(f"  best params         = {best['params']}")

    out = dict(
        cluster="all_1km_no_dead",
        radius_km=CLUSTER_RADIUS_KM,
        cluster_size=n_clean,
        n_dead_dropped=n_dropped,
        n_pendant_nodes=n_pendant,
        control_d2d=ctrl_d2d, control_d2m=ctrl_d2m,
        event_d2d=evt_d2d,    event_d2m=evt_d2m,
        tuned_best=best,
        reduction_pct=red,
    )
    with open(OUT / "all_1km_no_dead_results.json", "w") as f:
        json.dump(out, f, indent=2)
    # also write a mask file so the plotter can pick it up exactly
    np.save(OUT / "all_1km_no_dead_mask.npy", ring_no_dead)
    print(f"\nsaved {OUT/'all_1km_no_dead_results.json'}")
    print(f"saved {OUT/'all_1km_no_dead_mask.npy'}")
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
