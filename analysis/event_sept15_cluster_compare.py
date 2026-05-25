"""Compare cluster definitions for the Sept 15 event forecast.

Four cluster definitions:

  A. top-25  most-trafficked links within 1 km of Rice-Eccles
  B. top-50  most-trafficked links within 1 km of Rice-Eccles
  C. top-100 most-trafficked links within 1 km of Rice-Eccles
  D. 3-hop neighbour cluster: links within 200 m of the stadium
     (seed), plus their in/out neighbours, recursively for 3 layers.

For each cluster we run the same experiment:

  Experiment 1: default chain (beta=0.102, rho=0.101) on the
                Sept 8 -> Sept 15 event pair and on the
                Sept 1 -> Sept 8 control pair, report data->data
                and data->model corridor MRE.

  Experiment 3: 4D grid over (alpha_s, alpha_l, beta, rho) on the
                event pair; for each parameter combo the same chain
                output is scored on all 4 cluster definitions in one
                pass. We report the best 4D parameters per cluster.

Run-time tip: chain outputs are shared across the 4 clusters, so
the work is dominated by 144 + (control + event) chain runs.
"""
import json
import re
import sys
import time
from collections import deque
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

STADIUM_LAT = 40.7596
STADIUM_LON = -111.8485
CLUSTER_RADIUS_KM = 1.0
NEIGHBOR_SEED_M   = 200.0
NEIGHBOR_HOPS     = 3

WINDOW_START = "19:00:00"
WINDOW_END   = "19:55:00"

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
    out = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d %H:%M:%S"))
        cur += timedelta(minutes=5)
    return out


def parse_matsim_xml(xml_path):
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
    return nodes, link_endpoints


def identify_pendant_dead_ends(links_order, link_endpoints):
    """A pendant node is a node with only one unique undirected neighbour
    in the road graph. A dead-end link is any link whose source or
    destination is a pendant node (so cul-de-sacs and graph stubs both
    drop, both directions)."""
    node_neighbors = {}
    for lid in links_order:
        ep = link_endpoints.get(str(lid))
        if ep is None:
            continue
        fn, tn = ep
        node_neighbors.setdefault(fn, set()).add(tn)
        node_neighbors.setdefault(tn, set()).add(fn)
    pendant = {n for n, nbrs in node_neighbors.items() if len(nbrs) <= 1}
    N = len(links_order)
    mask = np.zeros(N, dtype=bool)
    for i, lid in enumerate(links_order):
        ep = link_endpoints.get(str(lid))
        if ep is None:
            continue
        fn, tn = ep
        if fn in pendant or tn in pendant:
            mask[i] = True
    return mask, len(pendant)


def build_diffusion_P(graph_data, links_order):
    id_to_idx = {lid: i for i, lid in enumerate(links_order)}
    adj = graph_data.get('adjacency', {})
    N = len(links_order)
    row, col, data = [], [], []
    for i, lid in enumerate(links_order):
        succ = [id_to_idx[ol] for ol in adj.get(lid, []) if ol in id_to_idx]
        if not succ:
            continue
        w = 1.0 / len(succ)
        for j in succ:
            row.append(i); col.append(j); data.append(w)
    return csr_matrix((data, (row, col)), shape=(N, N), dtype=np.float64)


def link_midpoints(links_order, nodes, link_endpoints):
    N = len(links_order)
    mid = np.full((N, 2), np.nan, dtype=np.float64)
    for i, lid in enumerate(links_order):
        ep = link_endpoints.get(str(lid))
        if ep is None or ep[0] not in nodes or ep[1] not in nodes:
            continue
        lon_a, lat_a = nodes[ep[0]]
        lon_b, lat_b = nodes[ep[1]]
        mid[i] = [0.5 * (lon_a + lon_b), 0.5 * (lat_a + lat_b)]
    return mid


def km_distance(mid, lat0, lon0):
    """Equirectangular approximation, returns km from each midpoint to (lat0, lon0)."""
    R_km = 6371.0
    cos_lat = np.cos(np.deg2rad(lat0))
    dlat_km = (mid[:, 1] - lat0) * np.pi / 180.0 * R_km
    dlon_km = (mid[:, 0] - lon0) * np.pi / 180.0 * R_km * cos_lat
    return np.sqrt(dlat_km ** 2 + dlon_km ** 2)


def build_ring_mask(mid, radius_km):
    d = km_distance(mid, STADIUM_LAT, STADIUM_LON)
    return d <= radius_km


def build_topk_cluster(ring_mask, popularity, k):
    """Top-k most popular links within ring_mask."""
    score = np.where(ring_mask, popularity, -1.0)
    top_idx = np.argsort(score)[::-1][:k]
    m = np.zeros_like(ring_mask, dtype=bool)
    m[top_idx] = True
    return m


def build_neighbor_cluster(graph_data, links_order, mid, seed_radius_m, n_hops):
    """Seed = links within seed_radius_m of stadium; BFS n_hops via in+out neighbours."""
    N = len(links_order)
    lid_to_idx = {lid: i for i, lid in enumerate(links_order)}
    adj_dict = graph_data.get('adjacency', {})
    # out adj
    out_adj = [[] for _ in range(N)]
    in_adj  = [[] for _ in range(N)]
    for lid, out_list in adj_dict.items():
        i = lid_to_idx.get(lid)
        if i is None:
            continue
        for next_lid in out_list:
            j = lid_to_idx.get(next_lid)
            if j is not None:
                out_adj[i].append(j)
                in_adj[j].append(i)
    # seed
    d_km = km_distance(mid, STADIUM_LAT, STADIUM_LON)
    seed = np.where(d_km <= seed_radius_m / 1000.0)[0].tolist()
    visited = set(seed)
    frontier = set(seed)
    for _ in range(n_hops):
        nxt = set()
        for u in frontier:
            for v in out_adj[u]:
                if v not in visited:
                    nxt.add(v); visited.add(v)
            for v in in_adj[u]:
                if v not in visited:
                    nxt.add(v); visited.add(v)
        frontier = nxt
    m = np.zeros(N, dtype=bool)
    for i in visited:
        m[i] = True
    return m, len(seed)


def load_corpus():
    print("[load] graph + popularity ...")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    print("[load] phase kernels ...")
    P_up_cs, P_down_cs = build_phase_kernels(graph_data)
    P_diff = build_diffusion_P(graph_data, links)

    print("[load] popularity npz ...")
    b = load_popularity_npz(str(config.DATA_DIR / "popularity_results_osm.npz"))
    matrix = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_link_ids = b["link_ids"]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_link_ids], dtype=np.int64)
    valid = proj >= 0
    time_index = {t: i for i, t in enumerate(times)}

    def get_E_diffused(time_str):
        ti = time_index.get(time_str)
        if ti is None:
            return None
        row = matrix.getrow(ti).toarray().ravel().astype(np.float64)
        C = np.zeros(N, dtype=np.float64)
        np.add.at(C, proj[valid], row[valid])
        c = C + ALPHA
        c_diff = (1.0 - GAMMA) * c + GAMMA * (P_diff @ c)
        s = c_diff.sum()
        return c_diff / s if s > 0 else None

    return graph_data, links, P_up_cs, P_down_cs, get_E_diffused, N


def baseline_popularity(get_E, N):
    accum = np.zeros(N, dtype=np.float64)
    n = 0
    for day in CORRIDOR_RANK_DAYS:
        cur = datetime.strptime(f"{day} {CORRIDOR_RANK_START}", "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(f"{day} {CORRIDOR_RANK_END}",   "%Y-%m-%d %H:%M:%S")
        while cur <= end:
            E = get_E(cur.strftime("%Y-%m-%d %H:%M:%S"))
            if E is not None:
                accum += E
                n += 1
            cur += timedelta(minutes=5)
    accum /= max(n, 1)
    return accum, n


def forecast_pair_multi(P_up_cs, P_down_cs, get_E, prior_day, target_day,
                          beta, rho, cluster_masks):
    """For each event-window bin, run chain on prior_day's diffused E_b
    and compute MRE on all_links and on every cluster_mask."""
    prior_bins  = bins_for_day(prior_day)
    target_bins = bins_for_day(target_day)
    eps = 1e-6
    out = {k: {"d2d": [], "d2m": []} for k in cluster_masks}
    out["all"] = {"d2d": [], "d2m": []}
    n_bins = 0
    for bp, bt in zip(prior_bins, target_bins):
        E_in = get_E(bp); E_tg = get_E(bt)
        if E_in is None or E_tg is None:
            continue
        v_up, v_down, _ = power_iteration(P_up_cs, P_down_cs, E_in,
                                           beta, rho, TOL, MAX_ITER)
        v = v_up + v_down; s = v.sum()
        if s > 0:
            v /= s
        denom = E_tg + eps
        err_d2d = np.abs(E_tg - E_in) / denom
        err_d2m = np.abs(E_tg - v)    / denom
        out["all"]["d2d"].append(float(err_d2d.mean()))
        out["all"]["d2m"].append(float(err_d2m.mean()))
        for k, m in cluster_masks.items():
            out[k]["d2d"].append(float(err_d2d[m].mean()))
            out[k]["d2m"].append(float(err_d2m[m].mean()))
        n_bins += 1
    return out, n_bins


def main():
    t0 = time.time()
    graph_data, links, P_up_cs, P_down_cs, get_E, N = load_corpus()
    print(f"  N = {N:,}")

    print("[load] parsing slc_network.xml for coordinates ...")
    nodes, link_endpoints = parse_matsim_xml(str(config.NETWORK_XML))
    mid = link_midpoints(links, nodes, link_endpoints)

    print("[baseline] computing mean popularity over baseline Saturday evenings ...")
    base_pop, n_base = baseline_popularity(get_E, N)
    print(f"  averaged over {n_base} baseline bins")

    # Build the four cluster masks
    # Identify dead-end links via pendant-node detection on the MATSim
    # XML. A pendant node touches only one unique undirected neighbour,
    # so its incident links are cul-de-sacs / driveway stubs.
    print("[graph] parsing slc_network.xml for node endpoints ...")
    _, link_endpoints = parse_matsim_xml(str(config.NETWORK_XML))
    dead_end_mask, n_pendant = identify_pendant_dead_ends(links, link_endpoints)
    n_dead = int(dead_end_mask.sum())
    print(f"[graph] pendant nodes: {n_pendant:,}; "
          f"dead-end links: {n_dead:,} / {N:,} ({100.0*n_dead/N:.1f}%)")

    ring_1km_raw = build_ring_mask(mid, CLUSTER_RADIUS_KM)
    ring_1km = ring_1km_raw & (~dead_end_mask)
    print(f"[cluster] 1 km ring: {int(ring_1km_raw.sum())} raw, "
          f"{int(ring_1km.sum())} after excluding dead-ends "
          f"({int(ring_1km_raw.sum()) - int(ring_1km.sum())} dead-ends dropped)")

    masks = {}
    masks["top25_1km"]  = build_topk_cluster(ring_1km, base_pop, 25)
    masks["top50_1km"]  = build_topk_cluster(ring_1km, base_pop, 50)
    masks["top100_1km"] = build_topk_cluster(ring_1km, base_pop, 100)
    masks["all_1km"]    = ring_1km.copy()
    neighbor_mask, n_seed = build_neighbor_cluster(
        graph_data, links, mid, NEIGHBOR_SEED_M, NEIGHBOR_HOPS)
    masks["neighbors_3hop"] = neighbor_mask & (~dead_end_mask)
    print(f"[cluster] sizes (post dead-end filter): "
          f"top25={int(masks['top25_1km'].sum())}, "
          f"top50={int(masks['top50_1km'].sum())}, "
          f"top100={int(masks['top100_1km'].sum())}, "
          f"all_1km={int(masks['all_1km'].sum())}, "
          f"neighbors_3hop={int(masks['neighbors_3hop'].sum())} "
          f"(from {n_seed} seed links within {NEIGHBOR_SEED_M:.0f} m)")

    cluster_keys = ["all", "top25_1km", "top50_1km", "top100_1km",
                     "all_1km", "neighbors_3hop"]

    # ----- Experiment 1: default chain on control + event pairs -----
    print()
    print("=" * 78)
    print(f"EXPERIMENT 1: default (beta={BETA_DEF}, rho={RHO_DEF})")
    print("=" * 78)
    ctrl, n_c = forecast_pair_multi(
        P_up_cs, P_down_cs, get_E,
        "2018-09-01", "2018-09-08",
        BETA_DEF, RHO_DEF, masks)
    evt, n_e = forecast_pair_multi(
        P_up_cs, P_down_cs, get_E,
        "2018-09-08", "2018-09-15",
        BETA_DEF, RHO_DEF, masks)
    print(f"\n{'cluster':<18s} {'size':>5s}  "
          f"{'ctrl d2d':>9s} {'ctrl d2m':>9s}  "
          f"{'evt d2d':>9s} {'evt d2m':>9s}  "
          f"{'d2m evt/ctrl':>13s}")
    for k in cluster_keys:
        sz = N if k == "all" else int(masks[k].sum())
        cd = np.mean(ctrl[k]["d2d"]); cm = np.mean(ctrl[k]["d2m"])
        ed = np.mean(evt[k]["d2d"]);  em = np.mean(evt[k]["d2m"])
        ratio = em / cm if cm > 0 else float("nan")
        print(f"{k:<18s} {sz:>5d}  {cd:>9.4f} {cm:>9.4f}  "
              f"{ed:>9.4f} {em:>9.4f}  {ratio:>12.2f}x")

    # ----- Experiment 3: 4D tune, scored on all clusters -----
    print()
    print("=" * 78)
    print("EXPERIMENT 3: 4D tune on event pair, scored on all clusters")
    print("=" * 78)
    print(f"  grid: {len(GRID_ALPHA_S)}*{len(GRID_ALPHA_L)}*"
          f"{len(GRID_BETA)}*{len(GRID_RHO)} = "
          f"{len(GRID_ALPHA_S)*len(GRID_ALPHA_L)*len(GRID_BETA)*len(GRID_RHO)} combos")

    best = {k: dict(mre=float("inf"), params=None) for k in cluster_keys}
    grid_records = []
    t_g = time.time()
    n_done = 0
    for a_s in GRID_ALPHA_S:
        for a_l in GRID_ALPHA_L:
            pagerank.PARAMS["alpha_s"] = a_s
            pagerank.PARAMS["alpha_l"] = a_l
            Pu, Pd = build_phase_kernels(graph_data)
            for beta in GRID_BETA:
                for rho in GRID_RHO:
                    out, _ = forecast_pair_multi(
                        Pu, Pd, get_E,
                        "2018-09-08", "2018-09-15",
                        beta, rho, masks)
                    rec = {"alpha_s": a_s, "alpha_l": a_l,
                            "beta": beta, "rho": rho}
                    for k in cluster_keys:
                        m = float(np.mean(out[k]["d2m"]))
                        rec[f"d2m_{k}"] = m
                        if m < best[k]["mre"]:
                            best[k]["mre"] = m
                            best[k]["params"] = dict(alpha_s=a_s, alpha_l=a_l,
                                                       beta=beta, rho=rho)
                    grid_records.append(rec)
                    n_done += 1
                    if n_done % 18 == 0:
                        print(f"  {n_done}/{144} combos, t={time.time()-t_g:.0f}s",
                              flush=True)
    print(f"  4D grid done in {time.time()-t_g:.0f}s")

    # ----- summary table -----
    print()
    print("Summary on Sept 8 -> Sept 15 event pair:")
    print(f"  {'cluster':<18s} {'size':>5s}  "
          f"{'d2d':>8s} {'d2m def':>8s}  {'d2m tuned':>9s}  "
          f"{'reduction':>9s}  {'best (a_s, a_l, beta, rho)':>30s}")
    summary = []
    for k in cluster_keys:
        sz = N if k == "all" else int(masks[k].sum())
        d2d_default = float(np.mean(evt[k]["d2d"]))
        d2m_default = float(np.mean(evt[k]["d2m"]))
        d2m_tuned   = best[k]["mre"]
        red = (1 - d2m_tuned / d2m_default) * 100 if d2m_default > 0 else float("nan")
        p = best[k]["params"]
        ps = (f"({p['alpha_s']:+.2f}, {p['alpha_l']:+.2f}, "
              f"{p['beta']:.2f}, {p['rho']:.2f})")
        print(f"  {k:<18s} {sz:>5d}  "
              f"{d2d_default:>8.4f} {d2m_default:>8.4f}  "
              f"{d2m_tuned:>9.4f}  {red:>8.1f}%  {ps:>30s}")
        summary.append(dict(cluster=k, size=sz,
                            d2d_default=d2d_default,
                            d2m_default=d2m_default,
                            d2m_tuned=d2m_tuned,
                            reduction_pct=red,
                            best_params=best[k]["params"]))

    # Save
    output = dict(
        radius_km=CLUSTER_RADIUS_KM,
        neighbor_seed_m=NEIGHBOR_SEED_M,
        neighbor_hops=NEIGHBOR_HOPS,
        cluster_sizes={k: int(masks[k].sum()) for k in masks},
        experiment_1_default={
            "beta": BETA_DEF, "rho": RHO_DEF,
            "control_pair": "2018-09-01 -> 2018-09-08",
            "event_pair":   "2018-09-08 -> 2018-09-15",
            "control": {k: {"d2d_per_bin": ctrl[k]["d2d"],
                              "d2m_per_bin": ctrl[k]["d2m"],
                              "d2d_mean":    float(np.mean(ctrl[k]["d2d"])),
                              "d2m_mean":    float(np.mean(ctrl[k]["d2m"]))}
                          for k in cluster_keys},
            "event":   {k: {"d2d_per_bin": evt[k]["d2d"],
                              "d2m_per_bin": evt[k]["d2m"],
                              "d2d_mean":    float(np.mean(evt[k]["d2d"])),
                              "d2m_mean":    float(np.mean(evt[k]["d2m"]))}
                          for k in cluster_keys},
        },
        experiment_3_tuned_4d={
            "grid": grid_records,
            "best_per_cluster": {k: best[k] for k in cluster_keys},
        },
        summary=summary,
    )
    out_path = OUT / "results_cluster_compare.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[done] {time.time()-t0:.0f}s. saved {out_path}")


if __name__ == "__main__":
    main()
