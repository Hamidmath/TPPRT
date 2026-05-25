"""Build a 'top-10 busiest links within 1 km' cluster, where 'busiest'
is averaged over the SAME time window as the event window
(19:00-19:55) on the two clean baseline Saturdays (Sept 1 and Sept 22).
The event day (Sept 15) is excluded.

This gives a cluster that is:
  - inside 1 km of Rice-Eccles
  - not a dead-end (pendant node touch)
  - ranked by typical 7-8 PM Saturday traffic across two known
    non-event Saturdays

Run the same forecast experiment (default chain + 4D tune) on this
top-10 cluster and save a stadium-network figure highlighting it.
"""
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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
TOP_K = 10
VIEW_RADIUS_KM = 1.5

# Event window (also the ranking window now)
WINDOW_START, WINDOW_END = "19:00:00", "19:55:00"

# Clean baseline Saturdays (no home football game)
RANK_DAYS = ["2018-09-01", "2018-09-22"]

GRID_BETA = [0.05, 0.102, 0.25, 0.5]
GRID_RHO  = [0.05, 0.101, 0.25, 0.5]
GRID_ALPHA_S = [-0.5, 0.0, 0.5]
GRID_ALPHA_L = [-0.5, 0.0, 0.5]


def bins_window(day_str):
    start = datetime.strptime(f"{day_str} {WINDOW_START}", "%Y-%m-%d %H:%M:%S")
    end   = datetime.strptime(f"{day_str} {WINDOW_END}",   "%Y-%m-%d %H:%M:%S")
    out, cur = [], start
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d %H:%M:%S"))
        cur += timedelta(minutes=5)
    return out


def parse_matsim_xml(xml_path):
    nodes, links_ep = {}, {}
    n_re = re.compile(r'<node\s+id="(\d+)"\s+x="(-?[\d.]+)"\s+y="(-?[\d.]+)"')
    l_re = re.compile(r'<link\s+id="(\d+)"\s+from="(\d+)"\s+to="(\d+)"')
    with open(xml_path) as f:
        for line in f:
            m = n_re.search(line)
            if m:
                nodes[m.group(1)] = (float(m.group(2)), float(m.group(3))); continue
            m = l_re.search(line)
            if m:
                links_ep[m.group(1)] = (m.group(2), m.group(3))
    return nodes, links_ep


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
        if fn in pendant or tn in pendant: m[i] = True
    return m, len(pendant)


def km_dist(mid, lat0, lon0):
    R_km = 6371.0; cos_lat = np.cos(np.deg2rad(lat0))
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

    print("[load] phase kernels + diffusion P ...")
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

    print("[graph] parsing XML for endpoints + dead-end detection ...")
    nodes, link_endpoints = parse_matsim_xml(str(config.NETWORK_XML))
    dead_mask, n_pendant = pendant_dead_end_mask(links, link_endpoints)
    print(f"  pendant nodes: {n_pendant:,}; dead-end links: {int(dead_mask.sum()):,}")

    # Build link midpoints
    mid = np.full((N, 2), np.nan, dtype=np.float64)
    for i, lid in enumerate(links):
        ep = link_endpoints.get(str(lid))
        if ep is None or ep[0] not in nodes or ep[1] not in nodes: continue
        a, b_ = nodes[ep[0]], nodes[ep[1]]
        mid[i] = [0.5 * (a[0] + b_[0]), 0.5 * (a[1] + b_[1])]
    d_km = km_dist(mid, STADIUM_LAT, STADIUM_LON)
    ring = (d_km <= CLUSTER_RADIUS_KM) & (~dead_mask)
    print(f"[cluster] 1 km ring (no dead-ends): {int(ring.sum())} candidate links")

    # Rank links by mean popularity over event-time-window on baseline Saturdays
    print(f"[ranking] averaging popularity over 19:00-19:55 on {RANK_DAYS}")
    accum = np.zeros(N, dtype=np.float64); n_bins = 0
    for day in RANK_DAYS:
        for tstr in bins_window(day):
            E = get_E(tstr)
            if E is not None:
                accum += E; n_bins += 1
    accum /= max(n_bins, 1)
    print(f"  averaged over {n_bins} baseline event-window bins")

    in_ring_score = np.where(ring, accum, -1.0)
    top_idx = np.argsort(in_ring_score)[::-1][:TOP_K]
    mask = np.zeros(N, dtype=bool); mask[top_idx] = True
    print(f"[cluster] top-{TOP_K} busiest links during 19:00-19:55 (clean baselines)")

    # Print which links got picked, with their location and popularity
    print(f"\nTop-{TOP_K} cluster members:")
    for rank, i in enumerate(top_idx, 1):
        lid = links[i]
        pop = accum[i]
        d_to_stadium = d_km[i]
        print(f"  {rank:2d}. link id={lid:>6s}  popularity={pop:.4e}  "
              f"dist to stadium={d_to_stadium*1000:.0f} m")

    # ----- experiment 1: default chain on control + event pairs -----
    eps = 1e-6
    def run_pair(prior_day, target_day, beta, rho):
        d2d, d2m = [], []
        for bp, bt in zip(bins_window(prior_day), bins_window(target_day)):
            E_in = get_E(bp); E_tg = get_E(bt)
            if E_in is None or E_tg is None: continue
            vu, vd, _ = power_iteration(P_up_cs, P_down_cs, E_in,
                                         beta, rho, TOL, MAX_ITER)
            v = vu + vd; s = v.sum()
            if s > 0: v /= s
            denom = E_tg + eps
            err_d2d = np.abs(E_tg - E_in) / denom
            err_d2m = np.abs(E_tg - v)    / denom
            d2d.append(float(err_d2d[mask].mean()))
            d2m.append(float(err_d2m[mask].mean()))
        return float(np.mean(d2d)), float(np.mean(d2m))

    print("\n--- EXPERIMENT 1: default (beta=0.102, rho=0.101) ---")
    ctrl_d2d, ctrl_d2m = run_pair("2018-09-01", "2018-09-08", BETA_DEF, RHO_DEF)
    evt_d2d,  evt_d2m  = run_pair("2018-09-08", "2018-09-15", BETA_DEF, RHO_DEF)
    print(f"  control (Sept 1 -> 8) : d2d={ctrl_d2d:.4f}  d2m={ctrl_d2m:.4f}")
    print(f"  event   (Sept 8 -> 15): d2d={evt_d2d:.4f}  d2m={evt_d2m:.4f}")

    # ----- experiment 3: 4D tune on event pair -----
    print("\n--- EXPERIMENT 3: 4D tune on event pair ---")
    best = dict(mre=float("inf"), params=None)
    done = 0; t_g = time.time()
    bins_p = bins_window("2018-09-08"); bins_t = bins_window("2018-09-15")
    for a_s in GRID_ALPHA_S:
        for a_l in GRID_ALPHA_L:
            pagerank.PARAMS["alpha_s"] = a_s
            pagerank.PARAMS["alpha_l"] = a_l
            Pu, Pd = build_phase_kernels(graph_data)
            for beta in GRID_BETA:
                for rho in GRID_RHO:
                    vals = []
                    for bp, bt in zip(bins_p, bins_t):
                        E_in = get_E(bp); E_tg = get_E(bt)
                        if E_in is None or E_tg is None: continue
                        vu, vd, _ = power_iteration(Pu, Pd, E_in,
                                                      beta, rho, TOL, MAX_ITER)
                        v = vu + vd; s = v.sum()
                        if s > 0: v /= s
                        err = np.abs(E_tg - v) / (E_tg + eps)
                        vals.append(float(err[mask].mean()))
                    m = float(np.mean(vals))
                    if m < best["mre"]:
                        best = dict(mre=m, params=dict(alpha_s=a_s, alpha_l=a_l,
                                                          beta=beta, rho=rho))
                    done += 1
                    if done % 18 == 0:
                        print(f"  {done}/144 t={time.time()-t_g:.0f}s "
                              f"best={best['mre']:.4f}", flush=True)
    red = (1 - best["mre"] / evt_d2m) * 100.0
    print(f"\nSummary (top-{TOP_K} busiest within 1 km, ranked on baseline event-window):")
    print(f"  size          = {int(mask.sum())} links")
    print(f"  event d2d     = {evt_d2d:.4f}")
    print(f"  event d2m def = {evt_d2m:.4f}")
    print(f"  event d2m tuned (best 4D) = {best['mre']:.4f}")
    print(f"  reduction     = {red:.1f}%")
    print(f"  best params   = {best['params']}")

    out = dict(
        cluster=f"top{TOP_K}_busiest_in_event_window",
        radius_km=CLUSTER_RADIUS_KM,
        top_k=TOP_K,
        ranking_window=f"{WINDOW_START}-{WINDOW_END}",
        ranking_days=RANK_DAYS,
        cluster_size=int(mask.sum()),
        cluster_members=[
            dict(rank=r+1, link_id=links[i], popularity=float(accum[i]),
                 dist_m=float(d_km[i]*1000))
            for r, i in enumerate(top_idx)
        ],
        control_d2d=ctrl_d2d, control_d2m=ctrl_d2m,
        event_d2d=evt_d2d,    event_d2m=evt_d2m,
        tuned_best=best, reduction_pct=red,
    )
    with open(OUT / "top10_busiest_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # ----- figure -----
    cos_lat = np.cos(np.deg2rad(STADIUM_LAT))
    lat_per_km = 1.0 / 111.0
    lon_per_km = 1.0 / (111.0 * cos_lat)
    lat_lo = STADIUM_LAT - VIEW_RADIUS_KM * lat_per_km
    lat_hi = STADIUM_LAT + VIEW_RADIUS_KM * lat_per_km
    lon_lo = STADIUM_LON - VIEW_RADIUS_KM * lon_per_km
    lon_hi = STADIUM_LON + VIEW_RADIUS_KM * lon_per_km

    bg, fg = [], []
    for i, lid in enumerate(links):
        ep = link_endpoints.get(str(lid))
        if ep is None or ep[0] not in nodes or ep[1] not in nodes: continue
        lon_a, lat_a = nodes[ep[0]]
        lon_b, lat_b = nodes[ep[1]]
        if (max(lon_a, lon_b) < lon_lo or min(lon_a, lon_b) > lon_hi or
            max(lat_a, lat_b) < lat_lo or min(lat_a, lat_b) > lat_hi):
            continue
        seg = [(lon_a, lat_a), (lon_b, lat_b)]
        (fg if mask[i] else bg).append(seg)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.04, 0.05, 0.62, 0.90])
    ax.add_collection(LineCollection(bg, colors="#c4c4c4", linewidths=0.4, zorder=1))
    ax.add_collection(LineCollection(
        fg, colors="#c0392b", linewidths=2.5, zorder=3,
        label=f"top-{TOP_K} busiest (no dead-ends)"))
    ax.plot(STADIUM_LON, STADIUM_LAT, marker="*", markersize=22,
             markerfacecolor="#f1c40f", markeredgecolor="black",
             markeredgewidth=1.0, zorder=5, label="Rice-Eccles Stadium")
    theta = np.linspace(0, 2*np.pi, 200)
    circ_lon = STADIUM_LON + CLUSTER_RADIUS_KM * lon_per_km * np.cos(theta)
    circ_lat = STADIUM_LAT + CLUSTER_RADIUS_KM * lat_per_km * np.sin(theta)
    ax.plot(circ_lon, circ_lat, color="#2c3e50", lw=0.8, ls=":", alpha=0.7,
             zorder=4, label=f"{CLUSTER_RADIUS_KM:.1f} km ring")
    ax.set_xlim(lon_lo, lon_hi); ax.set_ylim(lat_lo, lat_hi)
    ax.set_aspect(1.0 / cos_lat)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Salt Lake City road network around Rice-Eccles Stadium\n"
                  f"Top-{TOP_K} busiest links within 1 km on baseline 19:00-19:55 "
                  f"(Sept 1 + Sept 22)",
                  fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # side panel
    side = fig.add_axes([0.68, 0.05, 0.30, 0.90])
    side.axis("off")
    bp = best["params"]
    text = [
        f"Cluster: top-{TOP_K} busiest (1 km ring, no dead-ends)",
        f"  size            : {int(mask.sum())} links",
        f"",
        f"Ranking",
        f"  baseline days   : Sept 1, Sept 22 (Sat, no event)",
        f"  ranking window  : 19:00-19:55 (1 h pre-kickoff)",
        f"  baseline bins   : {n_bins}",
        f"",
        f"Event: Sept 8 -> Sept 15 (kickoff 8 PM)",
        f"Window: 19:00-19:55 MDT (12 bins)",
        f"Diffusion: gamma={GAMMA}, alpha={ALPHA}",
        f"",
        f"MRE on event pair",
        f"  d2d (noise floor) : {evt_d2d:.4f}",
        f"  d2m default chain : {evt_d2m:.4f}",
        f"  d2m tuned 4D      : {best['mre']:.4f}",
        f"  reduction         : {red:.1f} %",
        f"",
        f"Best 4D parameters",
        f"  alpha_s = {bp['alpha_s']:+.2f}",
        f"  alpha_l = {bp['alpha_l']:+.2f}",
        f"  beta    = {bp['beta']:.2f}",
        f"  rho     = {bp['rho']:.2f}",
        f"",
        f"Reference (control pair, default)",
        f"  d2d : {ctrl_d2d:.4f}",
        f"  d2m : {ctrl_d2m:.4f}",
    ]
    side.text(0.0, 1.0, "\n".join(text), fontsize=10, family="monospace",
              va="top", ha="left", transform=side.transAxes,
              bbox=dict(facecolor="white", edgecolor="#999", linewidth=0.6, pad=10))
    out_png = OUT / "fig_stadium_corridors_top10_busiest.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nsaved {out_png}")
    print(f"saved {OUT/'top10_busiest_results.json'}")
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
