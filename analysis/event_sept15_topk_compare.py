"""Compare top-10, top-20, top-30 busiest links within 1 km of
Rice-Eccles. Each cluster:

  - inside 1 km of the stadium
  - dead-end (pendant) links excluded
  - ranked by mean diffused popularity over 19:00-19:55 on the two
    clean baseline Saturdays (Sept 1 + Sept 22). Sept 15 NOT used.

For each K in {10, 20, 30}:
  - save a stadium-network figure highlighting the K links
  - score MRE on Sept 8 -> Sept 15 event pair, default + 4D tuned.

Chain runs are shared across the three clusters (only the mask
varies in the MRE step), so total wall time is one 4D sweep.
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

STADIUM_LAT, STADIUM_LON = 40.7596, -111.8485
CLUSTER_RADIUS_KM = 1.0
TOP_KS = [10, 20, 30]
VIEW_RADIUS_KM = 1.5
GAMMA, ALPHA = 0.20, 0.01

WINDOW_START, WINDOW_END = "19:00:00", "19:55:00"
RANK_DAYS = ["2018-09-01", "2018-09-22"]

BETA_DEF, RHO_DEF = 0.102, 0.101
TOL, MAX_ITER = 1e-6, 200
GRID_BETA = [0.05, 0.102, 0.25, 0.5]
GRID_RHO  = [0.05, 0.101, 0.25, 0.5]
GRID_ALPHA_S = [-0.5, 0.0, 0.5]
GRID_ALPHA_L = [-0.5, 0.0, 0.5]


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
    return m


def bins_window(day_str):
    start = datetime.strptime(f"{day_str} {WINDOW_START}", "%Y-%m-%d %H:%M:%S")
    end   = datetime.strptime(f"{day_str} {WINDOW_END}",   "%Y-%m-%d %H:%M:%S")
    out, cur = [], start
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d %H:%M:%S"))
        cur += timedelta(minutes=5)
    return out


def make_figure(k, mask, links, nodes, link_endpoints, top_idx, accum, d_km,
                 n_rank_bins, save_path):
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
        lon_a, lat_a = nodes[ep[0]]; lon_b, lat_b = nodes[ep[1]]
        if (max(lon_a, lon_b) < lon_lo or min(lon_a, lon_b) > lon_hi or
            max(lat_a, lat_b) < lat_lo or min(lat_a, lat_b) > lat_hi):
            continue
        seg = [(lon_a, lat_a), (lon_b, lat_b)]
        (fg if mask[i] else bg).append(seg)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.04, 0.05, 0.62, 0.90])
    ax.add_collection(LineCollection(bg, colors="#c4c4c4", linewidths=0.4, zorder=1))
    ax.add_collection(LineCollection(
        fg, colors="#c0392b", linewidths=2.8, zorder=3,
        label=f"top-{k} busiest (no dead-ends)"))
    ax.plot(STADIUM_LON, STADIUM_LAT, marker="*", markersize=22,
             markerfacecolor="#f1c40f", markeredgecolor="black",
             markeredgewidth=1.0, zorder=5, label="Rice-Eccles Stadium")
    theta = np.linspace(0, 2 * np.pi, 200)
    circ_lon = STADIUM_LON + CLUSTER_RADIUS_KM * lon_per_km * np.cos(theta)
    circ_lat = STADIUM_LAT + CLUSTER_RADIUS_KM * lat_per_km * np.sin(theta)
    ax.plot(circ_lon, circ_lat, color="#2c3e50", lw=0.8, ls=":", alpha=0.7,
             zorder=4, label=f"{CLUSTER_RADIUS_KM:.1f} km ring")
    ax.set_xlim(lon_lo, lon_hi); ax.set_ylim(lat_lo, lat_hi)
    ax.set_aspect(1.0 / cos_lat)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Top-{k} busiest links within 1 km of Rice-Eccles\n"
                  "Ranked by mean popularity over 19:00-19:55 on Sept 1 + Sept 22",
                  fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    side = fig.add_axes([0.68, 0.05, 0.30, 0.90])
    side.axis("off")
    lines = [
        f"Cluster: top-{k} busiest in 1 km",
        "  excludes dead-end links",
        "",
        "Ranking source",
        "  days  : Sept 1, Sept 22 (Sat)",
        f"  bins  : 19:00-19:55 ({n_rank_bins} total)",
        "  Sept 15: NOT used",
        "",
        f"Top-{k} members:",
    ]
    for r, i in enumerate(top_idx, 1):
        lines.append(f"  {r:2d}. id={links[i]:>5s}  "
                      f"pop={accum[i]:.2e}  "
                      f"{d_km[i]*1000:>4.0f} m")
    side.text(0.0, 1.0, "\n".join(lines), fontsize=9, family="monospace",
              va="top", ha="left", transform=side.transAxes,
              bbox=dict(facecolor="white", edgecolor="#999", linewidth=0.6, pad=10))
    fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    t0 = time.time()
    print("[load] graph ...")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links = list(graph_data["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    P_up_cs, P_down_cs = build_phase_kernels(graph_data)
    P_diff = build_diffusion_P(graph_data, links)

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

    nodes, link_endpoints = parse_matsim_xml(str(config.NETWORK_XML))
    dead_mask = pendant_dead_end_mask(links, link_endpoints)
    mid = np.full((N, 2), np.nan)
    for i, lid in enumerate(links):
        ep = link_endpoints.get(str(lid))
        if ep is None or ep[0] not in nodes or ep[1] not in nodes: continue
        a, b_ = nodes[ep[0]], nodes[ep[1]]
        mid[i] = [0.5 * (a[0] + b_[0]), 0.5 * (a[1] + b_[1])]
    R_km = 6371.0
    cos_lat = np.cos(np.deg2rad(STADIUM_LAT))
    dlat_km = (mid[:, 1] - STADIUM_LAT) * np.pi / 180.0 * R_km
    dlon_km = (mid[:, 0] - STADIUM_LON) * np.pi / 180.0 * R_km * cos_lat
    d_km = np.sqrt(dlat_km**2 + dlon_km**2)
    ring = (d_km <= CLUSTER_RADIUS_KM) & (~dead_mask)
    print(f"[ring] 1 km, no dead-ends: {int(ring.sum())} candidate links")

    accum = np.zeros(N, dtype=np.float64); n_rank_bins = 0
    for day in RANK_DAYS:
        for t in bins_window(day):
            E = get_E(t)
            if E is not None:
                accum += E; n_rank_bins += 1
    accum /= max(n_rank_bins, 1)
    print(f"[rank] over {n_rank_bins} bins ({WINDOW_START}-{WINDOW_END} on {RANK_DAYS})")

    score = np.where(ring, accum, -1.0)
    sorted_idx = np.argsort(score)[::-1]

    masks = {}
    top_idx_per_k = {}
    for k in TOP_KS:
        top_idx_per_k[k] = sorted_idx[:k].copy()
        m = np.zeros(N, dtype=bool); m[top_idx_per_k[k]] = True
        masks[k] = m

    # ---------- Step 1: figures ----------
    print("\n[step 1] saving figures ...")
    for k in TOP_KS:
        out_png = OUT / f"fig_stadium_corridors_top{k}_busiest.png"
        make_figure(k, masks[k], links, nodes, link_endpoints,
                     top_idx_per_k[k], accum, d_km, n_rank_bins, out_png)
        np.save(OUT / f"top{k}_busiest_mask.npy", masks[k])
        print(f"  saved {out_png}")

    # ---------- Step 2: experiment, scored on all three masks ----------
    print("\n[step 2] experiment on top-10, top-20, top-30 simultaneously ...")
    eps = 1e-6

    def score_chain_output(v, E_tg, E_in):
        denom = E_tg + eps
        err_d2d = np.abs(E_tg - E_in) / denom
        err_d2m = np.abs(E_tg - v)    / denom
        out = {}
        for k in TOP_KS:
            mk = masks[k]
            out[k] = (float(err_d2d[mk].mean()),
                       float(err_d2m[mk].mean()))
        return out

    def run_pair_all(prior_day, target_day, beta, rho, P_up, P_down):
        bins_p = bins_window(prior_day); bins_t = bins_window(target_day)
        per_k = {k: dict(d2d=[], d2m=[]) for k in TOP_KS}
        for bp, bt in zip(bins_p, bins_t):
            E_in = get_E(bp); E_tg = get_E(bt)
            if E_in is None or E_tg is None: continue
            vu, vd, _ = power_iteration(P_up, P_down, E_in,
                                         beta, rho, TOL, MAX_ITER)
            v = vu + vd; s = v.sum()
            if s > 0: v /= s
            sc = score_chain_output(v, E_tg, E_in)
            for k in TOP_KS:
                per_k[k]["d2d"].append(sc[k][0])
                per_k[k]["d2m"].append(sc[k][1])
        return {k: dict(d2d=float(np.mean(per_k[k]["d2d"])),
                          d2m=float(np.mean(per_k[k]["d2m"])))
                 for k in TOP_KS}

    print("  default control (Sept 1 -> 8) ...")
    ctrl = run_pair_all("2018-09-01", "2018-09-08", BETA_DEF, RHO_DEF,
                          P_up_cs, P_down_cs)
    print("  default event (Sept 8 -> 15) ...")
    evt  = run_pair_all("2018-09-08", "2018-09-15", BETA_DEF, RHO_DEF,
                          P_up_cs, P_down_cs)

    print("  4D tune sweep on event pair ...")
    best = {k: dict(mre=float("inf"), params=None) for k in TOP_KS}
    bins_p = bins_window("2018-09-08"); bins_t = bins_window("2018-09-15")
    done = 0; t_g = time.time()
    for a_s in GRID_ALPHA_S:
        for a_l in GRID_ALPHA_L:
            pagerank.PARAMS["alpha_s"] = a_s
            pagerank.PARAMS["alpha_l"] = a_l
            Pu, Pd = build_phase_kernels(graph_data)
            for beta in GRID_BETA:
                for rho in GRID_RHO:
                    per_k = {k: [] for k in TOP_KS}
                    for bp, bt in zip(bins_p, bins_t):
                        E_in = get_E(bp); E_tg = get_E(bt)
                        if E_in is None or E_tg is None: continue
                        vu, vd, _ = power_iteration(Pu, Pd, E_in,
                                                      beta, rho, TOL, MAX_ITER)
                        v = vu + vd; s = v.sum()
                        if s > 0: v /= s
                        err = np.abs(E_tg - v) / (E_tg + eps)
                        for k in TOP_KS:
                            per_k[k].append(float(err[masks[k]].mean()))
                    for k in TOP_KS:
                        mre = float(np.mean(per_k[k]))
                        if mre < best[k]["mre"]:
                            best[k] = dict(mre=mre,
                                            params=dict(alpha_s=a_s,
                                                          alpha_l=a_l,
                                                          beta=beta, rho=rho))
                    done += 1
                    if done % 18 == 0:
                        print(f"    {done}/144 t={time.time()-t_g:.0f}s "
                              f"best10={best[10]['mre']:.3f} "
                              f"best20={best[20]['mre']:.3f} "
                              f"best30={best[30]['mre']:.3f}", flush=True)

    # ---------- summary ----------
    print()
    print("=" * 90)
    print("Summary on Sept 8 -> Sept 15 event pair "
          "(corridors ranked on baseline 19:00-19:55)")
    print("=" * 90)
    print(f"  {'cluster':<12s} {'size':>5s}  "
          f"{'evt d2d':>9s}  {'d2m def':>9s}  {'d2m tuned':>10s}  "
          f"{'reduction':>10s}  {'best (a_s, a_l, beta, rho)':>30s}")
    summary = []
    for k in TOP_KS:
        red = (1 - best[k]["mre"] / evt[k]["d2m"]) * 100.0
        bp = best[k]["params"]
        ps = (f"({bp['alpha_s']:+.2f}, {bp['alpha_l']:+.2f}, "
              f"{bp['beta']:.2f}, {bp['rho']:.2f})")
        print(f"  top-{k:<6d} {k:>5d}  "
              f"{evt[k]['d2d']:>9.4f}  {evt[k]['d2m']:>9.4f}  "
              f"{best[k]['mre']:>10.4f}  {red:>9.1f}%  {ps:>30s}")
        summary.append(dict(cluster=f"top-{k}", k=k,
                             evt_d2d=evt[k]["d2d"],
                             d2m_default=evt[k]["d2m"],
                             d2m_tuned=best[k]["mre"],
                             reduction_pct=red,
                             best_params=best[k]["params"]))

    with open(OUT / "topk_compare_results.json", "w") as f:
        json.dump(dict(
            ks=TOP_KS,
            radius_km=CLUSTER_RADIUS_KM,
            ranking_window=f"{WINDOW_START}-{WINDOW_END}",
            ranking_days=RANK_DAYS,
            cluster_members={
                k: [dict(rank=r+1, link_id=links[i], popularity=float(accum[i]),
                          dist_m=float(d_km[i]*1000))
                     for r, i in enumerate(top_idx_per_k[k])]
                for k in TOP_KS
            },
            control_pair={k: ctrl[k] for k in TOP_KS},
            event_pair={k: evt[k] for k in TOP_KS},
            tuned_best={k: best[k] for k in TOP_KS},
            summary=summary,
        ), f, indent=2)
    print(f"\nsaved {OUT/'topk_compare_results.json'}")
    print(f"\n[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
