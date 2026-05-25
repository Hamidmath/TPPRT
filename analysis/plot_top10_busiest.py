"""Plot ONLY: top-10 busiest links within 1 km of Rice-Eccles, ranked by
mean popularity over 19:00-19:55 bins on the two clean baseline
Saturdays (Sept 1 and Sept 22). Sept 15 (event day) is NOT used.
Dead-end links excluded.

No experiment, no MRE - just the cluster + figure.
"""
import json
import re
import sys
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

OUT = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project/results/event_sept15")

STADIUM_LAT, STADIUM_LON = 40.7596, -111.8485
CLUSTER_RADIUS_KM = 1.0
TOP_K = 10
VIEW_RADIUS_KM = 1.5
GAMMA, ALPHA = 0.20, 0.01
WINDOW_START, WINDOW_END = "19:00:00", "19:55:00"
RANK_DAYS = ["2018-09-01", "2018-09-22"]


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


def main():
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links = list(graph_data["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

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

    accum = np.zeros(N, dtype=np.float64); n_bins = 0
    for day in RANK_DAYS:
        for t in bins_window(day):
            E = get_E(t)
            if E is not None:
                accum += E; n_bins += 1
    accum /= max(n_bins, 1)
    print(f"[rank] averaged over {n_bins} bins ({WINDOW_START}-{WINDOW_END} on {RANK_DAYS})")

    score = np.where(ring, accum, -1.0)
    top_idx = np.argsort(score)[::-1][:TOP_K]
    mask = np.zeros(N, dtype=bool); mask[top_idx] = True
    print(f"\nTop-{TOP_K} busiest cluster:")
    for r, i in enumerate(top_idx, 1):
        print(f"  {r:2d}. link id={links[i]:>6s}  popularity={accum[i]:.4e}  "
              f"dist to stadium={d_km[i]*1000:.0f} m")

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
        label=f"top-{TOP_K} busiest (no dead-ends)"))
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
    ax.set_title("Top-10 busiest links within 1 km of Rice-Eccles\n"
                  "Ranked by mean popularity over 19:00-19:55 on Sept 1 + Sept 22 "
                  "(event day excluded)",
                  fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    side = fig.add_axes([0.68, 0.05, 0.30, 0.90])
    side.axis("off")
    lines = [
        "Cluster: top-10 busiest in 1 km",
        "  excludes dead-end links",
        "",
        "Ranking source",
        "  days  : Sept 1, Sept 22 (Sat)",
        f"  bins  : 19:00-19:55 ({n_bins} total)",
        "  Sept 15 (event day): NOT used",
        "",
        f"Top-{TOP_K} members:",
    ]
    for r, i in enumerate(top_idx, 1):
        lines.append(f"  {r:2d}. id={links[i]:>5s}  "
                      f"pop={accum[i]:.2e}  "
                      f"{d_km[i]*1000:>4.0f} m")
    side.text(0.0, 1.0, "\n".join(lines), fontsize=10, family="monospace",
              va="top", ha="left", transform=side.transAxes,
              bbox=dict(facecolor="white", edgecolor="#999", linewidth=0.6, pad=10))

    out_png = OUT / "fig_stadium_corridors_top10_busiest.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nsaved {out_png}")


if __name__ == "__main__":
    main()
