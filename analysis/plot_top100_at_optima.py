"""Plot the SLC road network with the top-100 chain-output links
highlighted, for six (chain, cluster-optimum) combinations:

  Single-phase chain at:
    - top-10 optimum  (p, alpha_s, alpha_l) = (0.01000, -394.92, -13.77)
    - top-20 optimum  (p, alpha_s, alpha_l) = (0.01000, -250.69, -13.42)
    - top-30 optimum  (p, alpha_s, alpha_l) = (0.01000,  -20.95, +11.54)

  Two-phase chain at:
    - top-10 optimum  (alpha_s, alpha_l, beta, rho) = (-25.33,  -2.08, 0.0100, 0.1013)
    - top-20 optimum  (alpha_s, alpha_l, beta, rho) = (-16.93,  -1.83, 0.0100, 0.0996)
    - top-30 optimum  (alpha_s, alpha_l, beta, rho) = (+17.51, -10.12, 0.1500, 0.0100)

Chain is run on each of the 12 five-minute bins in 19:00-19:55
on Sept 8, 2018; the output distributions are averaged into a
single popularity vector; the top-100 of THAT vector are the
links highlighted.
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

HERE = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project")
sys.path.insert(0, str(HERE))
import config
from core.io import load_popularity_npz
from core import pagerank
from core.pagerank import build_phase_kernels, power_iteration

OUT = HERE / "results" / "event_sept15" / "top100_network"
OUT.mkdir(parents=True, exist_ok=True)

GAMMA, ALPHA = 0.20, 0.01
TOL, MAX_ITER = 1e-6, 200
WINDOW_START, WINDOW_END = "19:00:00", "19:55:00"
EVENT_DAY = "2018-09-08"

STADIUM_LAT = 40.7596
STADIUM_LON = -111.8485

# Optima from the Sept 15 event-window GD
SP_OPTIMA = {
    10: dict(p=0.01000, alpha_s=-394.92, alpha_l=-13.77),
    20: dict(p=0.01000, alpha_s=-250.69, alpha_l=-13.42),
    30: dict(p=0.01000, alpha_s= -20.95, alpha_l=+11.54),
}
TP_OPTIMA = {
    10: dict(alpha_s=-25.33, alpha_l= -2.08, beta=0.0100, rho=0.1013),
    20: dict(alpha_s=-16.93, alpha_l= -1.83, beta=0.0100, rho=0.0996),
    30: dict(alpha_s=+17.51, alpha_l=-10.12, beta=0.1500, rho=0.0100),
}


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


def build_diffusion_P(graph_data, links):
    id_to_idx = {lid: i for i, lid in enumerate(links)}
    adj = graph_data.get("adjacency", {})
    N = len(links)
    row, col, data = [], [], []
    for i, lid in enumerate(links):
        succ = [id_to_idx[ol] for ol in adj.get(lid, []) if ol in id_to_idx]
        if not succ: continue
        w = 1.0 / len(succ)
        for j in succ:
            row.append(i); col.append(j); data.append(w)
    return csr_matrix((data, (row, col)), shape=(N, N), dtype=np.float64)


def build_sp_P_cs(graph_data, links, lid_to_idx, weights):
    """Column-stochastic single-phase kernel weighted by `weights[j]`."""
    adj = graph_data.get("adjacency", {})
    rows, cols, data = [], [], []
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ: continue
        out_w = np.array([weights[j] for j in succ], dtype=np.float64)
        total = out_w.sum()
        if total <= 0: continue
        for k, j in enumerate(succ):
            rows.append(j); cols.append(i); data.append(out_w[k] / total)
    N = len(links)
    return csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float64)


def get_speed_lane_arrays(graph_data, links):
    speeds, lanes = [], []
    for lid in links:
        ed = graph_data["links"].get(lid, {})
        speeds.append(ed.get("speed", 11.17))
        lanes.append(ed.get("lanes", 1.0))
    s = np.array(speeds, dtype=np.float64); l = np.array(lanes, dtype=np.float64)
    if s.mean() > 0: s /= s.mean()
    if l.mean() > 0: l /= l.mean()
    return s, l


def single_phase_iter(P_cs, E_b, p, tol, max_iter):
    v = E_b.copy()
    for _ in range(max_iter):
        v_new = p * E_b + (1.0 - p) * (P_cs @ v)
        s = float(v_new.sum())
        if s > 0: v_new /= s
        diff = float(np.abs(v_new - v).sum())
        v = v_new
        if diff < tol: return v
    return v


def bins_window(day_str):
    s = datetime.strptime(f"{day_str} {WINDOW_START}", "%Y-%m-%d %H:%M:%S")
    e = datetime.strptime(f"{day_str} {WINDOW_END}",   "%Y-%m-%d %H:%M:%S")
    out, cur = [], s
    while cur <= e:
        out.append(cur.strftime("%Y-%m-%d %H:%M:%S"))
        cur += timedelta(minutes=5)
    return out


def plot_top100(links, link_endpoints, nodes, mask_top100, out_path, title):
    bg, fg = [], []
    for i, lid in enumerate(links):
        ep = link_endpoints.get(str(lid))
        if ep is None or ep[0] not in nodes or ep[1] not in nodes: continue
        a, b = nodes[ep[0]], nodes[ep[1]]
        seg = [(a[0], a[1]), (b[0], b[1])]
        (fg if mask_top100[i] else bg).append(seg)
    cos_lat = np.cos(np.deg2rad(STADIUM_LAT))
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.add_collection(LineCollection(bg, colors="#7f7f7f", linewidths=0.45,
                                       zorder=1))
    ax.add_collection(LineCollection(fg, colors="#0066cc", linewidths=2.6,
                                       zorder=3,
                                       label="top-100 chain output"))
    ax.plot(STADIUM_LON, STADIUM_LAT, marker="*", markersize=18,
             markerfacecolor="#f1c40f", markeredgecolor="black",
             markeredgewidth=1.0, zorder=5, label="Rice-Eccles Stadium")
    # Auto-extent over all nodes that appear
    xs = [a[0] for seg in bg+fg for a in seg]
    ys = [a[1] for seg in bg+fg for a in seg]
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    ax.set_aspect(1.0 / cos_lat)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.20, linewidth=0.4)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=280, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    print("[load]")
    with open(config.GRAPH_FILE) as f:
        graph_data = json.load(f)
    links = list(graph_data["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    P_diff = build_diffusion_P(graph_data, links)
    nodes, link_endpoints = parse_matsim_xml(str(config.NETWORK_XML))
    speeds, lanes = get_speed_lane_arrays(graph_data, links)
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

    bins_p = bins_window(EVENT_DAY)
    Es_prior = [get_E(t) for t in bins_p if get_E(t) is not None]
    print(f"  priors: {len(Es_prior)} bins on {EVENT_DAY}")
    print(f"  N={N}  edges_with_endpoints={sum(1 for lid in links if str(lid) in link_endpoints)}")

    def compute_weights(a_s, a_l):
        w = np.ones(N, dtype=np.float64)
        if a_s != 0.0: w = w * np.power(speeds, a_s)
        if a_l != 0.0: w = w * np.power(lanes, a_l)
        return w

    # SINGLE-PHASE: 3 panels
    for K, opt in SP_OPTIMA.items():
        print(f"\n[single-phase top-{K} optimum]")
        w = compute_weights(opt["alpha_s"], opt["alpha_l"])
        P_cs = build_sp_P_cs(graph_data, links, lid_to_idx, w)
        v_avg = np.zeros(N, dtype=np.float64)
        for E in Es_prior:
            v = single_phase_iter(P_cs, E, opt["p"], TOL, MAX_ITER)
            v_avg += v
        v_avg /= len(Es_prior)
        top100 = np.argsort(v_avg)[::-1][:100]
        mask = np.zeros(N, dtype=bool); mask[top100] = True
        title = (f"Single-phase chain @ top-{K} optimum: "
                 f"p={opt['p']:.4f}, $\\alpha_s$={opt['alpha_s']:+.2f}, "
                 f"$\\alpha_l$={opt['alpha_l']:+.2f}")
        plot_top100(links, link_endpoints, nodes, mask,
                    OUT / f"sp_top100_at_top{K}_optimum.png", title)

    # TWO-PHASE: 3 panels
    for K, opt in TP_OPTIMA.items():
        print(f"\n[two-phase top-{K} optimum]")
        pagerank.PARAMS["alpha_s"] = opt["alpha_s"]
        pagerank.PARAMS["alpha_l"] = opt["alpha_l"]
        Pu, Pd = build_phase_kernels(graph_data)
        v_avg = np.zeros(N, dtype=np.float64)
        for E in Es_prior:
            vu, vd, _ = power_iteration(Pu, Pd, E, opt["beta"], opt["rho"],
                                            TOL, MAX_ITER)
            v = vu + vd; s = float(v.sum())
            if s > 0: v /= s
            v_avg += v
        v_avg /= len(Es_prior)
        top100 = np.argsort(v_avg)[::-1][:100]
        mask = np.zeros(N, dtype=bool); mask[top100] = True
        title = (f"Two-phase chain @ top-{K} optimum: "
                 f"$\\alpha_s$={opt['alpha_s']:+.2f}, "
                 f"$\\alpha_l$={opt['alpha_l']:+.2f}, "
                 f"$\\beta$={opt['beta']:.3f}, $\\rho$={opt['rho']:.4f}")
        plot_top100(links, link_endpoints, nodes, mask,
                    OUT / f"tp_top100_at_top{K}_optimum.png", title)

    print("\n[done] 6 images saved to", OUT)


if __name__ == "__main__":
    main()
