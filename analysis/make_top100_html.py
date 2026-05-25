"""Generate 6 self-contained HTML files, one per (chain, cluster)
optimum. Each draws the SLC network as inline SVG and makes the
top-100 chain-output links BLINK via CSS animation.
"""
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

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
        if float(np.abs(v_new - v).sum()) < tol: return v_new
        v = v_new
    return v


def bins_window(day_str):
    s = datetime.strptime(f"{day_str} {WINDOW_START}", "%Y-%m-%d %H:%M:%S")
    e = datetime.strptime(f"{day_str} {WINDOW_END}",   "%Y-%m-%d %H:%M:%S")
    out, cur = [], s
    while cur <= e:
        out.append(cur.strftime("%Y-%m-%d %H:%M:%S"))
        cur += timedelta(minutes=5)
    return out


def write_html(links, link_endpoints, nodes, top100_mask, out_path,
               title, subtitle, params_html, W_px=2200):
    # Compute projected coordinates: equirectangular, scaled to W_px.
    valid_links = [(i, lid) for i, lid in enumerate(links)
                   if str(lid) in link_endpoints
                   and link_endpoints[str(lid)][0] in nodes
                   and link_endpoints[str(lid)][1] in nodes]
    lons = []
    lats = []
    for i, lid in valid_links:
        ep = link_endpoints[str(lid)]
        a, b = nodes[ep[0]], nodes[ep[1]]
        lons.append(a[0]); lons.append(b[0])
        lats.append(a[1]); lats.append(b[1])
    lon_lo, lon_hi = min(lons), max(lons)
    lat_lo, lat_hi = min(lats), max(lats)
    cos_lat = np.cos(np.deg2rad(0.5 * (lat_lo + lat_hi)))
    dlon = (lon_hi - lon_lo) * cos_lat
    dlat = (lat_hi - lat_lo)
    H_px = int(round(W_px * dlat / dlon))

    def proj(lon, lat):
        x = (lon - lon_lo) * cos_lat / dlon * W_px
        y = (lat_hi - lat) / dlat * H_px
        return x, y

    # Separate bg / fg
    bg_lines = []
    fg_lines = []
    for i, lid in valid_links:
        ep = link_endpoints[str(lid)]
        a, b = nodes[ep[0]], nodes[ep[1]]
        x1, y1 = proj(a[0], a[1])
        x2, y2 = proj(b[0], b[1])
        seg = f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'
        if top100_mask[i]:
            fg_lines.append(seg)
        else:
            bg_lines.append(seg)
    star_x, star_y = proj(STADIUM_LON, STADIUM_LAT)

    css = """
    body { margin: 0; padding: 0.5rem 1rem 1rem;
           font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                        Helvetica, Arial, sans-serif;
           background: #fafafa; color: #222; }
    h1 { font-size: 1.15rem; margin: 0.4rem 0 0.1rem; }
    .subtitle { color: #555; font-size: 0.92rem; margin: 0 0 0.2rem; }
    .params { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              color: #444; font-size: 0.86rem; margin: 0 0 0.7rem; }
    svg { width: 100%; height: auto; background: white;
          border: 1px solid #ddd; border-radius: 4px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06); display: block; }
    g.bg line { stroke: #7f7f7f; stroke-width: 0.6;
                shape-rendering: geometricPrecision; }
    g.fg line { stroke: #0066cc; stroke-width: 2.0;
                stroke-linecap: round;
                animation: blink 1.0s ease-in-out infinite; }
    @keyframes blink {
        0%, 100% { opacity: 1.0; stroke-width: 2.0; }
        50%      { opacity: 0.15; stroke-width: 0.8; }
    }
    .star { fill: #f1c40f; stroke: black; stroke-width: 1.0; }
    """

    html = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<title>{title}</title><style>{css}</style></head><body>'
        f'<h1>{title}</h1>'
        f'<div class="subtitle">{subtitle}</div>'
        f'<div class="params">{params_html}</div>'
        f'<svg viewBox="0 0 {W_px} {H_px}" xmlns="http://www.w3.org/2000/svg">'
        f'<g class="bg">' + "".join(bg_lines) + "</g>"
        f'<g class="fg">' + "".join(fg_lines) + "</g>"
        f'<polygon class="star" '
        f'points="{star_x:.1f},{star_y-14:.1f} '
        f'{star_x+4:.1f},{star_y-4:.1f} '
        f'{star_x+14:.1f},{star_y-4:.1f} '
        f'{star_x+6:.1f},{star_y+3:.1f} '
        f'{star_x+9:.1f},{star_y+13:.1f} '
        f'{star_x:.1f},{star_y+6:.1f} '
        f'{star_x-9:.1f},{star_y+13:.1f} '
        f'{star_x-6:.1f},{star_y+3:.1f} '
        f'{star_x-14:.1f},{star_y-4:.1f} '
        f'{star_x-4:.1f},{star_y-4:.1f}"/>'
        f'</svg></body></html>'
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"  saved {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")


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
    proj_idx = np.array([lid_to_idx.get(lid, -1) for lid in pop_link_ids],
                          dtype=np.int64)
    valid = proj_idx >= 0
    time_index = {t: i for i, t in enumerate(times)}

    def get_E(time_str):
        ti = time_index.get(time_str)
        if ti is None: return None
        row = matrix.getrow(ti).toarray().ravel().astype(np.float64)
        C = np.zeros(N, dtype=np.float64)
        np.add.at(C, proj_idx[valid], row[valid])
        c = C + ALPHA
        c_diff = (1.0 - GAMMA) * c + GAMMA * (P_diff @ c)
        s = c_diff.sum()
        return c_diff / s if s > 0 else None

    Es = [E for E in (get_E(t) for t in bins_window(EVENT_DAY)) if E is not None]
    print(f"  N={N}  bins={len(Es)}")

    def weights(a_s, a_l):
        w = np.ones(N, dtype=np.float64)
        if a_s != 0.0: w = w * np.power(speeds, a_s)
        if a_l != 0.0: w = w * np.power(lanes, a_l)
        return w

    # Single-phase
    for K, opt in SP_OPTIMA.items():
        print(f"\n[SP top-{K}]")
        P_cs = build_sp_P_cs(graph_data, links, lid_to_idx,
                              weights(opt["alpha_s"], opt["alpha_l"]))
        v_avg = np.zeros(N)
        for E in Es:
            v_avg += single_phase_iter(P_cs, E, opt["p"], TOL, MAX_ITER)
        v_avg /= len(Es)
        top100 = np.argsort(v_avg)[::-1][:100]
        mask = np.zeros(N, dtype=bool); mask[top100] = True
        params_html = (f"p = {opt['p']:.5f} &middot; "
                       f"&alpha;<sub>s</sub> = {opt['alpha_s']:+.2f} &middot; "
                       f"&alpha;<sub>l</sub> = {opt['alpha_l']:+.2f}")
        write_html(links, link_endpoints, nodes, mask,
                    OUT / f"sp_top100_at_top{K}_optimum.html",
                    title=f"Single-phase chain &middot; top-{K} optimum",
                    subtitle="Top-100 highest-mass links of the chain "
                             "output on the Sept 8 19:00&ndash;19:55 prior "
                             "(blue, blinking). Network in grey, "
                             "Rice-Eccles Stadium marked with a yellow star.",
                    params_html=params_html)

    # Two-phase
    for K, opt in TP_OPTIMA.items():
        print(f"\n[TP top-{K}]")
        pagerank.PARAMS["alpha_s"] = opt["alpha_s"]
        pagerank.PARAMS["alpha_l"] = opt["alpha_l"]
        Pu, Pd = build_phase_kernels(graph_data)
        v_avg = np.zeros(N)
        for E in Es:
            vu, vd, _ = power_iteration(Pu, Pd, E, opt["beta"], opt["rho"],
                                              TOL, MAX_ITER)
            v = vu + vd; s = float(v.sum())
            if s > 0: v /= s
            v_avg += v
        v_avg /= len(Es)
        top100 = np.argsort(v_avg)[::-1][:100]
        mask = np.zeros(N, dtype=bool); mask[top100] = True
        params_html = (f"&alpha;<sub>s</sub> = {opt['alpha_s']:+.2f} &middot; "
                       f"&alpha;<sub>l</sub> = {opt['alpha_l']:+.2f} &middot; "
                       f"&beta; = {opt['beta']:.4f} &middot; "
                       f"&rho; = {opt['rho']:.4f}")
        write_html(links, link_endpoints, nodes, mask,
                    OUT / f"tp_top100_at_top{K}_optimum.html",
                    title=f"Two-phase chain &middot; top-{K} optimum",
                    subtitle="Top-100 highest-mass links of the chain "
                             "output on the Sept 8 19:00&ndash;19:55 prior "
                             "(blue, blinking). Network in grey, "
                             "Rice-Eccles Stadium marked with a yellow star.",
                    params_html=params_html)

    print("\n[done] 6 HTML files saved to", OUT)


if __name__ == "__main__":
    main()
