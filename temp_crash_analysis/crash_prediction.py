#!/usr/bin/env python3
"""
Crash PREDICTION using demand-weighted model (d=0.96, gamma=0.20).

NO crash-day ground truth used. Instead:
  - Input: Sept 13 (normal Thursday) teleportation
  - M_normal: demand-weighted matrix (no crash conditions)
  - M_crash: same matrix but crash-site links have reduced speed/lanes
  - Compare predictions to crash-day truth

This tests whether modifying the transition matrix alone can predict
the crash's effect — now feasible because the matrix carries 96% of
the prediction weight.
"""

import sys, json, copy, time as _time
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, hstack, vstack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config
from core.pagerank import compute_speed_lane_weights, PARAMS
from core.gpu_backend import power_iteration, BACKEND_NAME
import logging; logging.basicConfig(level=logging.WARNING)

OUT_DIR = Path(__file__).resolve().parent

CRASH_LAT, CRASH_LON = 40.524, -111.893
CRASH_DATE = '2018-09-20'
INPUT_DATE = '2018-09-13'   # Normal Thursday — NO crash-day data used
SNAPSHOT_TIMES = ['06:00', '07:30', '08:30', '09:30', '10:30', '11:30']
CRASH_WINDOW = {'07:30', '08:30', '09:30', '10:30'}

# Crash conditioning: link attribute modifications
CRASH_SITE_SPEED = 2.0
CRASH_SITE_LANES = 1.0
UPSTREAM_SPD_FAC = 0.30
UPSTREAM_LN_FAC  = 0.50
NB_SPD_FAC       = 0.70

GAMMA = 0.20
DAMPING = 0.96

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'figure.dpi': 200})


# ── Data loading ─────────────────────────────────────────────────

def parse_network():
    tree = ET.parse(str(config.NETWORK_XML))
    root = tree.getroot()
    nodes = {}
    for n in root.find('nodes'):
        nodes[n.get('id')] = (float(n.get('y')), float(n.get('x')))
    links = {}
    for l in root.find('links'):
        links[l.get('id')] = {
            'from': l.get('from'), 'to': l.get('to'),
            'length': float(l.get('length')),
            'freespeed': float(l.get('freespeed')),
            'lanes': float(l.get('permlanes')),
        }
    return nodes, links


def classify_links(nodes, xml_links):
    crash_site_sb, upstream_sb = [], []
    all_sb, all_nb, surface = [], [], []
    for lid, ld in xml_links.items():
        fn, tn = ld['from'], ld['to']
        if fn not in nodes or tn not in nodes: continue
        lat1, lon1 = nodes[fn]; lat2, lon2 = nodes[tn]
        mid_lat = (lat1+lat2)/2; mid_lon = (lon1+lon2)/2
        dist = ((mid_lat-CRASH_LAT)**2+(mid_lon-CRASH_LON)**2)**0.5
        if ld['freespeed'] >= 25 and dist < 0.06:
            is_sb = lat2 < lat1
            if is_sb:
                all_sb.append(lid)
                if dist < 0.025:
                    crash_site_sb.append(lid)
                elif mid_lat > CRASH_LAT:
                    upstream_sb.append(lid)
            else:
                all_nb.append(lid)
        elif dist < 0.04 and 11 <= ld['freespeed'] < 25 and ld['lanes'] >= 2:
            surface.append(lid)
    return crash_site_sb, upstream_sb, all_sb, all_nb, surface


def load_pop():
    loader = np.load(str(config.POPULARITY_NPZ), allow_pickle=True)
    mat = csr_matrix((loader['matrix_data'], loader['matrix_indices'],
                      loader['matrix_indptr']), shape=loader['matrix_shape'])
    return mat, list(loader['times']), list(loader['link_ids'])


def compute_demand(mat, times, pop_ids, lid_to_idx, N):
    train_idx = [i for i, ts in enumerate(times)
                 if datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").day <= 15]
    dem = np.zeros(N)
    for ti in train_idx:
        row = mat.getrow(ti)
        for i, val in zip(row.indices, row.data):
            lid = pop_ids[i]
            if lid in lid_to_idx: dem[lid_to_idx[lid]] += float(val)
    s = dem.sum()
    return dem / s if s > 0 else np.ones(N) / N


def extract_E(mat, times, pop_ids, lid_to_idx, N, ts):
    if ts not in times: return np.ones(N) / N
    ti = times.index(ts)
    row = mat.getrow(ti)
    E = np.zeros(N)
    for i, val in zip(row.indices, row.data):
        lid = pop_ids[i]
        if lid in lid_to_idx: E[lid_to_idx[lid]] = float(val)
    s = E.sum()
    return E / s if s > 0 else np.ones(N) / N


# ── Matrix builders ──────────────────────────────────────────────

def build_demand_2phase(gd, demand, gamma):
    links = list(gd['links'].keys()); N = len(links)
    id_to_idx = {lid: i for i, lid in enumerate(links)}
    mu = PARAMS['mu']; beta = PARAMS['beta']
    adj = gd.get('adjacency', {})
    up_w = compute_speed_lane_weights(gd, True)
    down_w = compute_speed_lane_weights(gd, False)
    dfac = np.power(demand + 1e-15, gamma) if gamma != 0 else np.ones(N)

    def bpm(weights):
        r, c, d = [], [], []
        for i, lid in enumerate(links):
            outs = adj.get(lid, [])
            succ = [id_to_idx[s] for s in outs if s in id_to_idx]
            ow = [weights[j] * dfac[j] for j in succ]
            ed = gd['links'][lid]
            sw = mu * ed.get('length', 100) / max(ed.get('speed', 11.17), 0.1)
            tot = sum(ow) + sw
            if tot > 0:
                if sw > 0: r.append(i); c.append(i); d.append(sw/tot)
                for j, w in zip(succ, ow): r.append(i); c.append(j); d.append(w/tot)
        return csr_matrix((d, (r, c)), shape=(N, N))

    Pu = bpm(up_w); Pd = bpm(down_w)
    Pu2 = Pu.multiply(1-beta)
    Td = csr_matrix((np.ones(N)*beta, (np.arange(N), np.arange(N))), shape=(N, N))
    Z = csr_matrix((N, N))
    return vstack([hstack([Pu2, Td]), hstack([Z, Pd])])


def build_crash_graph(gd, crash_ids, upstream_ids, nb_ids):
    gd2 = copy.deepcopy(gd)
    for lid in crash_ids:
        if lid in gd2['links']:
            gd2['links'][lid]['speed'] = CRASH_SITE_SPEED
            gd2['links'][lid]['lanes'] = CRASH_SITE_LANES
    for lid in upstream_ids:
        if lid in gd2['links']:
            gd2['links'][lid]['speed'] *= UPSTREAM_SPD_FAC
            gd2['links'][lid]['lanes'] = max(1.0, gd2['links'][lid].get('lanes', 1) * UPSTREAM_LN_FAC)
    for lid in nb_ids:
        if lid in gd2['links']:
            gd2['links'][lid]['speed'] *= NB_SPD_FAC
    return gd2


def run_pr(M, E_N, N):
    E_2N = np.concatenate([E_N, np.zeros(N)])
    v_2N = power_iteration(M, E_2N, damping=DAMPING)
    v = v_2N[:N] + v_2N[N:]
    v /= v.sum()
    return v


# ── Main ─────────────────────────────────────────────────────────

def main():
    t0 = _time.time()
    print("=" * 70)
    print(f"CRASH PREDICTION (d={DAMPING}, gamma={GAMMA}, backend={BACKEND_NAME})")
    print(f"Input: {INPUT_DATE} (normal Thursday) — NO crash-day data used")
    print("=" * 70)

    nodes, xml_links = parse_network()
    crash_ids, upstream_ids, all_sb, all_nb, surface = classify_links(nodes, xml_links)
    print(f"  Crash-site: {len(crash_ids)}, Upstream: {len(upstream_ids)}, "
          f"SB: {len(all_sb)}, NB: {len(all_nb)}, Surface: {len(surface)}")

    with open(config.GRAPH_FILE) as f:
        gd = json.load(f)
    link_list = list(gd['links'].keys()); N = len(link_list)
    lid_to_idx = {lid: i for i, lid in enumerate(link_list)}

    mat, times, pop_ids = load_pop()
    demand = compute_demand(mat, times, pop_ids, lid_to_idx, N)

    corridors = {
        'I-15 SB': [lid_to_idx[l] for l in all_sb if l in lid_to_idx],
        'I-15 NB': [lid_to_idx[l] for l in all_nb if l in lid_to_idx],
        'Surface': [lid_to_idx[l] for l in surface if l in lid_to_idx],
    }

    # Build matrices
    print("\n  Building M_normal...")
    M_normal = build_demand_2phase(gd, demand, GAMMA)

    print("  Building M_crash (modified crash links)...")
    gd_crash = build_crash_graph(gd, crash_ids, upstream_ids, all_nb)
    M_crash = build_demand_2phase(gd_crash, demand, GAMMA)

    # Run predictions
    print("\n  Running predictions...")
    print(f"  {'Time':>6s} {'Corr':>10s} | {'Truth':>10s} {'Normal':>10s} "
          f"{'Crash':>10s} | {'Err_N':>7s} {'Err_C':>7s} {'Improv':>8s}")
    print(f"  {'─'*78}")

    results = []
    for st in SNAPSHOT_TIMES:
        crash_ts = f"{CRASH_DATE} {st}:00"
        input_ts = f"{INPUT_DATE} {st}:00"

        E_truth = extract_E(mat, times, pop_ids, lid_to_idx, N, crash_ts)
        E_input = extract_E(mat, times, pop_ids, lid_to_idx, N, input_ts)

        v_truth = run_pr(M_normal, E_truth, N)
        v_normal = run_pr(M_normal, E_input, N)
        v_crash = run_pr(M_crash, E_input, N)

        for cn, cidx in corridors.items():
            tm = sum(v_truth[i] for i in cidx)
            nm = sum(v_normal[i] for i in cidx)
            cm = sum(v_crash[i] for i in cidx)

            err_n = abs(nm - tm) / (tm + 1e-15) * 100
            err_c = abs(cm - tm) / (tm + 1e-15) * 100
            improv = (err_n - err_c) / err_n * 100 if err_n > 0 else 0

            results.append({
                'time': st, 'corridor': cn,
                'truth': tm, 'normal': nm, 'crash': cm,
                'err_normal': err_n, 'err_crash': err_c,
                'improvement': improv,
                'in_window': st in CRASH_WINDOW,
            })

            tag = ' ***' if improv > 5 else ''
            print(f"  {st:>6s} {cn:>10s} | {tm:10.6f} {nm:10.6f} "
                  f"{cm:10.6f} | {err_n:6.1f}% {err_c:6.1f}% {improv:+7.1f}%{tag}")

    # Crash-window averages
    print(f"\n  CRASH-WINDOW AVERAGES:")
    print(f"  {'Corr':>10s}  {'Err Normal':>12s}  {'Err Crash':>12s}  {'Improvement':>12s}")
    cw_summary = {}
    for cn in ['I-15 SB', 'I-15 NB', 'Surface']:
        cw = [r for r in results if r['corridor'] == cn and r['in_window']]
        en = np.mean([r['err_normal'] for r in cw])
        ec = np.mean([r['err_crash'] for r in cw])
        imp = (en - ec) / en * 100 if en > 0 else 0
        cw_summary[cn] = {'err_normal': en, 'err_crash': ec, 'improvement': imp}
        print(f"  {cn:>10s}  {en:11.1f}%  {ec:11.1f}%  {imp:+11.1f}%")

    # ── Figures ──────────────────────────────────────────────────

    # Fig 1: Error comparison over time
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, cn in zip(axes, ['I-15 SB', 'I-15 NB', 'Surface']):
        cd = [r for r in results if r['corridor'] == cn]
        ts = [r['time'] for r in cd]
        en = [r['err_normal'] for r in cd]
        ec = [r['err_crash'] for r in cd]
        x = np.arange(len(ts))
        ax.plot(x, en, '-o', color='#d62728', lw=2, ms=8, label='Normal matrix (no crash)')
        ax.plot(x, ec, '-s', color='#2ca02c', lw=2, ms=8, label='Crash-conditioned matrix')
        cw = [i for i, r in enumerate(cd) if r['in_window']]
        if cw: ax.axvspan(min(cw)-.4, max(cw)+.4, alpha=.08, color='red', label='Crash window')
        ax.set_xticks(x); ax.set_xticklabels(ts, rotation=45)
        ax.set_ylabel('Prediction Error (%)'); ax.set_title(cn, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=.3)
    plt.suptitle(f'Crash Prediction Error: Normal vs Crash-Conditioned Matrix\n'
                 f'(d={DAMPING}, γ={GAMMA}, input=Sept 13, NO crash-day data)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'fig_crash_pred_error.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Fig 2: Corridor mass comparison
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, cn in zip(axes, ['I-15 SB', 'I-15 NB', 'Surface']):
        cd = [r for r in results if r['corridor'] == cn]
        ts = [r['time'] for r in cd]
        x = np.arange(len(ts)); w = 0.25
        ax.bar(x-w, [r['truth']*1e4 for r in cd], w, color='steelblue', alpha=.85, label='Truth (Sept 20)')
        ax.bar(x, [r['normal']*1e4 for r in cd], w, color='#d62728', alpha=.85, label='Normal prediction')
        ax.bar(x+w, [r['crash']*1e4 for r in cd], w, color='#2ca02c', alpha=.85, label='Crash prediction')
        cw = [i for i, r in enumerate(cd) if r['in_window']]
        if cw: ax.axvspan(min(cw)-.4, max(cw)+.4, alpha=.06, color='red')
        ax.set_xticks(x); ax.set_xticklabels(ts, rotation=45)
        ax.set_ylabel('PageRank Mass (×10⁻⁴)'); ax.set_title(cn, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=.3, axis='y')
    plt.suptitle('Corridor Mass: Truth vs Normal vs Crash-Conditioned Prediction',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'fig_crash_pred_mass.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Fig 3: Improvement summary
    fig, ax = plt.subplots(figsize=(10, 6))
    cns = ['I-15 SB', 'I-15 NB', 'Surface']
    x = np.arange(len(cns)); w = 0.35
    ax.bar(x-w/2, [cw_summary[c]['err_normal'] for c in cns], w,
           color='#d62728', alpha=.85, label='Normal matrix error')
    ax.bar(x+w/2, [cw_summary[c]['err_crash'] for c in cns], w,
           color='#2ca02c', alpha=.85, label='Crash-conditioned error')
    for i, cn in enumerate(cns):
        imp = cw_summary[cn]['improvement']
        y = max(cw_summary[cn]['err_normal'], cw_summary[cn]['err_crash'])
        ax.annotate(f'{imp:+.1f}%', (i, y+1), ha='center', fontweight='bold', fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(cns)
    ax.set_ylabel('Mean Error During Crash Window (%)')
    ax.set_title(f'Crash-Window Average Error (d={DAMPING}, γ={GAMMA})', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=.3, axis='y')
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / 'fig_crash_pred_summary.png'), dpi=200)
    plt.close()

    print(f"\n  Figures saved.")

    # Save results
    with open(OUT_DIR / 'crash_prediction_results.json', 'w') as f:
        json.dump({'results': results, 'summary': cw_summary,
                   'config': {'damping': DAMPING, 'gamma': GAMMA,
                              'input_date': INPUT_DATE, 'crash_date': CRASH_DATE}},
                  f, indent=2)

    total = _time.time() - t0
    print(f"  Total: {total:.1f}s | Backend: {BACKEND_NAME}")


if __name__ == '__main__':
    main()
