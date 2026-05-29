"""Held-out test of the trained origin->popularity ladder.

Test slots: 200 random 5-min slots from the last 7 days
[2018-09-23 19:00:00 .. 2018-09-30 18:55:00], seed=99.

We test only the rows that HAVE tuned parameters (drop the
uniform_prior and diffuse_prior baselines).

Trained alphas from the 12-row 840-slot ladder:

  chain    mode        a_s     a_l     a_ang
  SP       lanes_only  --      +0.744  --
  SP       speed_only  +2.528  --      --
  SP       both        +1.980  +0.475  --
  SP       ang_only    --      --      -6.252
  TP-new   lanes_only  --      +0.730  --
  TP-new   speed_only  +2.478  --      --
  TP-new   both        +1.943  +0.455  --
  TP-new   ang_only    --      --      -3.037

For each row: build P with the trained alphas, run the appropriate chain
(SP at alpha=0.0508; TP-new at beta=0.102, rho=0.101) on each of the
200 slots' diffused-origins input, compare to diffused-popularity target,
and report mean Ov.MRE.
"""
import json
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.io import load_popularity_npz

ORIGINS_NPZ = str(ROOT / "data/origins_results.npz")
SMOOTH_NPZ  = str(ROOT / "data/popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON  = str(ROOT / "data/city_graph_full.json")
NETWORK_XML = str(ROOT / "data/slc_network.xml")
OUT         = ROOT / "results" / "origin_pop_ladder" / "test_last7days_200.json"

EPS = 1e-6
EPS_ANG = 0.05
TEST_SEED = 99
N_TEST = 200
TOL, MAX_ITER = 1e-5, 200
SP_ALPHA   = 0.0508
TP_BETA    = 0.102
TP_RHO     = 0.101
LAPLACE_ALPHA = 0.01
GAMMA      = 0.20

TEST_START = "2018-09-23 19:00:00"
TEST_END   = "2018-09-30 18:55:00"

# Each row: (chain, label, alpha_s, alpha_l, alpha_ang)
ROWS = [
    ("sp",     "lanes_only",  0.0,    0.744,  0.0),
    ("sp",     "speed_only",  2.528,  0.0,    0.0),
    ("sp",     "both",        1.980,  0.475,  0.0),
    ("sp",     "ang_only",    0.0,    0.0,   -6.252),
    ("tp_new", "lanes_only",  0.0,    0.730,  0.0),
    ("tp_new", "speed_only",  2.478,  0.0,    0.0),
    ("tp_new", "both",        1.943,  0.455,  0.0),
    ("tp_new", "ang_only",    0.0,    0.0,   -3.037),
]


def get_speed_lane(graph, links):
    s, l = [], []
    for lid in links:
        ed = graph["links"].get(lid, {})
        s.append(ed.get("speed", 11.17))
        l.append(ed.get("lanes", 1.0))
    s = np.array(s, dtype=np.float64); l = np.array(l, dtype=np.float64)
    if s.mean() > 0: s /= s.mean()
    if l.mean() > 0: l /= l.mean()
    return s, l


def parse_link_dirs(xml_path):
    nodes = {}; link_dir = {}
    for ev, el in ET.iterparse(xml_path, events=("start",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("x")), float(el.get("y")))
        elif el.tag == "link":
            f, t = el.get("from"), el.get("to")
            if f in nodes and t in nodes:
                x1, y1 = nodes[f]; x2, y2 = nodes[t]
                dx, dy = x2 - x1, y2 - y1
                norm = (dx*dx + dy*dy) ** 0.5
                if norm > 0:
                    link_dir[el.get("id")] = (dx / norm, dy / norm)
            el.clear()
    return link_dir


def build_per_source_arrays(graph, links, lid_to_idx, link_dir):
    adj = graph.get("adjacency", {})
    succ_idx = [None] * len(links)
    cos_th   = [None] * len(links)
    for i, lid in enumerate(links):
        succ_lids = [ol for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ_lids:
            succ_idx[i] = np.empty(0, dtype=np.int64)
            cos_th[i]   = np.empty(0, dtype=np.float64)
            continue
        idxs = np.array([lid_to_idx[ol] for ol in succ_lids], dtype=np.int64)
        v_i = link_dir.get(lid)
        if v_i is None:
            cs = np.zeros(len(succ_lids), dtype=np.float64)
        else:
            cs = np.empty(len(succ_lids), dtype=np.float64)
            for k, ol in enumerate(succ_lids):
                v_j = link_dir.get(ol)
                if v_j is None:
                    cs[k] = 0.0
                else:
                    c = v_i[0]*v_j[0] + v_i[1]*v_j[1]
                    cs[k] = max(-1.0, min(1.0, c))
        succ_idx[i] = idxs
        cos_th[i]   = cs
    return succ_idx, cos_th


def build_P_weighted(N, succ_idx, cos_th, dest_w, a_ang, eps_ang=EPS_ANG):
    rows, cols, data = [], [], []
    for i in range(N):
        idxs = succ_idx[i]
        if idxs.size == 0:
            continue
        out_w = dest_w[idxs].copy()
        if a_ang != 0.0:
            out_w = out_w * np.power(1.0 + cos_th[i] + eps_ang, a_ang)
        total = out_w.sum()
        if total <= 0:
            continue
        norm = out_w / total
        for k, j in enumerate(idxs):
            rows.append(int(j)); cols.append(i); data.append(float(norm[k]))
    return csr_matrix((data, (rows, cols)),
                       shape=(N, N), dtype=np.float64)


def build_P_diffusion(graph, links, lid_to_idx):
    adj = graph.get("adjacency", {})
    rows, cols, data = [], [], []
    for lid, out_links in adj.items():
        if lid not in lid_to_idx: continue
        i = lid_to_idx[lid]
        valid = [lid_to_idx[ol] for ol in out_links if ol in lid_to_idx]
        if not valid: continue
        w = 1.0 / len(valid)
        for j in valid:
            rows.append(i); cols.append(j); data.append(w)
    return csr_matrix((data, (rows, cols)),
                       shape=(len(links), len(links)), dtype=np.float64)


def sp_iter(P_cs, prior, alpha, tol=TOL, max_iter=MAX_ITER):
    prior_f = prior.astype(np.float64, copy=True)
    v = prior_f.copy()
    one_a = 1.0 - alpha
    for _ in range(max_iter):
        v_new = alpha * prior_f + one_a * (P_cs @ v)
        s = v_new.sum()
        if s > 0: v_new /= s
        if float(np.abs(v_new - v).sum()) < tol:
            return v_new
        v = v_new
    return v


def tp_new_iter(P_cs, u_in, beta, rho, tol=TOL, max_iter=MAX_ITER):
    u = u_in.astype(np.float64, copy=True)
    N = u.shape[0]
    v_up = u.copy(); v_down = np.zeros(N)
    for _ in range(max_iter):
        s_down = float(v_down.sum())
        v_up_new   = (1.0 - beta) * (P_cs @ v_up) + rho * u * s_down
        v_down_new = beta * v_up + (1.0 - rho) * (P_cs @ v_down)
        s = float(v_up_new.sum() + v_down_new.sum())
        if s > 0: v_up_new /= s; v_down_new /= s
        diff = float(np.abs(v_up_new - v_up).sum() + np.abs(v_down_new - v_down).sum())
        v_up, v_down = v_up_new, v_down_new
        if diff < tol: break
    return v_up + v_down


def load_pop_for(npz_path, N, lid_to_idx, slot_set):
    b = load_popularity_npz(npz_path)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_ids = [str(lid) for lid in b["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_ids], dtype=np.int64)
    valid = proj >= 0
    rows = {}
    for ti, t in enumerate(times):
        if t not in slot_set: continue
        row = M.getrow(ti).toarray().ravel().astype(np.float64)
        E = np.zeros(N, dtype=np.float64)
        np.add.at(E, proj[valid], row[valid])
        s = E.sum()
        if s > 0: E /= s
        rows[t] = E
    return rows


def diffuse_origin_slot(raw_row, raw_proj, raw_valid, N, P_diff):
    c = np.full(N, LAPLACE_ALPHA, dtype=np.float64)
    if raw_row is not None:
        c_add = np.zeros(N, dtype=np.float64)
        np.add.at(c_add, raw_proj[raw_valid], raw_row[raw_valid])
        c = c + c_add
    c_diff = (1.0 - GAMMA) * c + GAMMA * (P_diff @ c)
    s = c_diff.sum()
    if s > 0: c_diff /= s
    return c_diff


def load_origins_for(npz_path, N, lid_to_idx, slot_set, P_diff):
    b = load_popularity_npz(npz_path)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    orig_ids = [str(lid) for lid in b["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in orig_ids], dtype=np.int64)
    valid = proj >= 0
    time_idx = {t: i for i, t in enumerate(times)}
    rows = {}
    for t in slot_set:
        ti = time_idx.get(t)
        if ti is None:
            rows[t] = diffuse_origin_slot(None, proj, valid, N, P_diff)
            continue
        raw = M.getrow(ti).toarray().ravel().astype(np.float64)
        rows[t] = diffuse_origin_slot(raw, proj, valid, N, P_diff)
    return rows


def main():
    print("[load] graph + features ...")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    speeds, lanes = get_speed_lane(graph, links)
    P_diff = build_P_diffusion(graph, links, lid_to_idx)
    link_dir = parse_link_dirs(NETWORK_XML)
    succ_idx, cos_th = build_per_source_arrays(graph, links, lid_to_idx, link_dir)
    print(f"  N={N:,}  link_dirs={len(link_dir):,}")

    b = load_popularity_npz(SMOOTH_NPZ)
    all_times = [str(t) for t in b["times"]]
    del b
    pool = [t for t in all_times if TEST_START <= t <= TEST_END]
    print(f"  last-7-days candidate slots: {len(pool)}")
    rng = random.Random(TEST_SEED)
    test_slots = sorted(rng.sample(pool, N_TEST))
    print(f"  test slots: {len(test_slots)}, {test_slots[0]} ... {test_slots[-1]}")

    slot_set = set(test_slots)
    print("[load] diffused popularity (targets) + diffused origins (inputs) ...")
    targets_d = load_pop_for(SMOOTH_NPZ, N, lid_to_idx, slot_set)
    inputs_d  = load_origins_for(ORIGINS_NPZ, N, lid_to_idx, slot_set, P_diff)
    inputs  = [inputs_d[t]  for t in test_slots]
    targets = [targets_d[t] for t in test_slots]

    print(f"\n[eval] {len(ROWS)} (chain, mode) rows on {N_TEST} test slots\n")
    print(f"  {'chain':<7} {'mode':<11} {'a_s':>7} {'a_l':>7} {'a_ang':>7}  "
          f"{'test MRE mean':>13} {'test MRE median':>15}")
    out = []
    for chain, mode, a_s, a_l, a_ang in ROWS:
        dest_w = np.ones(N, dtype=np.float64)
        if a_s != 0: dest_w = dest_w * np.power(speeds, a_s)
        if a_l != 0: dest_w = dest_w * np.power(lanes, a_l)
        P = build_P_weighted(N, succ_idx, cos_th, dest_w, a_ang)
        ov = []
        for u, tgt in zip(inputs, targets):
            v = sp_iter(P, u, SP_ALPHA) if chain == "sp" \
                else tp_new_iter(P, u, TP_BETA, TP_RHO)
            ov.append(float(np.mean(np.abs(tgt - v) / (tgt + EPS))))
        mean = float(np.mean(ov)); med = float(np.median(ov))
        print(f"  {chain:<7} {mode:<11} {a_s:+7.3f} {a_l:+7.3f} {a_ang:+7.3f}  "
              f"{mean:13.4f} {med:15.4f}")
        out.append(dict(chain=chain, mode=mode,
                         alpha_s=a_s, alpha_l=a_l, alpha_ang=a_ang,
                         test_mre_mean=mean, test_mre_median=med,
                         per_slot=ov))

    OUT.write_text(json.dumps(dict(
        n_test=N_TEST, test_seed=TEST_SEED,
        test_range=[TEST_START, TEST_END],
        test_slots=test_slots, rows=out,
    ), indent=2))
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
