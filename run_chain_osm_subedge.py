"""Run the two-phase chain at OSM sub-edge granularity, computing the
EVALUATION_CONTRACT three-comparison MRE table.

No diffusion smoothing in this prototype: at 1.63 M sub-edges a dense
T x N smoothed matrix would be ~56 GB. We use the raw OSM popularity
prior. ε = 1e-6 in the MRE denominator (matches the table's
"No diffusion ε=10^-6" column).

Inputs:
    data/network_osm.pkl
    data/popularity_results_osm.npz
    data/matched_routes_osm.json   -- only needed to derive popularity at OSM level
"""
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy import sparse

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("TPPR_DATA") or os.path.join(ROOT, "data")
NETWORK_PKL = os.environ.get("TPPR_NETWORK_PKL", os.path.join(DATA, "network_osm.pkl"))
RAW_POP_NPZ = os.environ.get("TPPR_RAW_POP", os.path.join(DATA, "popularity_results_osm.npz"))

BETA = 0.027   # from OSM-sub-edge phase split (E[L_up] = 35.79)
RHO  = 0.023
TOL = 1e-7
MAX_ITERS = 300
EPS_MRE = 1e-6
NUM_FRAMES = 49
TOP_K = 100
SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- Build OSM-level graph in memory -------------------------------------

def build_osm_graph():
    log(f"Loading {NETWORK_PKL} ...")
    with open(NETWORK_PKL, "rb") as f:
        net = pickle.load(f)
    proj = net["proj_nodes"]
    edges = net["edges"]
    e2m = net["edge_to_matsim"]
    log(f"  {len(proj):,} nodes, {len(edges):,} directed sub-edges")

    log("Indexing sub-edges ...")
    edge_idx = {e: i for i, e in enumerate(edges)}
    N = len(edges)

    log("Building adjacency: each (a,b) -> all (b,c) ...")
    out_from_node = defaultdict(list)
    for i, (a, b) in enumerate(edges):
        out_from_node[a].append(i)
    adj = [None] * N
    n_adj_total = 0
    for i, (a, b) in enumerate(edges):
        succ = out_from_node.get(b, [])
        adj[i] = succ
        n_adj_total += len(succ)
    log(f"  total adjacency entries: {n_adj_total:,}")

    log("Computing per-edge lengths from projected coords ...")
    lengths = np.empty(N, dtype=np.float32)
    for i, (a, b) in enumerate(edges):
        ax, ay = proj[a]; bx, by = proj[b]
        lengths[i] = math.hypot(ax - bx, ay - by)
    log(f"  mean length: {lengths.mean():.1f} m, median: {np.median(lengths):.1f} m")

    return {
        "edges": edges,
        "edge_idx": edge_idx,
        "edge_to_matsim": e2m,
        "adj": adj,
        "lengths": lengths,
        "N": N,
    }


def build_phase_matrices(graph):
    """Uniform-over-neighbours row-stochastic P with self-loop dwell.
    With alpha_s = alpha_l = 0 (default), up and down phases have the
    same matrix.
    """
    N = graph["N"]
    adj = graph["adj"]
    lengths = graph["lengths"]

    log("Building row-stochastic P ...")
    row, col, data = [], [], []
    for i in range(N):
        succ = adj[i]
        if not succ:
            continue
        w_each = 1.0 / len(succ)
        for j in succ:
            row.append(i); col.append(j); data.append(w_each)
    P = sparse.csr_matrix((data, (row, col)), shape=(N, N), dtype=np.float64)
    log(f"  P shape={P.shape}, nnz={P.nnz:,}")
    return P


# --- Build OSM-level popularity matrix from MATSim raw -------------------

def build_osm_popularity(graph):
    log(f"Loading {RAW_POP_NPZ} ...")
    raw = np.load(RAW_POP_NPZ, allow_pickle=True)
    M = sparse.csr_matrix(
        (raw["matrix_data"], raw["matrix_indices"], raw["matrix_indptr"]),
        shape=raw["matrix_shape"]
    )
    times = list(raw["times"])
    matsim_link_ids = list(raw["link_ids"])
    log(f"  raw shape={M.shape}, nnz={M.nnz:,}, T={len(times)}")

    # MATSim-id -> list of OSM sub-edge indices
    log("Inverting OSM->MATSim mapping ...")
    matsim_to_osm = defaultdict(list)
    for (a, b), m_id in graph["edge_to_matsim"].items():
        matsim_to_osm[m_id].append(graph["edge_idx"][(a, b)])
    log(f"  {len(matsim_to_osm):,} matsim links have OSM expansion")

    # Build expanded T x N_osm sparse popularity by column expansion
    log("Expanding columns: each MATSim col -> all OSM sub-edges of its chain")
    T = M.shape[0]
    N = graph["N"]
    coo = M.tocoo()
    # For each (t, matsim_idx, val), expand to (t, osm_idx, val) for each osm child
    new_row, new_col, new_data = [], [], []
    n_dropped = 0
    for r, c, v in zip(coo.row, coo.col, coo.data):
        m_id = matsim_link_ids[c]
        children = matsim_to_osm.get(m_id, [])
        if not children:
            n_dropped += 1
            continue
        for osm_idx in children:
            new_row.append(r); new_col.append(osm_idx); new_data.append(v)
    log(f"  dropped {n_dropped:,} matsim cells with no OSM mapping")
    M_osm = sparse.csr_matrix(
        (np.asarray(new_data, dtype=np.float32),
         (np.asarray(new_row, dtype=np.int64),
          np.asarray(new_col, dtype=np.int64))),
        shape=(T, N),
    )
    M_osm.sum_duplicates()
    log(f"  expanded shape={M_osm.shape}, nnz={M_osm.nnz:,}")
    return M_osm, times


# --- Chain iteration (V1 single-matrix two-phase) ------------------------

def chain_iter(P_T, E_b):
    """V1 single-matrix two-phase iteration. P_T: column-stochastic
    (= row-stochastic transposed) sparse. Returns v_total = v_up + v_down."""
    N = E_b.shape[0]
    v_up = E_b.copy()
    v_down = np.zeros(N, dtype=np.float64)
    for _ in range(MAX_ITERS):
        s_down = float(v_down.sum())
        v_up_new = (1.0 - BETA) * (P_T @ v_up) + RHO * E_b * s_down
        v_down_new = BETA * v_up + (1.0 - RHO) * (P_T @ v_down)
        s = float(v_up_new.sum() + v_down_new.sum())
        if s > 0:
            v_up_new /= s
            v_down_new /= s
        if (np.abs(v_up_new - v_up).sum() + np.abs(v_down_new - v_down).sum()) < TOL:
            v_up, v_down = v_up_new, v_down_new
            break
        v_up, v_down = v_up_new, v_down_new
    v_total = v_up + v_down
    s = v_total.sum()
    if s > 0:
        v_total /= s
    return v_total


# --- Metrics + sampling --------------------------------------------------

def mre(truth, pred, top_k=TOP_K):
    rel = np.abs(truth - pred) / (truth + EPS_MRE)
    overall = float(rel.mean())
    top = np.argsort(truth)[::-1][:top_k]
    top_mre = float(rel[top].mean())
    return overall, top_mre


def parse_dt(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def find_siblings(times, target_dt, max_per=8):
    out = []
    for t in times:
        dt = parse_dt(t)
        if dt == target_dt:
            continue
        if (dt.weekday() == target_dt.weekday()
                and dt.hour == target_dt.hour
                and dt.minute == target_dt.minute):
            out.append(t)
            if len(out) >= max_per:
                break
    return out


def extract_E(M, t_idx):
    row = M.getrow(t_idx).toarray().flatten().astype(np.float64)
    s = row.sum()
    if s > 0:
        row /= s
    return row


# --- Main ----------------------------------------------------------------

def main():
    t0 = time.time()
    graph = build_osm_graph()
    P = build_phase_matrices(graph)
    P_T = P.T.tocsr()
    log(f"Phase-matrix construction: {time.time()-t0:.0f}s")

    M_osm, times = build_osm_popularity(graph)
    log(f"Popularity build: {time.time()-t0:.0f}s")

    import random
    random.seed(SEED)
    sample_times = random.sample(times, min(NUM_FRAMES, len(times)))

    naive_top, naive_ovr = [], []
    forecast_top, forecast_ovr = [], []
    reconstruct_top, reconstruct_ovr = [], []

    log(f"Starting eval over {len(sample_times)} frames ...")
    log(f"  beta={BETA}, rho={RHO}, eps_mre={EPS_MRE}")
    log(f"  N_links={graph['N']:,}")

    for k, t_str in enumerate(sample_times):
        t_idx = times.index(t_str)
        E_t = extract_E(M_osm, t_idx)
        target_dt = parse_dt(t_str)
        sibs = find_siblings(times, target_dt, max_per=8)
        if not sibs:
            continue

        # Reconstruct: model started from E_t vs E_t
        v = chain_iter(P_T, E_t)
        ovr, top = mre(E_t, v)
        reconstruct_top.append(top); reconstruct_ovr.append(ovr)

        # Per-sibling forecast and naive
        for sib in sibs:
            s_idx = times.index(sib)
            E_sib = extract_E(M_osm, s_idx)
            ovr_n, top_n = mre(E_t, E_sib)
            naive_top.append(top_n); naive_ovr.append(ovr_n)
            v_sib = chain_iter(P_T, E_sib)
            ovr_f, top_f = mre(E_t, v_sib)
            forecast_top.append(top_f); forecast_ovr.append(ovr_f)
        log(f"  {k+1}/{len(sample_times)} {t_str}: rec_top={top:.4f} rec_ovr={ovr:.4f}")

    print()
    log(f"Total wall-clock: {time.time()-t0:.0f}s")
    print()
    print("=" * 75)
    print(f"  AGGREGATE (OSM sub-edge granularity, no diffusion, eps={EPS_MRE})")
    print("=" * 75)
    print(f"  {'comparison':35s}  {'Top-100':>10s}  {'Overall':>10s}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}")
    print(f"  {'1. dataset->dataset (naive)':35s}  "
          f"{np.mean(naive_top):10.4f}  {np.mean(naive_ovr):10.4f}")
    print(f"  {'2. dataset->model (forecast)':35s}  "
          f"{np.mean(forecast_top):10.4f}  {np.mean(forecast_ovr):10.4f}")
    print(f"  {'3. dataset->model (reconstruct)':35s}  "
          f"{np.mean(reconstruct_top):10.4f}  {np.mean(reconstruct_ovr):10.4f}")
    print()


if __name__ == "__main__":
    main()
