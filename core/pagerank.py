"""
Two-phase PageRank chain — implementation of the model defined in
`documents/walkthrough2/main.tex`, §7.

Column-stochastic block matrix M_b (state ordering [up; down]):

    M_b = [ (1 - beta) * P_up        rho * E_b * 1^T   ]
          [ beta * I                  (1 - rho) * P_down ]

Iteration: v_{n+1} = M_b @ v_n, with v in R^{2N}.

Implementation note: the (1, 2) block is rank-1 (every column equals
rho * E_b). Materialising it as a sparse matrix would cost N**2
nonzeros (~10**10 for SLC). We instead apply M_b through a rank-1
trick at iteration time:

    (M_b @ v)_up   = (1 - beta) * P_up @ v_up + rho * E_b * sum(v_down)
    (M_b @ v)_down = beta * v_up + (1 - rho) * P_down @ v_down

`P_up` and `P_down` here are column-stochastic; the helper
`build_phase_matrix` produces row-stochastic matrices on the directed
graph, which we transpose once.

PARAMS:
    beta   commit rate (up -> down)              default 0.124
    rho    restart rate (down -> up)             default 0.147
    alpha_s, alpha_l  speed/lane physics weights default 0
    max_iters, tol    iteration controls
"""
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict

from scipy.sparse import csr_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

PARAMS = dict(config.DEFAULT_PARAMS)
PARAMS.setdefault('rho', 0.147)
PARAMS.setdefault('beta', 0.124)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_speed_lane_weights(graph_data: Dict, is_up_phase: bool) -> np.ndarray:
    """Per-link weights used to build P_up / P_down outgoing distributions.
    With alpha_s = alpha_l = 0 (default) this returns all-ones, giving
    uniform-over-neighbours transitions on the directed road graph.
    """
    links = list(graph_data['links'].keys())
    N = len(links)
    weights = np.ones(N)

    alpha_s = PARAMS['alpha_s'] if is_up_phase else -PARAMS['alpha_s']
    alpha_l = PARAMS['alpha_l'] if is_up_phase else -PARAMS['alpha_l']

    speeds, lanes = [], []
    for lid in links:
        ed = graph_data['links'].get(lid, {})
        speeds.append(ed.get('speed', 11.17))
        lanes.append(ed.get('lanes', 1.0))
    speeds = np.array(speeds)
    lanes = np.array(lanes)
    if np.mean(speeds) > 0: speeds = speeds / np.mean(speeds)
    if np.mean(lanes) > 0: lanes = lanes / np.mean(lanes)

    for i in range(N):
        w = 1.0
        if alpha_s != 0: w *= speeds[i] ** alpha_s
        if alpha_l != 0: w *= lanes[i] ** alpha_l
        weights[i] = w
    return weights


def build_phase_matrix(graph_data: Dict, weights: np.ndarray) -> csr_matrix:
    """Row-stochastic N x N transition matrix supported on the directed
    road graph. Row i is the next-link distribution from link i, weighted
    by `weights[j]` over outgoing neighbours j. No self-loops.
    """
    links = list(graph_data['links'].keys())
    N = len(links)
    id_to_idx = {lid: i for i, lid in enumerate(links)}

    row, col, data = [], [], []
    adj = graph_data.get('adjacency', {})

    for i, lid in enumerate(links):
        out_links = adj.get(lid, [])
        succ = [id_to_idx[ol] for ol in out_links if ol in id_to_idx]
        out_w = [weights[j] for j in succ]

        total = sum(out_w)
        if total > 0:
            for j, w in zip(succ, out_w):
                row.append(i); col.append(j); data.append(w / total)

    return csr_matrix((data, (row, col)), shape=(N, N))


def build_phase_kernels(graph_data: Dict):
    """Build column-stochastic P_up_cs, P_down_cs (transposes of row-stochastic
    P_up, P_down)."""
    up_w = compute_speed_lane_weights(graph_data, is_up_phase=True)
    down_w = compute_speed_lane_weights(graph_data, is_up_phase=False)
    P_up = build_phase_matrix(graph_data, up_w)
    P_down = build_phase_matrix(graph_data, down_w)
    return P_up.T.tocsr(), P_down.T.tocsr()


def build_teleportation_vector(graph_data: Dict, target_time: str,
                                npz_path: str = None) -> np.ndarray:
    """E_N = row-normalised popularity at `target_time`, projected onto the
    canonical link order of `graph_data`. Uses raw count matrix by default
    (POPULARITY_RAW_NPZ), per the walkthrough §6.
    """
    if npz_path is None:
        npz_path = str(config.POPULARITY_RAW_NPZ)
    logger.info(f"Loading teleportation prior E_b for {target_time} from {npz_path}")

    links = list(graph_data['links'].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}

    from core.io import load_popularity_npz
    bundle = load_popularity_npz(npz_path)
    matrix = bundle['matrix']
    times = bundle['times']
    pop_link_ids = bundle['link_ids']

    if target_time not in times:
        target_time = times[0]
        logger.warning(f"Target time not in matrix; defaulting to {target_time}")

    t_idx = times.index(target_time)
    row_vals = matrix.getrow(t_idx).toarray().ravel()

    proj = np.fromiter(
        (lid_to_idx.get(lid, -1) for lid in pop_link_ids),
        dtype=np.int64,
        count=len(pop_link_ids),
    )
    E_N = np.zeros(N)
    valid = proj >= 0
    np.add.at(E_N, proj[valid], row_vals[valid])

    s = E_N.sum()
    if s > 0:
        E_N /= s
    else:
        E_N = np.ones(N) / N
    return E_N


def power_iteration(P_up_cs: csr_matrix, P_down_cs: csr_matrix,
                     E_b: np.ndarray, beta: float, rho: float,
                     tol: float, max_iter: int):
    """Apply M_b (column-stochastic, rank-1 trick) until the L1 change drops
    below `tol`. Returns (v_up, v_down, n_iter)."""
    N = E_b.shape[0]
    v_up = E_b.copy()
    v_down = np.zeros(N)
    for k in range(max_iter):
        s_down = float(v_down.sum())
        v_up_new = (1.0 - beta) * (P_up_cs @ v_up) + rho * E_b * s_down
        v_down_new = beta * v_up + (1.0 - rho) * (P_down_cs @ v_down)
        s = float(v_up_new.sum() + v_down_new.sum())
        if s > 0:
            v_up_new /= s
            v_down_new /= s
        diff = float(np.abs(v_up_new - v_up).sum() + np.abs(v_down_new - v_down).sum())
        v_up, v_down = v_up_new, v_down_new
        if diff < tol:
            return v_up, v_down, k + 1
    return v_up, v_down, max_iter


def main():
    logger.info("Two-phase PageRank (PDF spec)")
    with open(config.GRAPH_FILE, 'r') as f:
        graph_data = json.load(f)
    links = list(graph_data['links'].keys())
    N = len(links)

    logger.info(f"Building column-stochastic phase kernels P_up, P_down for N={N}")
    P_up_cs, P_down_cs = build_phase_kernels(graph_data)

    target_time = "2018-09-08 08:00:00"
    E_b = build_teleportation_vector(graph_data, target_time)

    beta = PARAMS['beta']
    rho = PARAMS['rho']
    tol = PARAMS['tol']
    max_iter = PARAMS['max_iters']
    logger.info(f"Iterating: beta={beta} rho={rho} tol={tol} max_iter={max_iter}")
    v_up, v_down, n_iter = power_iteration(
        P_up_cs, P_down_cs, E_b, beta, rho, tol, max_iter
    )
    logger.info(f"Converged in {n_iter} iterations.")

    v_final = v_up + v_down
    v_final /= v_final.sum()

    abs_err = np.abs(E_b - v_final)
    overall_mre = float(np.mean(abs_err / (E_b + 1e-9)))
    logger.info(f"Overall MRE vs E_b: {overall_mre:.4f}")

    out = {lid: float(v_final[i]) for i, lid in enumerate(links)}
    with open(config.OUTPUT_FILE, 'w') as f:
        json.dump(out, f)
    logger.info(f"Wrote {config.OUTPUT_FILE}")


if __name__ == "__main__":
    main()
