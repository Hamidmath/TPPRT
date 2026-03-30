"""
Independent verification of the optimal smoothing parameter gamma*.
Uses a DIFFERENT random seed (seed=123) and a FINER grid around the
candidate optimal value gamma=0.26 to confirm it is a true minimum.
"""
import numpy as np
import scipy.sparse as sparse
import sys
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core import build_two_phase_matrix, PARAMS
import config

def build_directed_adjacency(city_graph, link_ids):
    n = len(link_ids)
    id_to_idx = {lid: i for i, lid in enumerate(link_ids)}
    row_ind, col_ind, data = [], [], []
    adj = city_graph.get('adjacency', {})
    for lid, out_links in adj.items():
        if lid not in id_to_idx: continue
        i = id_to_idx[lid]
        valid_outs = [nid for nid in out_links if nid in id_to_idx]
        if not valid_outs: continue
        for out_lid in valid_outs:
            row_ind.append(i); col_ind.append(id_to_idx[out_lid])
            data.append(1.0 / len(valid_outs))
    return sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))

def load_json(fp):
    with open(fp) as f: return json.load(f)

def load_pop(fp):
    loader = np.load(fp, allow_pickle=True)
    mat = sparse.csr_matrix(
        (loader['matrix_data'], loader['matrix_indices'], loader['matrix_indptr']),
        shape=loader['matrix_shape'])
    return {'matrix': mat, 'times': list(loader['times']),
            'link_ids': list(loader['link_ids'])}

def main():
    print("Loading data...")
    raw = load_pop(str(config.POPULARITY_RAW_NPZ))
    graph = load_json(str(config.GRAPH_FILE))

    links_raw = raw['link_ids']; N_raw = len(links_raw)
    links_graph = list(graph['links'].keys()); N_graph = len(links_graph)

    graph_idx_map = {lid: i for i, lid in enumerate(links_graph)}
    raw_to_graph, valid_raw = [], []
    for i, lid in enumerate(links_raw):
        if lid in graph_idx_map:
            raw_to_graph.append(graph_idx_map[lid]); valid_raw.append(i)
    raw_to_graph = np.array(raw_to_graph); valid_raw = np.array(valid_raw)

    print("Building adjacency...")
    P_dir = build_directed_adjacency(graph, links_raw)

    print("Building Two-Phase Matrix...")
    PARAMS['alpha_s'] = 0.0; PARAMS['alpha_l'] = 0.0; PARAMS['beta'] = 0.2
    M_2N = build_two_phase_matrix(graph)
    M_2N_T = M_2N.transpose().tocsr()

    times = raw['times']
    tf = defaultdict(list)
    for t_idx, t_str in enumerate(times):
        dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
        tf[(dt.weekday(), dt.strftime("%H:%M:%S"))].append(t_idx)
    valid_splits = {k: v for k, v in tf.items() if len(v) >= 2}

    random.seed(123)
    num_splits = min(80, len(valid_splits))
    selected = random.sample(list(valid_splits.keys()), num_splits)
    print(f"Selected {num_splits} splits (seed=123).")

    gammas = [0.0, 0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.24,
              0.25, 0.26, 0.27, 0.28, 0.30, 0.35, 0.40, 0.50, 0.60]
    n_g = len(gammas)

    avg_mse_raw = np.zeros(n_g)
    avg_corr_raw = np.zeros(n_g)
    avg_mse_pr = np.zeros(n_g)
    avg_corr_pr = np.zeros(n_g)
    avg_var_ret = np.zeros(n_g)

    alpha = 0.1; damping = 0.89; valid_count = 0
    batch_size = 10

    for b_start in range(0, num_splits, batch_size):
        b_keys = selected[b_start:b_start+batch_size]
        print(f"  Batch {b_start//batch_size+1}/{(num_splits+batch_size-1)//batch_size}...")

        target_B, c_A = [], []
        for k in b_keys:
            t_indices = valid_splits[k]
            np.random.seed(hash(k) % (2**32))
            idx_copy = list(t_indices); random.shuffle(idx_copy)
            half = len(idx_copy) // 2
            sA, sB = idx_copy[:half], idx_copy[half:]

            rA = np.zeros(N_raw)
            for idx in sA:
                row = raw['matrix'].getrow(idx)
                for i, val in zip(row.indices, row.data): rA[i] += float(val)

            rB = np.zeros(N_raw)
            for idx in sB:
                row = raw['matrix'].getrow(idx)
                for i, val in zip(row.indices, row.data): rB[i] += float(val)

            if rB.sum() <= 0: continue
            target_B.append(rB / rB.sum()); c_A.append(rA + alpha)

        if not c_A: continue
        bs = len(c_A); valid_count += bs

        E_2N_B = np.zeros((2*N_graph, bs))
        for b in range(bs):
            E_2N_B[raw_to_graph, b] = target_B[b][valid_raw]
            s = E_2N_B[:N_graph, b].sum()
            if s > 0: E_2N_B[:N_graph, b] /= s
            else: E_2N_B[:N_graph, b] = 1.0/N_graph

        v_2N_B = E_2N_B.copy()
        for _ in range(60):
            v_next = damping * M_2N_T.dot(v_2N_B) + (1-damping) * E_2N_B
            S = np.sum(v_next, axis=0); miss = 1.0 - S; miss[miss<0] = 0
            v_next += E_2N_B * miss; v_2N_B = v_next

        tpr_full = v_2N_B[:N_graph,:] + v_2N_B[N_graph:,:]
        tpr = np.zeros((N_raw, bs))
        for b in range(bs):
            tpr[valid_raw, b] = tpr_full[raw_to_graph, b]
            s = tpr[:,b].sum()
            if s > 0: tpr[:,b] /= s

        total_cols = bs * n_g
        E_2N_A = np.zeros((2*N_graph, total_cols))
        dist_A = np.zeros((N_raw, total_cols))
        base_vars = []

        for b in range(bs):
            cA = c_A[b]; P_cA = P_dir.dot(cA)
            base_vars.append(np.var(cA / cA.sum()))
            for g_i, g in enumerate(gammas):
                c_diff = (1-g)*cA + g*P_cA
                dist = c_diff / c_diff.sum()
                col = b*n_g + g_i; dist_A[:,col] = dist
                E_2N_A[raw_to_graph, col] = dist[valid_raw]
                s = E_2N_A[:N_graph,col].sum()
                if s > 0: E_2N_A[:N_graph,col] /= s
                else: E_2N_A[:N_graph,col] = 1.0/N_graph

        v_2N_A = E_2N_A.copy()
        for _ in range(60):
            v_next = damping * M_2N_T.dot(v_2N_A) + (1-damping) * E_2N_A
            S = np.sum(v_next, axis=0); miss = 1.0 - S; miss[miss<0] = 0
            v_next += E_2N_A * miss; v_2N_A = v_next

        dpr_full = v_2N_A[:N_graph,:] + v_2N_A[N_graph:,:]
        dpr = np.zeros((N_raw, total_cols))
        for col in range(total_cols):
            dpr[valid_raw, col] = dpr_full[raw_to_graph, col]
            s = dpr[:,col].sum()
            if s > 0: dpr[:,col] /= s

        for b in range(bs):
            tB = target_B[b]; tprB = tpr[:,b]; bv = base_vars[b]
            for g_i, g in enumerate(gammas):
                col = b*n_g + g_i
                mse_r = np.mean((dist_A[:,col] - tB)**2)
                r_r, _ = pearsonr(dist_A[:,col], tB)
                mse_p = np.mean((dpr[:,col] - tprB)**2)
                r_p, _ = pearsonr(dpr[:,col], tprB)
                avg_mse_raw[g_i] += mse_r
                if not np.isnan(r_r): avg_corr_raw[g_i] += r_r
                avg_mse_pr[g_i] += mse_p
                if not np.isnan(r_p): avg_corr_pr[g_i] += r_p
                avg_var_ret[g_i] += (np.var(dist_A[:,col]) / bv)

    if valid_count > 0:
        avg_mse_raw /= valid_count; avg_corr_raw /= valid_count
        avg_mse_pr /= valid_count; avg_corr_pr /= valid_count
        avg_var_ret = (avg_var_ret / valid_count) * 100

    opt_raw = gammas[np.argmin(avg_mse_raw)]
    opt_corr = gammas[np.argmax(avg_corr_raw)]
    opt_pr = gammas[np.argmin(avg_mse_pr)]

    print(f"\n{'='*60}")
    print(f"VERIFICATION RESULTS (seed=123, {valid_count} splits, {n_g} gamma values)")
    print(f"{'='*60}")
    print(f"{'g':>6} | {'MSE_raw':>14} | {'Corr':>8} | {'MSE_PR':>14} | {'Var%':>7}")
    print(f"{'-'*6}-+-{'-'*14}-+-{'-'*8}-+-{'-'*14}-+-{'-'*7}")
    for i, g in enumerate(gammas):
        marker = " *" if g == opt_raw else ""
        print(f"{g:6.2f} | {avg_mse_raw[i]:.6e} | {avg_corr_raw[i]:.6f} | "
              f"{avg_mse_pr[i]:.6e} | {avg_var_ret[i]:6.1f}{marker}")
    print(f"\nOptimal gamma (min MSE_raw): {opt_raw}")
    print(f"Optimal gamma (max corr):    {opt_corr}")
    print(f"Optimal gamma (min MSE_PR):  {opt_pr}")

    out = config.PROJECT_ROOT / 'docs' / 'verification_results.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'seed': 123, 'splits_used': valid_count,
            'gammas': gammas,
            'mse_raw': avg_mse_raw.tolist(),
            'corr_raw': avg_corr_raw.tolist(),
            'mse_pr': avg_mse_pr.tolist(),
            'corr_pr': avg_corr_pr.tolist(),
            'variance_ret': avg_var_ret.tolist(),
            'optimal_gamma_mse_raw': opt_raw,
            'optimal_gamma_corr': opt_corr,
            'optimal_gamma_mse_pr': opt_pr,
        }, f, indent=4)
    print(f"\nResults saved to {out}")

if __name__ == '__main__':
    main()
