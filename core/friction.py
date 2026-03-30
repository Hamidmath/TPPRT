import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack


def get_eval_metrics(truth, pred, top_k=100):
    """Compute Top-K MRE and overall MRE between truth and predicted distributions."""
    top_k_idx = np.argsort(truth)[::-1][:top_k]
    t_top = truth[top_k_idx]
    p_top = pred[top_k_idx]
    mre_top = np.mean(np.abs(t_top - p_top) / (t_top + 1e-9))
    mre_all = np.mean(np.abs(truth - pred) / (truth + 1e-9))
    return mre_top, mre_all


def build_custom_matrix(graph_data, config_dict):
    """Build a 2N x 2N Two-Phase matrix with configurable friction self-loops.

    Parameters
    ----------
    graph_data : dict
        Graph with 'links' and 'adjacency' keys.
    config_dict : dict
        Must contain 'alpha_s', 'alpha_l', 'beta'.
        Optional: 'self_loops' (bool), 'friction_factor' (float).
    """
    links = list(graph_data['links'].keys())
    N = len(links)
    id_to_idx = {lid: i for i, lid in enumerate(links)}
    adj = graph_data.get('adjacency', {})

    speeds = np.array([graph_data['links'][lid].get('speed', 11.17) for lid in links])
    lanes = np.array([graph_data['links'][lid].get('lanes', 1.0) for lid in links])
    lengths = np.array([graph_data['links'][lid].get('length', 100.0) for lid in links])

    mean_s = np.mean(speeds) if np.mean(speeds) > 0 else 1.0
    mean_l = np.mean(lanes) if np.mean(lanes) > 0 else 1.0

    def build_phase(is_up):
        a_s = config_dict['alpha_s'] if is_up else -config_dict['alpha_s']
        a_l = config_dict['alpha_l'] if is_up else -config_dict['alpha_l']

        row, col, data = [], [], []

        for i, lid in enumerate(links):
            out_lids = adj.get(lid, [])
            succ_idx = [id_to_idx[x] for x in out_lids if x in id_to_idx]

            weights = []
            for j in succ_idx:
                s_j = speeds[j] / mean_s
                l_j = lanes[j] / mean_l
                w = (s_j ** a_s) * (l_j ** a_l)
                weights.append(w)

            self_w = 0.0
            if config_dict.get('self_loops', True):
                travel_time = lengths[i] / max(speeds[i], 0.1)
                self_w = travel_time * config_dict.get('friction_factor', 0.05)

            total_w = sum(weights) + self_w
            if total_w > 0:
                for j, w in zip(succ_idx, weights):
                    row.append(i)
                    col.append(j)
                    data.append(w / total_w)
                if self_w > 0:
                    row.append(i)
                    col.append(i)
                    data.append(self_w / total_w)

        return csr_matrix((data, (row, col)), shape=(N, N))

    P_up = build_phase(True)
    P_down = build_phase(False)

    beta = config_dict['beta']
    P_up = P_up.multiply(1.0 - beta)

    row_T = np.arange(N)
    col_T = np.arange(N)
    data_T = np.ones(N) * beta
    T_down = csr_matrix((data_T, (row_T, col_T)), shape=(N, N))

    zero_block = csr_matrix((N, N))
    top_block = hstack([P_up, T_down])
    bottom_block = hstack([zero_block, P_down])

    return vstack([top_block, bottom_block])
