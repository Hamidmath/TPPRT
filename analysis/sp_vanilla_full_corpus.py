"""SP vanilla baseline on the FULL corpus (8,640 five-minute slots).

No tuning. Chain alpha = 0.0508 FIXED, alpha_s = alpha_l = alpha_ang = 0
so the transition is uniform over each source's out-neighbors.

Two priors, selected via TPPR_PRIOR:

    TPPR_PRIOR=uniform   prior = (1/N, ..., 1/N) for every slot
                         (a single chain pass gives the stationary v*,
                         which is then compared to F_b at each slot)

    TPPR_PRIOR=origin    prior = diffused trip-origins E_b at slot b
                         (Laplace 0.01 + gamma=0.20 graph diffusion,
                         same as the popularity smoothing)

For each of the 8,640 slots we compute the Overall MRE
    MRE_b = (1/N) sum_i |F_{b,i} - v_{b,i}| / (F_{b,i} + 10^-6)
where F_b is the diffused popularity at slot b. Output: per-slot list,
mean, median.

Output: results/sp_vanilla_full_<PRIOR>.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from core.io import load_popularity_npz

INPUTS  = Path(os.environ.get("TPPR_INPUTS",  str(config.DATA_DIR)))
RESULTS = Path(os.environ.get("TPPR_RESULTS", str(config.PROJECT_ROOT / "results")))
RESULTS.mkdir(parents=True, exist_ok=True)

ORIGINS_NPZ = str(INPUTS / "origins_results.npz")
SMOOTH_NPZ  = str(INPUTS / "popularity_results_smoothed_osm_gamma020.npz")
GRAPH_JSON  = str(INPUTS / "city_graph_full.json")

EPS = 1e-6
TOL, MAX_ITER = 1e-5, 200
ALPHA_FIXED = 0.0508
LAPLACE_ALPHA = 0.01
GAMMA = 0.20

PRIOR = os.environ.get("TPPR_PRIOR", "uniform")
assert PRIOR in ("uniform", "origin"), \
    f"TPPR_PRIOR must be 'uniform' or 'origin'; got {PRIOR!r}"

OUT = RESULTS / f"sp_vanilla_full_{PRIOR}.json"


def build_P_uniform(graph, links, lid_to_idx):
    """Out-uniform column-stochastic transition: per source i, equal
    weight 1/k_i on each of k_i successors."""
    adj = graph.get("adjacency", {})
    rows, cols, data = [], [], []
    for i, lid in enumerate(links):
        succ = [lid_to_idx[ol] for ol in adj.get(lid, []) if ol in lid_to_idx]
        if not succ: continue
        w = 1.0 / len(succ)
        for j in succ:
            rows.append(j); cols.append(i); data.append(w)
    return csr_matrix((data, (rows, cols)),
                       shape=(len(links), len(links)), dtype=np.float64)


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


def load_pop_full(npz_path, N, lid_to_idx):
    b = load_popularity_npz(npz_path)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    pop_ids = [str(lid) for lid in b["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in pop_ids], dtype=np.int64)
    valid = proj >= 0
    out = np.zeros((len(times), N), dtype=np.float32)
    for ti in range(len(times)):
        row = M.getrow(ti).toarray().ravel().astype(np.float64)
        E = np.zeros(N, dtype=np.float64)
        np.add.at(E, proj[valid], row[valid])
        s = E.sum()
        if s > 0: E /= s
        out[ti] = E.astype(np.float32)
    return times, out


def diffuse_one(raw_row, raw_proj, raw_valid, N, P_diff):
    c = np.full(N, LAPLACE_ALPHA, dtype=np.float64)
    if raw_row is not None:
        c_add = np.zeros(N, dtype=np.float64)
        np.add.at(c_add, raw_proj[raw_valid], raw_row[raw_valid])
        c = c + c_add
    c_diff = (1.0 - GAMMA) * c + GAMMA * (P_diff @ c)
    s = c_diff.sum()
    if s > 0: c_diff /= s
    return c_diff


def load_diffused_origins_full(npz_path, N, lid_to_idx, all_times, P_diff):
    b = load_popularity_npz(npz_path)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    orig_ids = [str(lid) for lid in b["link_ids"]]
    proj = np.array([lid_to_idx.get(lid, -1) for lid in orig_ids], dtype=np.int64)
    valid = proj >= 0
    time_idx = {t: i for i, t in enumerate(times)}
    out = np.zeros((len(all_times), N), dtype=np.float32)
    for k, t in enumerate(all_times):
        ti = time_idx.get(t)
        if ti is None:
            v = diffuse_one(None, proj, valid, N, P_diff)
        else:
            raw = M.getrow(ti).toarray().ravel().astype(np.float64)
            v = diffuse_one(raw, proj, valid, N, P_diff)
        out[k] = v.astype(np.float32)
    return out


def main():
    t0 = time.time()
    print(f"[start] SP vanilla full corpus  prior={PRIOR}  alpha={ALPHA_FIXED}")

    print("[load] graph ...")
    with open(GRAPH_JSON) as f:
        graph = json.load(f)
    links = list(graph["links"].keys())
    N = len(links)
    lid_to_idx = {lid: i for i, lid in enumerate(links)}
    print(f"  N={N:,}")

    P = build_P_uniform(graph, links, lid_to_idx)
    print(f"  P (uniform out): nnz={P.nnz:,}")

    print("[load] diffused popularity (full 8,640 slots) ...")
    times, F = load_pop_full(SMOOTH_NPZ, N, lid_to_idx)
    print(f"  F shape: {F.shape}  times: {times[0]} ... {times[-1]}")

    if PRIOR == "uniform":
        print("[chain] one pass from uniform u ...")
        u = np.full(N, 1.0 / N, dtype=np.float64)
        v_star = sp_iter(P, u, ALPHA_FIXED)
        print(f"  v* sum={v_star.sum():.6f}  max={v_star.max():.6e}")
        print("[eval] MRE against each F_b ...")
        ov_list = []
        for ti in range(len(times)):
            tgt = F[ti].astype(np.float64)
            ov_list.append(float(np.mean(np.abs(tgt - v_star) / (tgt + EPS))))
    else:
        print("[load+diffuse] per-slot origins ...")
        P_diff = build_P_diffusion(graph, links, lid_to_idx)
        E = load_diffused_origins_full(ORIGINS_NPZ, N, lid_to_idx, times, P_diff)
        print(f"  E shape: {E.shape}")
        print("[chain] per-slot SP iter from E_b ...")
        ov_list = []
        for ti in range(len(times)):
            v = sp_iter(P, E[ti], ALPHA_FIXED)
            tgt = F[ti].astype(np.float64)
            ov_list.append(float(np.mean(np.abs(tgt - v) / (tgt + EPS))))
            if (ti + 1) % 1000 == 0:
                print(f"  slot {ti+1}/{len(times)} done")

    mean = float(np.mean(ov_list))
    med  = float(np.median(ov_list))
    print(f"\n[done] {time.time()-t0:.0f}s")
    print(f"  Ov.MRE mean   = {mean:.4f}")
    print(f"  Ov.MRE median = {med:.4f}")

    OUT.write_text(json.dumps(dict(
        model="SP vanilla full-corpus",
        prior=PRIOR, alpha_fixed=ALPHA_FIXED,
        laplace_alpha=LAPLACE_ALPHA, gamma=GAMMA,
        n_slots=len(times),
        overall_mre_mean=mean,
        overall_mre_median=med,
        per_slot_ov=ov_list,
    ), indent=2))
    print(f"  saved {OUT}")


if __name__ == "__main__":
    main()
