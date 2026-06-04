"""Week-to-week Overall MRE of the DIFFUSED popularity on the full corpus.
No model, no prior, no origin: just the diffused popularity matrix compared
to itself across consecutive weeks. For every 5-min bin t whose sibling
t+7 days also exists, treat next week as truth and this week as prediction:
    rel_i = |truth_i - pred_i| / (truth_i + 1e-6)
Overall MRE of a bin = mean of rel_i over ALL links. Aggregate (mean,
median, std) over every bin of the week. Pairs: wk2 vs wk1, wk3 vs wk2,
wk4 vs wk3. Each bin normalized to a probability over links."""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.io import load_popularity_npz

INPUTS  = Path(os.environ.get("TPPR_INPUTS",  str(ROOT / "data")))
RESULTS = Path(os.environ.get("TPPR_RESULTS", str(ROOT / "results/origin_pop_ladder")))
DIFF_NPZ = str(INPUTS / "popularity_results_smoothed_osm_gamma020.npz")
OUT_JSON = RESULTS / "week_to_week_mre.json"
EPS = 1e-6
DAY7 = np.timedelta64(7, "D")
LABELS = {0: "Week 2 vs Week 1", 1: "Week 3 vs Week 2", 2: "Week 4 vs Week 3"}

def row_prob(M, j):
    r = np.asarray(M.getrow(j).toarray()).ravel().astype(np.float64)
    s = r.sum()
    return r / s if s > 0 else r

def main():
    t0 = time.time()
    print(f"[load] {DIFF_NPZ}", flush=True)
    b = load_popularity_npz(DIFF_NPZ)
    M = b["matrix"]; T, N = M.shape
    times = [np.datetime64(t) for t in b["times"]]
    idx = {t: i for i, t in enumerate(times)}
    tstart = min(times)
    week_of = {t: int((t - tstart) / np.timedelta64(1, "D")) // 7 for t in times}
    print(f"  T={T} bins  N={N} links  span {times[0]} .. {times[-1]}", flush=True)
    # sanity on a sample row
    r0 = np.asarray(M.getrow(0).toarray()).ravel().astype(np.float64)
    print(f"  row0 sum={r0.sum():.4f} nnz={(r0>0).sum()} min_pos={r0[r0>0].min() if (r0>0).any() else 0:.2e} max={r0.max():.2e}", flush=True)

    acc = defaultdict(list)
    skipped_truth_zero = 0
    for j, t in enumerate(times):
        w = week_of[t]
        if w not in (0, 1, 2):
            continue
        js = idx.get(t + DAY7)
        if js is None:
            continue
        pred = row_prob(M, j)        # this week
        truth = row_prob(M, js)      # next week  = truth
        if truth.sum() <= 0:
            skipped_truth_zero += 1
            continue
        rel = np.abs(truth - pred) / (truth + EPS)
        acc[w].append(float(rel.mean()))

    out = {"npz": os.path.basename(DIFF_NPZ), "eps": EPS, "n_links": int(N),
           "skipped_truth_zero": skipped_truth_zero, "pairs": {}}
    print(f"\n{'pair':<18} {'#bins':>6} {'mean':>9} {'median':>9} {'std':>9}", flush=True)
    for w in (0, 1, 2):
        a = np.array(acc[w], float)
        rec = {"n_bins": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
               "std": float(a.std()), "min": float(a.min()), "max": float(a.max())}
        out["pairs"][LABELS[w]] = rec
        print(f"{LABELS[w]:<18} {a.size:>6d} {rec['mean']:>9.4f} {rec['median']:>9.4f} {rec['std']:>9.4f}", flush=True)

    print("\n--- LaTeX rows (mean, std) ---", flush=True)
    for w in (0, 1, 2):
        rec = out["pairs"][LABELS[w]]
        print(f"  {LABELS[w]} & ${rec['mean']:.4f}$ & ${rec['std']:.4f}$ \\\\", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\n[done] {time.time()-t0:.0f}s saved {OUT_JSON}", flush=True)

if __name__ == "__main__":
    main()
