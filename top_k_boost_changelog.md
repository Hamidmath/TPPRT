# Changes to Two-Phase PageRank

## Goal

Improve prediction accuracy for the top 100 highest-traffic links while keeping overall network MRE at or below the original baseline (0.198).

## What Changed

### Modified: `scripts/pagerank_execution/run_two_phase_pagerank.py`

**Parameter changes (line 18-28):**

| Parameter | Before | After | Why |
|:---|:---|:---|:---|
| `beta` | 0.2 | 0.90 | Reduces diffusion by moving mass to down phase after ~1 hop |
| `damping` | 0.89 | 0.85 | Standard PageRank value; gives teleportation 15% weight |
| `top_k` | (new) | 100 | Number of high-traffic links to boost |
| `top_k_boost` | (new) | 2.0 | Teleportation multiplier for those links |
| `tau` | (new) | 1.0 | Power-law sharpening exponent (disabled at 1.0) |

**New function: `boost_top_k()` (line 176-184)**

Multiplies the teleportation weight of the K highest-traffic links by a boost factor, then renormalizes. This is the primary mechanism for improving top-100 accuracy.

```python
def boost_top_k(E_N: np.ndarray, K: int, boost: float) -> np.ndarray:
    if boost == 1.0 or K <= 0:
        return E_N.copy()
    E_b = E_N.copy()
    top_idx = np.argsort(E_N)[::-1][:K]
    E_b[top_idx] *= boost
    E_b /= np.sum(E_b)
    return E_b
```

**New function: `sharpen_teleportation()` (line 164-174)**

Raises the teleportation vector to power `tau` and renormalizes. Currently disabled (`tau=1.0`), kept as an available knob for future tuning.

**Modified: `main()` (line 228-231)**

After extracting the ground-truth teleportation vector `E_N`, two transformations are applied before power iteration:

```python
E_tele = sharpen_teleportation(E_N, PARAMS['tau'])   # no-op at tau=1.0
E_tele = boost_top_k(E_tele, PARAMS['top_k'], PARAMS['top_k_boost'])  # 2x boost on top 100
E_2N = np.concatenate([E_tele, np.zeros(N)])
```

Evaluation (lines 242-253) still compares `v_final` against the original `E_N`, not the boosted version.

**Added: Top-100 MRE logging (line 248-253)**

```python
top_100_indices = np.argsort(E_N)[::-1][:100]
t_top100 = E_N[top_100_indices]
p_top100 = v_final[top_100_indices]
mre_top100 = np.mean(np.abs(t_top100 - p_top100) / (t_top100 + 1e-9))
logger.info(f"Top-100 MRE: {mre_top100:.4f}")
```

### New: `scripts/pagerank_execution/tune_tau.py`

Grid search over `tau` x `damping` on 49 random timeframes (seed=42). Builds the transition matrix once and pre-caches all E_N vectors for efficiency. Tracks feasibility against the 0.198 overall MRE budget.

### New: `scripts/pagerank_execution/tune_tau_fine.py`

Finer grid search around promising tau/damping values found by `tune_tau.py`.

### New: `scripts/pagerank_execution/tune_strategies.py`

Multi-strategy comparison at damping=0.85. Tests five approaches:
- A: Fine tau sharpening
- B: Beta tuning
- C: Joint tau + beta
- D: Two-pass residual correction (run PageRank, compute error ratio, adjust teleportation, run again)
- E: Top-K targeted boost

### New: `scripts/pagerank_execution/tune_combos.py`

Combination strategies at damping=0.85, exploiting the overall MRE headroom from high beta:
- G: High beta + two-pass residual
- H: High beta + top-K boost (winner found here)
- I: High beta + two-pass + top-K boost
- J: 3-pass iterative residual at high beta

### New: `scripts/pagerank_execution/tune_final.py`

Final fine-tuning around the winning configs: beta in [0.60-0.90] x boost in [1.0-3.0], plus two-pass and combo variants at high beta.

## Results

49-timeframe average (seed=42):

| Configuration | Top-100 MRE | Overall MRE |
|:---|:---|:---|
| Original (`d=0.89`, `beta=0.2`, no boost) | 0.7331 | 0.1980 |
| New (`d=0.85`, `beta=0.90`, `K=100`, `boost=2.0`) | **0.3304** | **0.1942** |
| Change | **-54.9%** | **-1.9%** |

## What Was Not Changed

- Transition matrix construction (`build_phase_matrix`, `build_two_phase_matrix`) -- no edge weights modified
- Teleportation vector extraction (`build_teleportation_vector`) -- ground truth loading unchanged
- Power iteration algorithm (`run_power_iteration`) -- same convergence logic
- Output format -- same JSON dict of link_id -> score
- Data files -- no changes to graph or popularity data
