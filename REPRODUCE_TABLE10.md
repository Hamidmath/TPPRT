# Reproducing Table 10 (additive ladder, whole-network)

This document lets a reviewer re-run every row in Table 10 of the paper
(`documents/walkthrough2/main.tex`, section "Additive ladder: damping and
road-type tuning on the whole network").

All scripts run from the project root.

## 1. Environment

Pinned in `requirements.txt`:

```
python 3.12.3   (>=3.11, <3.13 acceptable)
numpy  2.4.4
scipy  1.17.1
```

Install in a fresh venv:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Input data (with hashes)

| File | Size | Shape | SHA256 |
|------|------|-------|--------|
| `data/city_graph_full.json` | 16 MB | 99,716 links, 99,681 adjacency entries | `2f9023ce7b7543b7d4ca1489996867549e4431c2b28ff86ffa376a9038d8af3b` |
| `data/popularity_results_smoothed_osm_gamma020.npz` | 62 MB | matrix (8640, 99716) float32; times (8640,); link_ids (99716,) | `7e8fd4221dd3ba21c1805ca626a6d4cd17d8e321d7112bb3b2488e7da83af1a5` |

Verify locally:

```bash
sha256sum data/city_graph_full.json data/popularity_results_smoothed_osm_gamma020.npz
```

`popularity_results_smoothed_osm_gamma020.npz` is the diffused popularity
prior $\tilde E_b$ at $\gamma=0.20$. It is the output of the diffusion stage
described in section "Diffusion" of the paper; from that section onward, all
inputs and targets used in this paper are diffused at $\gamma=0.20$.

## 3. Sampling spec (frozen across all rows)

Identical for every row in Table 10:

```
seed                 = 7
bins per weekday     = 120
pairs                = 840   (120 per weekday for 7 weekdays)
pair construction    = (t, t + 7 days)  where both t and t+7d are present
                       in the diffused popularity matrix
input vector  for t  = diffused popularity at t
target vector for t  = diffused popularity at t+7d
epsilon              = 1e-6   (in MRE: |target - pred| / (target + eps))
top_k                = 100    (top-100 busiest links of target)
```

The pair sampler lives in every script as `sample_pairs(times, seed=7, bpw=120)`.
It is deterministic given the same input times list (sorted lexicographically).

Iteration spec for the chains:

```
single-phase    max_iter = 100   tol on L1 increment = 1e-5
two-phase       max_iter = 200   tol on L1 increment = 1e-6
```

Optimizer (tuning rows): L-BFGS-B, 5 random restarts unless noted,
`eps=2e-3` (finite-difference step), `ftol=1e-6`, `gtol=1e-4`, `maxiter=20`.
The bounds used: $\alpha,\beta,\rho \in [0.01, 0.15]$; $\alpha_s,\alpha_l$ unbounded.

## 4. Row-by-row manifest

The "JSON key" column is the dot-path inside the output JSON that holds
the Overall MRE used in the paper.

| # | Variant (paper) | Script | Output JSON | JSON key | Ov. MRE |
|---|-----------------|--------|-------------|----------|---------|
| 1 | SP vanilla, $\alpha=0.05$ | `analysis/vanilla_pr_two_alphas.py` | `results/vanilla_pr_two_alphas.json` | `results.alpha_0.05.overall_mean` | 1.072 |
| 2 | SP vanilla, $\alpha=0.15$ | `analysis/vanilla_pr_two_alphas.py` | `results/vanilla_pr_two_alphas.json` | `results.alpha_0.15.overall_mean` | 0.923 |
| 3 | SP + lanes only | `analysis/sp_tune_three_ov.py` | `results/sp_tune_three_ov.json` | `variants.lanes_only.overall_mean` | 0.916 |
| 4 | SP + speed only | `analysis/sp_tune_three_ov.py` | `results/sp_tune_three_ov.json` | `variants.speed_only.overall_mean` | 0.870 |
| 5 | SP + both | `analysis/sp_tune_three_ov.py` | `results/sp_tune_three_ov.json` | `variants.both.overall_mean` | 0.869 |
| 6 | SP + both, $\alpha$ fixed | `analysis/sp_tune_alpha005_fixed.py` | `results/sp_tune_alpha005_fixed.json` | `overall_mean` | 1.002 |
| 7 | TP vanilla, $\beta=0.102, \rho=0.101$ | `analysis/tp_eval_calibrated.py` | `results/tp_eval_calibrated.json` | `overall_mean` | 1.041 |
| 8 | TP vanilla, $\beta=\rho=0.15$ | `analysis/tp_two_betas.py` | `results/tp_two_betas.json` | `results.beta_0.15.overall_mean` | 0.984 |
| 9 | TP + lanes only | `analysis/tp_tune_three_ov.py` | `results/tp_tune_three_ov.json` | `variants.lanes_only.overall_mean` | 0.962 |
| 10 | TP + speed only | `analysis/tp_tune_three_ov.py` | `results/tp_tune_three_ov.json` | `variants.speed_only.overall_mean` | (in progress) |
| 11 | TP + both | `analysis/tp_tune_three_ov.py` | `results/tp_tune_three_ov.json` | `variants.both.overall_mean` | (in progress) |
| 12 | TP + both, $(\beta,\rho)$ fixed | `analysis/tp_tune_calibrated_fixed.py` | `results/tp_tune_calibrated_fixed.json` | `overall_mean` | 0.970 |

Best parameter values reported (for the additive rows):

| # | Parameters at optimum |
|---|----------------------|
| 3 | $\alpha=0.150$, $\alpha_l=+0.337$ |
| 4 | $\alpha=0.150$, $\alpha_s=+2.859$ |
| 5 | $\alpha=0.150$, $\alpha_s=+2.674$, $\alpha_l=+0.169$ |
| 6 | $\alpha=0.050$ (fixed), $\alpha_s=+2.774$, $\alpha_l=+0.243$ |
| 9 | $\beta=\rho=0.150$, $\alpha_l=+1.875$ |
| 12 | $\beta=0.102$, $\rho=0.101$ (fixed), $\alpha_s=+2.816$, $\alpha_l=+0.209$ |

## 5. How to run

Every script reads paths from these env vars (or falls back to project layout):

```bash
export TPPR_INPUTS=$PWD/data
export TPPR_RESULTS=$PWD/results
```

### Local

```bash
# Vanilla SP at alpha in {0.05, 0.15} (rows 1, 2). ~5 min on a single core.
python analysis/vanilla_pr_two_alphas.py

# SP three-way road-type tuning (rows 3, 4, 5). ~3-4 h with 5 starts.
python analysis/sp_tune_three_ov.py

# SP at alpha=0.05 fixed, tune (alpha_s, alpha_l) (row 6). ~1-2 h.
python analysis/sp_tune_alpha005_fixed.py

# TP at (beta=0.102, rho=0.101) one-off (row 7). ~10 min.
python analysis/tp_eval_calibrated.py

# TP vanilla at beta=rho in {0.05, 0.15} (row 8). ~30 min.
python analysis/tp_two_betas.py

# TP three-way road-type tuning (rows 9, 10, 11). ~12 h with 5 starts.
python analysis/tp_tune_three_ov.py

# TP at (beta=0.102, rho=0.101) fixed, tune (alpha_s, alpha_l) (row 12). ~4 h.
python analysis/tp_tune_calibrated_fixed.py
```

### CHPC (Granite)

Each script has a matching `.slurm` file in `analysis/`. They request
`granite-guest` QOS, 40 GB memory, 4 CPUs, 12 h walltime; submit with
`sbatch analysis/<name>.slurm`.

The slurm wrappers stage inputs from `~/tppr/sept15_sweep/inputs/` and write
to `~/tppr/sept15_sweep/results/`. On the local machine the paths come from
`config.py`.

## 6. Determinism

- Pair sampling: `random.Random(seed=7).shuffle` over `by_weekday[wd]`
  lists built in iteration order from `sorted(times)`. Same input file
  $\Rightarrow$ same 840 pairs.
- Single-phase iteration: deterministic given $\alpha$ and the transition
  matrix $P$.
- L-BFGS-B finite-difference gradient: deterministic at fixed $\epsilon=2e-3$.
- Start points for L-BFGS-B: hard-coded in each script (not RNG-drawn).
  See the `starts = [...]` list in each tuning script.

Re-running these scripts on the same inputs should reproduce every cell
to within $10^{-4}$ of the reported MRE.

## 7. Climatology baseline

Not in Table 10 but referenced in the section narrative.

| Variant | Script | Output JSON | Ov. MRE |
|---------|--------|-------------|---------|
| Climatology (mean of 3 prior weeks $\to$ 4th) | `analysis/climatology_diffused_840.py` | `results/climatology_diffused_840.json` | 0.629 |
