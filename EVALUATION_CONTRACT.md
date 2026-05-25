# Evaluation Contract — Two-Phase PageRank

When the user asks for "the error", "evaluation", or "how good is the
model", **read this file and follow it exactly**. Do not improvise the
sampling, metric, or comparison set unless the user explicitly asks
for something else.

The script that implements this contract is
`analysis/run_contract_eval.py`. Re-run it whenever the underlying
pipeline outputs change.

## Sampling

- **Bin count:** exactly **84 random target bins**, stratified as
  **12 random bins per day of the week** (Mon, Tue, Wed, Thu, Fri,
  Sat, Sun).
- A bin is eligible only if **both** the target $t$ and the
  *next-week sibling* $t + 7\,\text{days}$ exist in the popularity
  matrix. Bins from the last 7 days of the corpus are therefore not
  eligible as targets.
- Random seed is fixed (default `--seed 42`) so results are
  reproducible across re-runs.

## Error metric

MRE with a numerical floor in the denominator:

```
rel_i = |truth_i - pred_i| / (truth_i + 1e-6)
```

Two aggregates per (truth, pred) pair:

- **Top-100 MRE** — mean of `rel_i` over the 100 link indices with
  the largest `truth_i`.
- **Overall MRE** — mean of `rel_i` over all `N = 99,716` links.

## Comparisons (all three, always)

For every sampled target $t$ with sibling $t' = t + 7\,\text{days}$:

| # | Name | Truth | Prediction | What it measures |
|---|------|-------|------------|------------------|
| 1 | **dataset → dataset (week-to-week)** | $E_b(t')$ | $E_b(t)$ | Natural week-to-week variability of the empirical signal at the same weekday/hour-of-day. The noise floor. |
| 2 | **dataset → model (forecast)** | $E_b(t')$ | model($E_b(t)$) | How well the model, fed this week's $E_b$, predicts next week's empirical $E_b$. |
| 3 | **dataset → model (reconstruction)** | $E_b(t)$ | model($E_b(t)$) | How well the model reconstructs its own input bin. |

Each comparison is reported with its **Top-100 MRE** and **Overall
MRE**, mean and median over the 84 sampled bins.

## Pass criterion

The model passes if both:

- **forecast Top-100 MRE** $\le$ **week-to-week Top-100 MRE**, and
- **forecast Overall MRE** $\le$ **week-to-week Overall MRE**.

In words: the model's forecast must be no worse than just using last
week's data at the same time slot.

## Model parameters

The contract evaluation uses the calibrated values from the
route-length distribution study:

- $\beta = 0.124$
- $\rho = 0.147$
- $\alpha_s = \alpha_l = 0$
- tolerance $= 10^{-7}$, max iterations $= 300$

The popularity matrix used is the **raw count-based** matrix at
`data/popularity_results.npz` (i.e. the matrix produced by
`pipeline/analyze_popularity.py` per the walkthrough's count rule).
Smoothed matrices are explicitly out of scope for this contract.

## Output

`analysis/run_contract_eval.py` writes:

- `results/contract_eval/contract_eval.json` — full per-bin and
  aggregate numbers
- `results/contract_eval/contract_eval.png` — a side-by-side bar
  chart of the three comparisons

and prints a final table to stdout.

## When this contract changes

Edit this file in the same commit that changes the script, and
report the change to the user up front. Do not silently shift
sampling, metric, or comparison choices.
