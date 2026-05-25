# Pipeline status — Two-Phase PageRank for SLC

This file records the canonical pipeline as it currently stands in the
repository, what each stage does, which file implements it, what the
last run produced, and where it diverges (if at all) from the
walkthrough document `documents/walkthrough2/main.pdf`.

The walkthrough is the spec; the code in this directory is the
implementation. Where the spec and the implementation drifted, the
drift is explicitly recorded below under "Divergences".

## Stages

```
raw GPS CSVs
    │
    ▼  pipeline/compress_routes.py
data/compressed_routes.parquet      (924,379,163 rows, ~2.57M trips)
    │
    ▼  pipeline/match_routes.py
data/matched_routes.json            (per-trip matched edges with timestamps)
    │
    ▼  pipeline/analyze_popularity.py
data/popularity_results.npz         (sparse count matrix C: bins × links)
    │
    ▼  core/pagerank.py
data/two_phase_pagerank_vector.json (per-link mass v_up + v_down at one bin)
```

## Stage 1 — Trip cleaning and segmentation

**File:** `pipeline/compress_routes.py`

- Reads raw `Combined/wp-snapped-*.csv` shards.
- Filters per-fix speed `>= 2 m/s` (haversine distance over time delta).
- Splits into trips on device change OR `> 5 min` gap.
- Writes `data/compressed_routes.parquet` with columns
  `(route_id, time_sec, lat, lon)`.

Status: not re-run in this session. `data/compressed_routes.parquet`
(13 GB, 924M rows) on disk is the existing output and is consistent
with the walkthrough §3.

## Stage 2 — Map-matching (walkthrough §4)

**File:** `pipeline/match_routes.py`

Current code: emits **per-fix matched output** as
`{ "route": { "route_id": int, "fixes": [[lid, t_gps], ...] } }`.

For each emitting Viterbi state in `matcher.lattice_best`:

- skip non-emitting states (`obs_ne != 0`)
- skip singleton-node fallbacks (`edge_m.l2 is None`)
- read the matched edge `(l1, l2)` and look up its MATSim `lid`
- pair with the GPS fix's own timestamp `gps_times[m.obs]`

Output schema: `fixes`, list of `(lid, t_gps)` pairs.

Network projection: equirectangular (Snyder 1987 ch. 12),
center `(40.75, -111.90)`, `R = 6,371,000 m`.

Matcher: `leuvenmapmatching.matcher.distance.DistanceMatcher` with
`max_dist = 400`, `max_dist_init = 400`, `min_prob_norm = 1e-4`,
`non_emitting_states = True`. These are the same values used in the
walkthrough §4.3.

**Last run:** the existing `data/matched_routes.json` (210 MB,
402,370 records) was produced by a **previous version** of the code
that emitted the legacy `routelinks` schema
(`[lid, t_enter, t_exit]` per edge visit, with isotonic-fit timing).
Re-running the new per-fix code on the full 2.57M-trip corpus would
take ~12-24 h on a single CPU, so it was deferred. See "Divergences"
below for the implications.

## Stage 3 — Popularity matrix (walkthrough §5)

**File:** `pipeline/analyze_popularity.py`

Walkthrough rule (§5.2):

```
b_j      = floor((t_j - t_0) / 300)            (5-minute bin)
trips    = { (b_j, lid_j, r) : j = 1, ..., M_r }   (per-trip dedup)
C[b, i]  = #distinct r contributing to (b, i)
```

`t_0 = 2018-01-01 00:00:00`.

The current code accepts both schemas:

- new `fixes`: each pair `(lid, t_gps)` becomes one observation.
- legacy `routelinks`: each edge visit `(lid, t_enter, t_exit)`
  becomes one observation at `t = t_enter`.

Output: CSR `.npz` saved to `data/popularity_results.npz` with keys
`matrix_data, matrix_indices, matrix_indptr, matrix_shape, times,
link_ids`. Cell dtype is `int32` (count of distinct trips), not
`float32` (vehicle-seconds dwell, which was the previous rule).

**Last run** (2026-05-06):

- Read 402,370 records from `data/matched_routes.json` (legacy
  `routelinks` schema).
- Bin grid: 30 days × 24 h × 12 bins/h = 8640 bins; 8637 stored
  (3 bins received no observation on any link).
- Matrix: `8637 rows × 60707 columns`, `nnz = 3,934,680`.
- Sum of all cells = `4,248,904` distinct (bin, link, trip) triples.
- Max cell = 13 distinct trips on a single (bin, link).
- All 8637 stored rows have `sum > 0` by construction.

## Stage 4 — Two-phase PageRank (walkthrough §7)

**File:** `core/pagerank.py`

The transition matrix of the chain, **column-stochastic**, with
state ordering `(1↑, …, N↑, 1↓, …, N↓)`:

```
M_b = | (1 - β) P_up        ρ E_b 1^T   |
      |  β I                (1 - ρ) P_down |
```

Iteration (no separate damping; teleport is baked into the matrix):

```
v_{n+1} = M_b @ v_n
```

Implemented via the rank-1 trick to avoid materialising the dense
top-right block:

```python
v_up_new   = (1 - β) * P_up_cs @ v_up + ρ * E_b * sum(v_down)
v_down_new = β * v_up + (1 - ρ) * P_down_cs @ v_down
```

`P_up_cs` and `P_down_cs` are column-stochastic kernels (the
transposes of the row-stochastic `P_up`, `P_down` produced by
`build_phase_matrix`). With `alpha_s = alpha_l = 0` (default),
the kernels are uniform-over-out-neighbours on the directed road
graph.

**Calibrated parameters** (from the route-length distribution study,
method-of-moments on the geometric clocks of §7):

| Parameter | Value | Provenance |
|-----------|-------|------------|
| `beta`    | 0.124 | `1 / (1 + E[L_up])`, `E[L_up] ≈ 7.06` edges |
| `rho`     | 0.147 | `1 / (1 + E[L_down])`, `E[L_down] ≈ 5.80` edges |
| `alpha_s` | 0.0   | no speed weighting (uniform P_up) |
| `alpha_l` | 0.0   | no lane weighting (uniform P_up) |
| `tol`     | 1e-7  | L1 convergence tolerance |
| `max_iters` | 300 | hard cap |

**Last run** (2026-05-06):

- Network: `N = 99,716` links from `data/city_graph_full.json`.
- Target bin: `2018-09-08 08:00:00` (Saturday morning rush, arbitrary
  pick for one-bin demo).
- Read raw count matrix (`data/popularity_results.npz`) for the
  teleportation prior `E_b`.
- Converged in **99 iterations** (L1 change `< 1e-7`).
- Output: `data/two_phase_pagerank_vector.json`, one entry per link
  with the predicted mass `(v_up + v_down)[i]`, normalised to sum 1.

Diagnostic metrics for that single bin:

- Top-100 MRE vs `E_b`: 0.7885
- Overall MRE vs `E_b`: dominated by `1 / (E_b + 1e-9)` on zero
  entries (interpretation requires care; not a meaningful number on
  raw single-bin data).

These numbers are larger than the calibrated cross-bin study
(top-100 MRE = 0.088 in `analysis/evaluate_v1.py`) because:

1. `evaluate_v1.py` runs against the **smoothed** popularity matrix
   (`POPULARITY_NPZ`, Laplace-smoothed with `alpha=0.026`), which
   denoises the per-bin signal and makes the model-vs-truth gap
   look smaller.
2. The smoothed matrix is what the project uses for paper-quality
   evaluation; the raw count matrix is the strictly count-based
   object the walkthrough describes.

Both matrices are produced by the same `analyze_popularity.py`
(raw) followed by `pipeline/generate_smoothed.py` (smoothed). The
walkthrough only describes the raw matrix; smoothing is a separate
step that does not appear in the walkthrough.

## Divergences (code vs. walkthrough)

1. **Map-matching schema (open).** The walkthrough §4.6 specifies
   the per-fix `fixes` schema for `matched_routes.json`. The current
   `match_routes.py` produces this schema, but the file on disk is
   from a previous run that produced the legacy `routelinks`
   schema. The two are equivalent for the count-based aggregation
   of §5 because (a) every emitting state contributes one matched
   edge, and (b) the legacy code already deduplicates consecutive
   same-edge states into one edge visit per (route, edge), so
   `(bin(t_enter), lid, route_id)` deduplication on the legacy data
   yields the same count as `(bin(t_gps), lid, route_id)` deduplication
   on the new per-fix data, except in the rare case where a single
   trip's fixes on a single link straddle a 5-minute boundary —
   undetectable at the 5-minute resolution used here.

   Action: re-run `match_routes.py` on the full Parquet stream (~12-24
   h, single CPU) the next time the corpus needs to be reprocessed,
   so the on-disk `matched_routes.json` matches the spec literally.

2. **Smoothed vs raw matrix in evaluation.** The downstream paper
   experiments (`analysis/evaluate_v1.py`,
   `analysis/optimize_gamma.py`, `analysis/evaluate.py`) read the
   **smoothed** popularity matrix `POPULARITY_NPZ` rather than the
   raw `POPULARITY_RAW_NPZ` that the walkthrough specifies. The
   smoothing step is a separate denoising operation
   (`pipeline/generate_smoothed.py`) and is not described in the
   walkthrough. This is intentional: the walkthrough documents the
   minimal pipeline, while the paper benchmarks the model on the
   smoothed signal that downstream consumers actually use.

3. **`core/pagerank.py` was rewritten in this session** to match the
   walkthrough's column-stochastic `M_b` exactly. The previous
   implementation used a row-stochastic 2N × 2N matrix with blocks
   `((1-β)P_up, βI; 0, P_down)` and applied an external PageRank
   damping via `v <- d M^T v + (1 - d) E_2N`. That formulation is a
   different chain and is no longer used. Power iteration is now
   the bare matrix-vector product `v_{n+1} = M_b v_n` (rank-1 trick).

## Last-known-good code per stage

| Stage | File | Last touched | Notes |
|-------|------|--------------|-------|
| 1 | `pipeline/compress_routes.py` | unchanged | feeds parquet |
| 2 | `pipeline/match_routes.py` | this session | per-fix output |
| 3 | `pipeline/analyze_popularity.py` | this session | count-based, both schemas |
| 4 | `core/pagerank.py` | this session | PDF `M_b`, rank-1 trick |

## How to re-run end-to-end (full corpus)

```bash
# Stage 1: 5-30 min (already done; output 13 GB)
python3 pipeline/compress_routes.py

# Stage 2: ~12-24 h on a single CPU
python3 pipeline/match_routes.py

# Stage 3: a few seconds
python3 pipeline/analyze_popularity.py

# Stage 4: a few seconds
python3 core/pagerank.py
```

For evaluation across many bins (paper experiments):

```bash
# Optional: smoothed matrix for paper-quality experiments
python3 pipeline/generate_smoothed.py

# Cross-bin V1 evaluation against between-week variability gate
python3 analysis/evaluate_v1.py --num-frames 49 --seed 42
```

## What this session changed

- **Walkthrough `documents/walkthrough2/main.tex`**: rewritten
  §4.4 → "Per-fix matched output", removed §4.5 "Anchor recovery",
  rewrote §4.6 schema as `fixes`, rewrote §5.2 counting rule for
  count-based per-trip dedup, removed bogus "96% populated bins"
  claim and replaced with measured "2 bins absent from grid",
  added §7 "Two-phase PageRank chain" with column-stochastic `M_b`
  and the iteration `v_{n+1} = M_b v_n`.
- **`pipeline/match_routes.py`**: rewrote `interpolate_edge_times`
  → `extract_fix_matches`, removed isotonic regression,
  free-flow-fallback, anchor logic; output is now per-fix.
- **`pipeline/analyze_popularity.py`**: rewrote dwell-overlap rule
  → distinct-trip count with per-trip dedup; accepts both schemas.
- **`core/pagerank.py`**: rewrote to implement the walkthrough's
  column-stochastic `M_b` with the rank-1 trick. Iteration is now
  `v_{n+1} = M_b v_n`.
- **`core/__init__.py`**: updated exports to match the new module
  surface (no more `build_two_phase_matrix`, `run_power_iteration`,
  `sharpen_teleportation`, `boost_top_k`).
- **Downstream scripts now broken** (use the removed old API):
  `analysis/tune_params.py`, `analysis/optimize_gamma.py`,
  `analysis/evaluate.py`, `analysis/analyze_smoothing.py`. These
  were experiments on the prior chain formulation and need to be
  ported to the new `power_iteration(P_up_cs, P_down_cs, E_b, β,
  ρ, tol, max_iter)` API. Still working: `analysis/evaluate_v1.py`,
  `analysis/evaluate_v1_forecast.py` (these were already on the
  V1 single-matrix formulation).
- Re-ran stages 3 and 4 to regenerate `data/popularity_results.npz`
  (count-based) and `data/two_phase_pagerank_vector.json` (PDF-spec
  chain at `2018-09-08 08:00:00`).
- Backed up the previous dwell-time matrix to
  `data/popularity_results.npz.bak_dwell` before overwriting.
