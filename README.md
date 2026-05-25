# Two-Phase PageRank

Link-level traffic prediction on the Salt Lake City road network (99,716
links) from INRIX probe trajectories. A two-phase PageRank chain runs over
a graph-diffused popularity prior; chain parameters (transition weights
by road type, phase commit and restart rates) are fit on 840 (t, t+7 days)
calendar pairs and evaluated by Overall MRE.

The SIGSPATIAL paper for this project lives in
`documents/walkthrough2/main.tex`. Table 10 (the additive ladder of
modeling choices) is reproducible end-to-end via `REPRODUCE_TABLE10.md`
and `verify_table10.py`.

## Project structure

```
TwoPhase_PageRank_Project/
├── README.md                       (this file)
├── REPRODUCE_TABLE10.md            Per-row script -> input -> JSON -> value manifest
├── verify_table10.py               Hash + value verifier (10/10 rows OK)
├── EVALUATION_CONTRACT.md          Locked sampling and metric spec
├── requirements.txt                Pinned Python / numpy / scipy versions
├── config.py                       Project paths and defaults
├── .gitignore
│
├── core/                           Chain implementation (importable)
│   ├── pagerank.py                 Block-matrix PageRank iteration
│   ├── io.py                       Sparse popularity-matrix loader
│   ├── gpu_backend.py              Optional CuPy backend
│   └── __init__.py
│
├── pipeline/                       Preprocessing: raw GPS -> diffused popularity
│   ├── compress_routes.py          1. Raw GPS CSV -> Parquet
│   ├── match_routes.py             2. Map-match to road network
│   ├── analyze_popularity.py       3. Duration-weighted binning into 5-min windows
│   └── generate_smoothed.py        4. Graph-diffusion smoothing (--gamma)
│
├── run_chain_osm_subedge.py        Top-level OSM-subedge pipeline drivers
├── run_eval_osm.py
├── run_popularity_osm.py
├── run_smoothed_osm.py
│
├── analysis/                       Tuning, evaluation, and per-row Table 10 scripts
│   ├── vanilla_pr_two_alphas.py    SP at alpha in {0.05, 0.15}            (rows 1, 2)
│   ├── sp_tune_three_ov.py         SP + lanes / speed / both (Ov. MRE)    (rows 3, 4, 5)
│   ├── sp_tune_alpha005_fixed.py   SP + both, alpha=0.05 fixed            (row 6)
│   ├── tp_eval_calibrated.py       TP at (beta=0.102, rho=0.101)          (row 7)
│   ├── tp_two_betas.py             TP at beta=rho in {0.05, 0.15}         (row 8)
│   ├── tp_tune_three_ov.py         TP + lanes / speed / both              (rows 9-11)
│   ├── tp_tune_calibrated_fixed.py TP at (beta,rho) fixed, tune road-type (row 12)
│   ├── climatology_diffused_840.py Climatology baseline (mean of 3 weeks)
│   ├── noise_floor_*.py            Noise-floor estimation for the paper
│   ├── quantify_geometric_fit.py   Geometric fit for damping bound (alpha ~ 0.05)
│   ├── compute_phase_split*.py     Empirical phase-length distribution
│   ├── event_sept15_*.py           Stadium-event intervention scripts
│   └── (other tuning + plotting scripts)
│
├── data/                           Inputs
│   ├── city_graph_full.json        99,716 links, 99,681 adjacency entries
│   ├── popularity_results_osm.npz  Raw OSM popularity matrix (17 MB)
│   ├── popularity_results.npz      Raw popularity counts (17 MB)
│   ├── slc_network.xml             MATSim network for map-matching
│   ├── origins_results.npz         Per-bin origin distribution
│   └── two_phase_pagerank_vector.json
│   (the 61 MB diffused gamma=0.20 npz is NOT tracked; regenerate from
│   popularity_results_osm.npz, see Quick start below)
│
├── results/                        Table 10 outputs + per-row JSONs (verified)
│
└── documents/
    ├── README.md
    ├── notes/                      Discussion + appendix notes
    └── walkthrough2/               Current SIGSPATIAL paper
        ├── main.tex
        ├── main.pdf                (built)
        ├── figures/                Paper figures + their generator scripts
        └── results/                Per-row JSONs copied for the paper build
```

## Algorithm

Two phases run on the column-stochastic out-adjacency $P_{cs}$ of the
directed road graph. The chain iterates

$$
v_{up}^{(k+1)} = (1-\beta)\, P_{cs}\, v_{up}^{(k)} + \rho\, E_b\, \mathbf{1}^{T} v_{down}^{(k)}
$$

$$
v_{down}^{(k+1)} = \beta\, v_{up}^{(k)} + (1-\rho)\, P_{cs}\, v_{down}^{(k)}
$$

In block-matrix form,

$$
\begin{bmatrix} v_{up} \\ v_{down} \end{bmatrix}^{(k+1)}
=
\begin{bmatrix} (1-\beta)\, P_{cs} & \rho\, E_b\, \mathbf{1}^{T} \\[2pt] \beta\, I & (1-\rho)\, P_{cs} \end{bmatrix}
\begin{bmatrix} v_{up} \\ v_{down} \end{bmatrix}^{(k)}
$$

$\rho\, E_b\, \mathbf{1}^{T} v_{down}^{(k)}$ is the restart term that puts a
fraction $\rho$ of down-phase mass back into the up phase, distributed by
the diffused popularity prior $E_b$. Total mass is conserved each step:
$\|v_{up}^{(k+1)}\| + \|v_{down}^{(k+1)}\| = \|v_{up}^{(k)}\| + \|v_{down}^{(k)}\|$.

Final prediction is $v = v_{up} + v_{down}$ at convergence (L1 increment
below $10^{-6}$, capped at 200 iterations).

The single-phase baseline used throughout the paper is the standard
PageRank power iteration on the same $P_{cs}$:

$$
v^{(k+1)} = \alpha\, E_b + (1-\alpha)\, P_{cs}\, v^{(k)}
$$

## Parameters

Bounds and best-fit values from Table 10 (additive ladder on the whole
network, 840 (t, t+7 days) pairs, seed=7):

| Parameter | Bound | Best-fit | Role |
|:---|:---|:---|:---|
| `alpha` (single-phase) | $[0.01, 0.15]$ | 0.15 (pins) | Restart rate; geometric fit suggests $\alpha \approx \frac{1}{20}$ |
| `beta` (two-phase) | $[0.01, 0.15]$ | 0.102 / 0.15 | Up-phase commit rate, equals $\frac{1}{1+E[L_{up}]}$ |
| `rho`  (two-phase) | $[0.01, 0.15]$ | 0.101 / 0.15 | Down-phase restart rate, equals $\frac{1}{1+E[L_{down}]}$ |
| `alpha_s` | unbounded | $+2.7$ (SP), $+2.8$ (TP) | Speed-limit weighting exponent on $P_{cs}$ |
| `alpha_l` | unbounded | $+0.2$ to $+0.3$ | Lane-count weighting exponent on $P_{cs}$ |
| `gamma` | (data prep) | 0.20 | Graph-diffusion smoothing of the popularity prior |

## Pipeline

```
Raw GPS (CSV)
    |
    v
[pipeline/compress_routes.py]                Clean, filter speed < 2 m/s,
    |                                        split on 5-min gaps
    v
Compressed routes (Parquet)
    |
    v
[pipeline/match_routes.py]                   Leuven map matching to the road network
    |
    v
Map-matched routes (JSON)
    |
    v
[pipeline/analyze_popularity.py]             Duration-weighted binning into
    |                                        5-min windows (8,640 bins)
    v
Raw popularity matrix (NPZ, 8640 x 99716)
    |
    v
[pipeline/generate_smoothed.py --gamma 0.20] Graph-diffusion smoothing
    |
    v
Diffused popularity (NPZ)         <- both input and target for Table 10
    |
    v
[core/pagerank.py]                Two-phase PageRank
[analysis/vanilla_pr_*.py]        Single-phase baselines
[analysis/sp_tune_*.py]           Road-type tuning (alpha_s, alpha_l)
[analysis/tp_tune_*.py]
    |
    v
Per-row JSON in results/ -> Table 10 in documents/walkthrough2/main.tex
```

## Quick start

```bash
git clone git@github.com:Hamidmath/TPPRT.git
cd TPPRT
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Regenerate the 61 MB diffused input from the raw OSM counts (1-3 min):
python pipeline/generate_smoothed.py \
    --input  data/popularity_results_osm.npz \
    --output data/popularity_results_smoothed_osm_gamma020.npz \
    --gamma  0.20

# Verify every cell of Table 10 matches the paper:
python verify_table10.py
```

`REPRODUCE_TABLE10.md` documents the per-row script, input, JSON output,
and expected MRE for every cell, plus the sampling spec (seed=7, 840
pairs, 120 per weekday, $\epsilon = 10^{-6}$) and the L-BFGS-B settings.

## Data coverage

- **Source**: INRIX probe data (possibly volunteer drivers through insurance).
- **Period**: August 31, 2018 19:00 -> September 30, 2018 18:45.
- **Resolution**: 5-minute bins, 8,640 timeframes (8,637 active after
  removing three overnight zero-trip bins).
- **Network**: 99,716 road links, 99,681 adjacency entries, Salt Lake City
  bounding box (about 80% of raw GPS points fall outside this bbox and are
  excluded).
