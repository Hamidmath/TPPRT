# Two-Phase PageRank with Travel-Time Self-Loops

## Overview
Predicts **link-level traffic congestion** across a massive urban road network (Salt Lake City, 99,716 road links) using a modified PageRank algorithm. The key innovation is **Travel-Time Self-Loops**: each link retains probability mass proportional to its physical traversal time, transforming the discrete random walk into an approximation of a continuous-time Markov chain.

## The Core Problem
Standard PageRank treats every link transition as instantaneous. A 1 km highway segment and a 10 m intersection connector receive identical treatment per iteration step. This causes excessive diffusion, flattening traffic peaks on major roads. The original model suffered a **Top-100 MRE of 0.733** on the busiest links.

## The Solution
A self-loop is added to each link, weighted by its traversal time:

$$\text{self\_weight}_i = \mu \cdot \frac{\text{length}_i}{\text{speed}_i}$$

With $\mu = 20$, long highway segments retain 99%+ of their mass per step, naturally concentrating traffic where vehicles actually spend time.

## Results

| Metric | Before ($\mu=0$) | After ($\mu=20$) | Improvement |
|:---|:---|:---|:---|
| Top-100 MRE | 0.7331 | **0.0317** | 95.7% |
| Overall MRE | 0.1980 | **0.0471** | 76.2% |

Evaluated on 49 random timeframes from September 2018 (seed=42).

## Project Structure

```
TwoPhase_PageRank_Project/
│
├── config.py                       # Centralized paths and default parameters
│
├── core/                           # Core algorithm (importable package)
│   ├── __init__.py                 #   Re-exports all public functions
│   ├── pagerank.py                 #   Two-Phase PageRank implementation
│   └── friction.py                 #   Alternative friction-based model
│
├── scripts/
│   ├── pipeline/                   # Data processing pipeline
│   │   ├── compress_routes.py      #   1. Raw GPS CSV -> Parquet
│   │   ├── match_routes.py         #   2. Map-match GPS to road network
│   │   ├── analyze_routes.py       #   3. Route statistics & histograms
│   │   ├── analyze_popularity.py   #   4. Bin traversals into 15-min windows
│   │   ├── generate_smoothed.py    #   5. Graph-diffusion smoothing (--gamma)
│   │   └── create_population.py    #   Zip code population data
│   │
│   ├── tuning/                     # Parameter search experiments
│   │   ├── tune_mu.py              #   Grid search: mu x damping x beta
│   │   ├── tune_tau.py             #   Grid search: tau x damping
│   │   ├── tune_tau_fine.py        #   Fine-grained tau search
│   │   ├── tune_strategies.py      #   Multi-strategy comparison (A-F)
│   │   ├── tune_combos.py          #   Combination strategies (G-J)
│   │   ├── tune_final.py           #   Final beta x boost fine-tuning
│   │   ├── tune_top_100.py         #   Alpha_s/alpha_l for top-100
│   │   └── friction_tuning.py      #   Friction self-loop experiment
│   │
│   ├── evaluation/                 # Evaluation scripts
│   │   ├── evaluate_timeframes.py  #   49-timeframe evaluation
│   │   ├── evaluate_gamma_026.py   #   Evaluate with optimal smoothing
│   │   ├── run_49_overall_mre.py   #   Overall MRE baseline
│   │   └── run_baseline_extreme.py #   Extreme parameter baseline
│   │
│   ├── comparison/                 # Comparison scripts
│   │   ├── compare_baselines.py    #   Baseline vs. friction model
│   │   ├── compare_top_100.py      #   Detailed top-100 link analysis
│   │   ├── compare_smoothing_datasets.py  # Raw vs. smoothed datasets
│   │   ├── compare_predictive_weeks.py    # Cross-week prediction
│   │   └── compare_raw_vs_smoothed.py     # Smoothing effect analysis
│   │
│   └── smoothing_analysis/         # Smoothing parameter optimization
│       ├── cross_val_smoothing.py  #   Monte Carlo cross-validation
│       ├── verify_smoothing.py     #   Independent verification (seed=123)
│       └── plot_results.py         #   Smoothing analysis figures
│
├── data/                           # Input and output data
│   ├── city_graph_full.json        #   Road network (99,716 links)
│   ├── slc_network.xml             #   OSM network for map-matching
│   ├── popularity_results.npz      #   Raw popularity matrix (2,880 x N)
│   ├── popularity_results_smoothed.npz  # Smoothed (gamma=0.26)
│   ├── two_phase_pagerank_vector.json   # Final PageRank output
│   └── zipcode_population.json     #   Census population data
│
├── results/                        # Experiment output logs
│   ├── mu_tuning_results.txt       #   120-config grid search
│   ├── strategy_tuning_results.txt #   Strategy A-F comparison
│   ├── combo_tuning_results.txt    #   Combination strategies
│   └── ...                         #   Other tuning results
│
├── figures/                        # All generated figures
│   ├── project/                    #   MRE comparison, mu effect, etc.
│   ├── doc/                        #   Publication figures (16 figs)
│   │   └── generate_figures.py
│   ├── smoothing/                  #   Cross-validation plots
│   └── generate_project_figures.py #   Project figure generation
│
├── docs/                           # Documentation
│   ├── algorithm.md                #   Algorithm specification
│   ├── self_loop_report.md         #   Full travel-time self-loop report
│   ├── top_k_changelog.md          #   Parameter search changelog
│   ├── smoothing_justification.md  #   Smoothing parameter analysis
│   ├── traffic_events.md           #   Sept 2018 traffic events
│   └── issue_report/               #   Development issue documentation
│
├── latex/                          # LaTeX papers
│   ├── two_phase_pagerank.tex/pdf  #   Main algorithm paper
│   └── eigenvector_equivalence.tex/pdf  # Eigenvector proofs
│
└── presentation/                   # Beamer slides
    ├── slides.tex
    └── slides.pdf
```

## Data Pipeline

```
Raw GPS (CSV)
    |
    v
[compress_routes.py] -- Clean, filter speed < 2 m/s, split on 5-min gaps
    |
    v
Compressed Routes (Parquet)
    |
    v
[match_routes.py] -- Leuven Map Matching to road network
    |
    v
Matched Routes (JSON)
    |
    v
[analyze_popularity.py] -- Bin into 15-minute time windows
    |
    v
Raw Popularity Matrix (NPZ, 2880 x N)
    |
    v
[generate_smoothed.py --gamma 0.26] -- Graph-diffusion smoothing
    |
    v
Smoothed Popularity (NPZ) --> Teleportation vector E_N for PageRank
```

## Algorithm

The network is duplicated into a **2N x 2N block matrix** with two phases:

$$M_{2N} = \begin{bmatrix} (1 - \beta) \cdot P & \beta \cdot I \\ 0 & P \end{bmatrix}$$

where $P$ includes travel-time self-loops on its diagonal. Power iteration:

$$v^{(k+1)} = d \cdot M_{2N}^T \cdot v^{(k)} + (1 - d) \cdot E_{2N}$$

Final output: $v_{final} = v_{up} + v_{down}$, renormalized.

## Parameters

| Parameter | Value | Role |
|:---|:---|:---|
| `mu` | 20.0 | Travel-time self-loop scaling |
| `damping` | 0.80 | Markov chain vs. teleportation weight |
| `beta` | 0.9 | Up-to-down phase transition rate |
| `gamma` | 0.26 | Graph-diffusion smoothing factor |

## Quick Start

```bash
# Run the core PageRank algorithm
python core/pagerank.py

# Run with different smoothing
python scripts/pipeline/generate_smoothed.py --gamma 0.10

# Evaluate on 49 random timeframes
python scripts/evaluation/evaluate_timeframes.py
```

## Data Coverage
- **Period**: August 31, 2018 19:00 -- September 30, 2018 18:45
- **Resolution**: 15-minute bins (2,880 timeframes)
- **Network**: 99,716 road links, Salt Lake City
