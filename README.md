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
├── config.py                        # Centralized paths and default parameters
│
├── core/                            # Core algorithm (importable package)
│   ├── __init__.py                  #   Re-exports all public functions
│   └── pagerank.py                  #   Two-Phase PageRank implementation
│
├── scripts/
│   ├── pipeline/                    # Data processing pipeline
│   │   ├── compress_routes.py       #   Step 1: Raw GPS CSV -> Parquet
│   │   ├── match_routes.py          #   Step 2: Map-match GPS to road network
│   │   ├── analyze_popularity.py    #   Step 3: Bin traversals into 15-min windows
│   │   ├── generate_smoothed.py     #   Step 4: Graph-diffusion smoothing (--gamma)
│   │   └── analyze_routes.py        #   Data statistics & diagnostics
│   │
│   ├── tuning/                      # Parameter optimization
│   │   └── tune_params.py           #   Grid search: mu x damping x beta
│   │
│   ├── evaluation/                  # Model evaluation
│   │   └── evaluate.py              #   Comprehensive multi-metric evaluation
│   │
│   └── smoothing_analysis/          # Smoothing justification
│       ├── analyze_smoothing.py     #   Why smooth, how much, what impact
│       └── optimize_gamma.py        #   Cross-validation to find optimal gamma
│
├── data/                            # Input and output data
│   ├── city_graph_full.json         #   Road network (99,716 links)
│   ├── slc_network.xml              #   OSM network for map-matching
│   ├── compressed_routes.parquet    #   Cleaned GPS trajectories
│   ├── matched_routes.json          #   Map-matched routes
│   ├── popularity_results.npz       #   Raw popularity matrix (2,880 x N)
│   ├── popularity_results_smoothed.npz  # Smoothed (gamma=0.26)
│   └── two_phase_pagerank_vector.json   # Final PageRank output
│
├── results/                         # Experiment output logs & JSON
│
├── figures/                         # All generated figures
│   ├── project/                     #   MRE comparison, mu effect, etc.
│   ├── doc/                         #   Publication figures (16 figs)
│   │   └── generate_figures.py
│   ├── analysis/                    #   Data analysis plots
│   ├── evaluation/                  #   Evaluation plots
│   ├── smoothing/                   #   Legacy smoothing plots
│   ├── smoothing_analysis/          #   Smoothing justification & gamma CV plots
│   └── generate_project_figures.py  #   Project figure generation
│
├── docs/                            # Documentation
│   ├── algorithm.md                 #   Algorithm specification
│   ├── self_loop_report.md          #   Full travel-time self-loop report
│   ├── top_k_changelog.md           #   Parameter search changelog
│   ├── smoothing_justification.md   #   Smoothing parameter analysis
│   ├── traffic_events.md            #   Sept 2018 traffic events
│   └── issue_report/                #   Development issue documentation
│
├── latex/                           # LaTeX papers
│   ├── two_phase_pagerank.tex/pdf   #   Main algorithm paper
│   └── eigenvector_equivalence.tex/pdf  # Eigenvector proofs
│
└── presentation/                    # Beamer slides
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
    |
    v
[core/pagerank.py] -- Two-Phase PageRank with travel-time self-loops
    |
    v
Stationary Distribution (JSON)
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

# Comprehensive evaluation (49 random timeframes)
python scripts/evaluation/evaluate.py

# Quick evaluation (10 frames)
python scripts/evaluation/evaluate.py --num-frames 10

# Parameter tuning
python scripts/tuning/tune_params.py --num-frames 10
python scripts/tuning/tune_params.py --mu 10,15,20,25 --damping 0.75,0.80,0.85

# Smoothing justification
python scripts/smoothing_analysis/analyze_smoothing.py
python scripts/smoothing_analysis/optimize_gamma.py

# Generate smoothed data with different gamma
python scripts/pipeline/generate_smoothed.py --gamma 0.10

# Data analysis and statistics
python scripts/pipeline/analyze_routes.py
```

## Data Coverage
- **Period**: August 31, 2018 19:00 -- September 30, 2018 18:45
- **Resolution**: 15-minute bins (2,880 timeframes)
- **Network**: 99,716 road links, Salt Lake City
