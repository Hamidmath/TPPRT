# Two-Phase PageRank

## Overview
Predicts **link-level traffic congestion** across a massive urban road network (Salt Lake City, 99,716 road links) using a two-phase PageRank chain calibrated from probe GPS trajectories. The chain decomposes a single random walk into a peak-seeking up phase and a destination-seeking down phase, with geometric phase lengths whose parameters are fit from the empirical trip-length distribution.

## Project Structure

```
TwoPhase_PageRank_Project/
│
├── Makefile                         # Reproducible build: make all | pipeline | docs
├── README.md                        # (this file)
├── config.py                        # Centralized paths and default parameters
│
├── core/                            # Two-Phase PageRank algorithm (importable)
│   ├── __init__.py
│   ├── pagerank.py                  #   2N x 2N block-matrix PageRank
│   └── gpu_backend.py               #   optional CuPy backend
│
├── pipeline/                        # Raw data -> teleportation vector (run in order)
│   ├── compress_routes.py           #   1. Raw GPS CSV -> Parquet
│   ├── match_routes.py              #   2. Map-match GPS to road network
│   ├── analyze_popularity.py        #   3. Duration-weighted binning into 15-min windows
│   └── generate_smoothed.py         #   4. Graph-diffusion smoothing (--gamma)
│
├── analysis/                        # Evaluation and tuning
│   ├── evaluate.py                  #   49-frame multi-metric evaluation
│   ├── analyze_smoothing.py         #   Why smooth, how much, what impact
│   └── optimize_gamma.py            #   Cross-validation for gamma
│
├── data/                            # Inputs + derived artefacts
│   ├── city_graph_full.json         #   Road network (99,716 links)
│   ├── slc_network.xml              #   MATSim network for map-matching
│   ├── matched_routes.json          #   Map-matched trips
│   ├── compressed_routes.parquet    #   Cleaned GPS trajectories
│   ├── popularity_results.npz       #   Raw popularity matrix (float, duration-weighted)
│   ├── popularity_results_smoothed_026.npz
│   ├── two_phase_pagerank_vector.json
│   └── *.pre_duration_split.*       #   pre-fix backups (entry-only semantics)
│
├── results/                         # JSON / text metrics (evaluate.py, tune_params.py)
│
├── figures/                         # Regenerable figures (PDF only)
│   ├── paper/                       #   generate_figures.py writes here
│   │   ├── fig{1..16}_*.pdf
│   │   └── generate_figures.py
│   ├── smoothing/                   #   analyze_smoothing + optimize_gamma plots
│   └── evaluation/                  #   evaluate.py plots (summary + temporal)
│
└── documents/                       # All human-authored artefacts
    ├── README.md                    #   index + figure-search-order rules
    │
    ├── paper/                       #   Main publication draft
    │   ├── main.tex
    │   ├── references.bib
    │   ├── main.pdf                 #     (built)
    │   └── figures/                 #     paper-only assets (non-regenerable PNGs
    │                                #      shared with slides / combined note)
    │
    ├── walkthrough/                 #   Step-by-step pipeline description
    │   ├── main.tex
    │   └── main.pdf                 #     (built)
    │
    ├── notes/                       #   Short technical notes (one folder per note)
    │   └── two_phase_pagerank_combined/
    │       ├── two_phase_pagerank_combined.tex
    │       └── two_phase_pagerank_combined.pdf
    │
    ├── presentation/                #   Beamer slides
    │   ├── slides.tex
    │   └── slides.pdf               #     (built)
    │
    └── reports/                     #   Long-form markdown reports
        ├── algorithm.md
        ├── self_loop_report.md
        ├── smoothing_justification.md
        ├── traffic_events.md
        ├── issue_report/
        └── data/                    #     JSON artefacts cited by reports + figures
            ├── smoothing_grid_search_results.json
            └── verification_results.json
```

See `documents/README.md` for figure-search-order rules (paper and slides share
`figures/paper/` via `\graphicspath`), per-document build commands, and
guidance on where to put new artefacts.

### Untracked scratch directories
`experiments/`, `scripts/`, and `temp_crash_analysis/` at the top level are
user-owned working directories, gitignored, and not part of the reproducible
build. Move anything worth keeping into the relevant canonical folder
(`analysis/`, `pipeline/`, `documents/reports/`) when it stabilises.

## Data Pipeline

```
Raw GPS (CSV)
    |
    v
[pipeline/compress_routes.py] -- Clean, filter speed < 2 m/s, split on 5-min gaps
    |
    v
Compressed Routes (Parquet)
    |
    v
[pipeline/match_routes.py] -- Leuven Map Matching to road network
    |
    v
Matched Routes (JSON)
    |
    v
[pipeline/analyze_popularity.py] -- Bin into 15-minute time windows
    |
    v
Raw Popularity Matrix (NPZ, 2880 x N)
    |
    v
[pipeline/generate_smoothed.py --gamma 0.26] -- Graph-diffusion smoothing
    |
    v
Smoothed Popularity (NPZ) --> Teleportation vector E_N for PageRank
    |
    v
[core/pagerank.py] -- Two-Phase PageRank
    |
    v
Stationary Distribution (JSON)
```

## Algorithm

The network is duplicated into a **2N x 2N block matrix** with two phases:

$$M_{2N} = \begin{bmatrix} (1 - \beta) \cdot P & \beta \cdot I \\ 0 & P \end{bmatrix}$$

where $P$ is the row-stochastic out-adjacency on the directed road graph (no self-loops). Power iteration:

$$v^{(k+1)} = d \cdot M_{2N}^T \cdot v^{(k)} + (1 - d) \cdot E_{2N}$$

Final output: $v_{final} = v_{up} + v_{down}$, renormalized.

## Parameters

| Parameter | Value | Role |
|:---|:---|:---|
| `beta` | 0.102 | Up-phase commit rate (1 / (1 + E[L_up])) |
| `rho`  | 0.101 | Down-phase restart rate (1 / (1 + E[L_down])) |
| `gamma` | 0.20 | Graph-diffusion smoothing factor |
| `alpha` | 0.01 | Laplace pseudo-count |

## Quick Start

```bash
# Build everything (pipeline + evaluation + figures + all PDFs)
make all

# Or individual stages:
make pipeline       # popularity -> smoothing -> PageRank vector
make evaluate       # 49-frame evaluation metrics
make figures        # regenerate paper figures (PDF only)
make docs           # build paper, walkthrough, slides, combined note

# Single-document builds
make paper
make walkthrough
make slides
make note

# Cleanup
make clean          # remove derived data, keep source and built PDFs
make distclean      # also wipe built PDFs and LaTeX aux files

# Ad-hoc variants (skip Make)
python pipeline/generate_smoothed.py --gamma 0.10
python analysis/evaluate.py --num-frames 10
```

## Data Coverage
- **Period**: August 31, 2018 19:00 -- September 30, 2018 18:45
- **Resolution**: 15-minute bins (2,880 timeframes)
- **Network**: 99,716 road links, Salt Lake City
