# Two-Phase PageRank with Travel-Time Self-Loops

## Overview
This repository contains the implementation, experiments, and mathematical justification for the **Two-Phase PageRank** algorithm, customized for predicting macroscopic traffic congestion across a massive urban road network (Salt Lake City, featuring 99,716 distinct road links). 

The algorithm bridges the gap between discrete random walks (standard PageRank) and continuous physical reality by incorporating **Travel-Time Self-Loops**, fundamentally solving the model's tendency to drastically underestimate traffic on major highways.

## The Core Problem: "The Top-100 Error"
Standard PageRank treats every transition between nodes as mathematically instantaneous. In a road network, this means a simulated vehicle transitions from a 1-kilometer highway segment to the next link in the exact same mathematical "time step" as a vehicle transitioning through a 10-meter intersection connector. 

Because it ignores **physical dwell time**, standard PageRank causes excessive network diffusion. Traffic flows through the graph too quickly and spreads uniformly, washing out the concentration peaks that naturally form on major arterial roads. As a result, the original model suffered a severe Mean Relative Error (MRE) of **0.733** on the Top-100 busiest links.

## The Solution: Physical Dwell Time ($\mu$)
To resolve this, we introduced a physical constraint to the transition matrix $P$: **Travel-Time Self-Loops**. 

For every link $i$, a self-loop is added to the transition matrix, dictating the fraction of traffic that *stays* on the link during an iteration. This self-loop weight is directly proportional to the physical time required to traverse the link:

$$ \text{self-weight}_i = \mu \cdot \frac{\text{length}_i}{\text{speed}_i} $$

Where $\mu$ (mu) is the global scaling parameter that dictates the strength of the dwell time. By setting $\mu = 20$, long highways naturally retain upwards of 99% of their probability mass per step, approximating a **continuous-time Markov chain**. Over the course of the Power Iteration, this allows realistic traffic bottlenecks to emerge purely through network topology and physical dimensions.

## 2N x 2N Block Matrix & Two Phases
To accurately capture the difference between active driving and local parking/dispersal, the network ($N$) is duplicated into two layers to form a $2N \times 2N$ block transition matrix:

1. **Up Phase (Forward Flow):** Vehicles actively driving, transitioning forward along the directed graph.
2. **Down Phase (Local Dispersal):** Vehicles parking or moving slowly in local areas. Mass transitions from the Up phase to the Down phase at a rate determined by $\beta$, but can never return to the Up phase.

## Spatial Smoothing Pre-processing
Because raw GPS trajectories are incredibly sparse, a single timeframe yields mostly 0s across the 99k links. Before PageRank runs, the raw trajectory data undergoes **Graph-Diffusion Smoothing** ($\gamma = 0.26$) using the physical adjacency matrix. This converts the sparse points into a continuous probability field, which acts as the ground-truth teleportation vector ($E_N$) for the PageRank algorithm.

## Results
The introduction of the $\mu$ travel-time parameter proved to be a fundamental, physical fix to the underlying mathematical model, yielding massive simultaneous improvements without ad-hoc parameter hacking:

*   **Top-100 MRE:** Reduced from 0.7331 $\rightarrow$ **0.0317** (95.7% Improvement)
*   **Overall MRE:** Reduced from 0.1980 $\rightarrow$ **0.0471** (76.2% Improvement)

## Documentation & Visualizations
We have recently generated extensive supporting materials inside this project:
*   **`pagerank_explanation/two_phase_pagerank.pdf`**: A fully compiled LaTeX document detailing the exact matrix mathematics, transition probabilities, and phase changes of the model.
*   **`project_visualizations/`**: Contains Python-generated graphics, including:
    *   `1_mre_comparison.png`: Accuracy improvements across strategy iterations.
    *   `2_self_loop_probability.png`: The curve of how $\mu$ affects dwell time based on physical seconds.
    *   `3_block_matrix_visualization.png`: A visual heatmap of the $2N \times 2N$ Markov transition block.
    *   `4_mu_grid_search.png`: Empirical results showing the monotonic improvement as $\mu$ scales.

## Execution
The core mathematical implementation and entry point is located at:
`scripts/pagerank_execution/run_two_phase_pagerank.py`
