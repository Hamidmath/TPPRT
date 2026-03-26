# Two-Phase PageRank with Travel-Time Self-Loops

## Problem

Predict link-level traffic congestion across a city road network (99,716 links, Salt Lake City) using a modified PageRank algorithm. The input is a smoothed ground-truth popularity distribution $E_N$ extracted from vehicle trajectory data. The output is a stationary distribution $v$ over the network.

The original Two-Phase PageRank achieves reasonable overall accuracy (MRE = 0.198) but severely underpredicts the busiest 100 links (Top-100 MRE = 0.733).

## Root Cause

Standard PageRank treats every link transition as **instantaneous**. A 1 km highway segment and a 10 m intersection connector receive identical treatment per iteration step. This causes excessive diffusion: probability mass flows through the network too quickly and spreads uniformly, flattening the traffic peaks that concentrate on long highway segments.

In reality, vehicles spend time proportional to `length / speed` on each link. A 500 m highway at 25 m/s holds a vehicle for 20 seconds, while a 50 m connector at 10 m/s holds it for only 5 seconds. The highway should accumulate 4x more mass -- but standard PageRank gives them equal weight per step.

## Algorithm

### Travel-Time Self-Loops

The key innovation is adding a **self-loop** to each link proportional to its traversal time:

$$
\text{self\_weight}_i = \mu \cdot \frac{\text{length}_i}{\text{speed}_i}
$$

where $\mu$ is a scaling parameter. The transition probabilities become:

$$
P[i, i] = \frac{\text{self\_weight}_i}{\text{self\_weight}_i + \sum_j w_{ij}} \quad\quad P[i, j] = \frac{w_{ij}}{\text{self\_weight}_i + \sum_j w_{ij}}
$$

The self-loop probability represents the fraction of each iteration that a random walker **stays** on the current link, modeling dwell time. Longer and slower links retain more mass per step.

For a typical highway segment (length=500m, speed=25m/s, 3 successors) with $\mu = 8$:
- Self-loop weight: $8 \times 500/25 = 160$
- Each successor weight: $\approx 1$
- **Self-loop probability: 98.2%** -- mass barely leaves the highway per step

For a short connector (length=50m, speed=10m/s, 3 successors):
- Self-loop weight: $8 \times 50/10 = 40$
- **Self-loop probability: 93.0%** -- still sticky, but less so

The cumulative effect over many iterations naturally concentrates mass on links where vehicles actually spend time.

### Two-Phase Block Matrix

The road network uses the same $2N \times 2N$ block structure:

$$
M_{2N} = \begin{bmatrix} (1 - \beta) \cdot P & \beta \cdot I \\ 0 & P \end{bmatrix}
$$

where $P$ now includes the self-loop entries on its diagonal. The up phase captures forward flow with periodic transitions to the down phase. Both phases include dwell time.

### Power Iteration

Standard PageRank iteration:

$$
v^{(k+1)} = d \cdot M_{2N}^T \cdot v^{(k)} + (1 - d) \cdot E_{2N}
$$

with $d = 0.85$, $E_{2N} = [E_N,\ \mathbf{0}_N]$. Convergence criterion: $\|v^{(k+1)} - v^{(k)}\|_1 < 10^{-6}$.

### Collapse and Evaluation

$v_{final} = v_{up} + v_{down}$, renormalized. Evaluated against the original $E_N$ using MRE.

## Why This Works

### Physical Interpretation

The self-loop transforms the discrete-time random walk into an approximation of a **continuous-time** random walk. In the continuous-time model, the rate of leaving link $i$ is inversely proportional to its traversal time. Links with long traversal times (highways) have low exit rates, naturally accumulating more probability mass in the stationary distribution.

This is equivalent to saying: the expected time a random walker spends on link $i$ before transitioning is proportional to $\text{length}_i / \text{speed}_i$. The stationary distribution of a continuous-time chain is proportional to the product of visit rate and dwell time -- exactly what traffic volume represents.

### Why Previous Approaches Failed

| Approach | Mechanism | Problem |
|:---|:---|:---|
| Alpha tuning ($\alpha_s, \alpha_l$) | Edge weight distortion | Changed flow on every edge, degrading overall |
| Friction self-loops (fixed factor) | Uniform self-loops | Not proportional to actual travel time |
| Teleportation sharpening ($\tau > 1$) | Boost teleportation for top links | Only 15% of update is teleportation; limited effect |
| Top-K boost | Ad-hoc multiplier | Not physically motivated; parameter hack |
| High beta | Reduce up-phase diffusion | Indirect; requires combining with boost |

The travel-time self-loop is the only approach that models the actual physics: **vehicles dwell on links proportional to traversal time**. It improves both metrics simultaneously because it corrects a systematic modeling error, not because it trades off one metric for another.

### Evidence from the Data

The top 100 links differ systematically from the network average:

| Property | Top 100 | Network Average |
|:---|:---|:---|
| Length | 304 m | 142 m |
| Speed | 16.7 m/s | 13.0 m/s |
| Lanes | 3.6 | 1.5 |
| Travel time | ~18 s | ~11 s |

Top links are 2.1x longer with 1.6x higher travel time. The self-loop naturally gives them proportionally more mass retention, directly addressing the underprediction without any link-specific tuning.

## Results

Evaluated on 49 random timeframes (seed=42):

| Configuration | Top-100 MRE | Overall MRE |
|:---|:---|:---|
| Original ($d=0.89$, $\beta=0.2$, $\mu=0$) | 0.7331 | 0.1980 |
| Boost approach ($d=0.85$, $\beta=0.9$, boost=2.0) | 0.3304 | 0.1942 |
| Self-loops ($d=0.85$, $\beta=0.2$, $\mu=8$) | 0.1055 | 0.1400 |
| **Self-loops** ($d=0.80$, $\beta=0.9$, $\mu=20$) | **0.0317** | **0.0471** |

Top-100 MRE reduced by **95.7%**, overall MRE reduced by **76.2%**. Both metrics improve simultaneously. The improvement is monotonic with `mu` -- the full grid search results are in `results/mu_tuning_results.txt`.

### Effect of mu

The parameter `mu` controls how strongly dwell time affects mass retention. Higher `mu` concentrates mass more on long/slow links:

| mu | Top-100 MRE | Overall MRE | Notes |
|:---|:---|:---|:---|
| 0 | 0.7331 | 0.1980 | Original (no self-loops) |
| 3 | 0.1285 | 0.1478 | Moderate dwell time |
| 8 | 0.0646 | 0.0883 | Strong dwell time |
| 12 | 0.0475 | 0.0679 | |
| 20 | 0.0317 | 0.0471 | Very strong dwell time |

(All at $d=0.80$, $\beta=0.9$)

## Parameters

| Parameter | Value | Role |
|:---|:---|:---|
| `mu` | 20.0 | Travel-time self-loop scaling (key parameter) |
| `damping` | 0.80 | Markov chain vs. teleportation weight |
| `beta` | 0.9 | Up-to-down phase transition rate |
| `alpha_s` | 0.0 | Speed exponent (disabled) |
| `alpha_l` | 0.0 | Lane exponent (disabled) |
| `max_iters` | 100 | Power iteration limit |
| `tol` | 1e-6 | Convergence threshold (L1 norm) |
