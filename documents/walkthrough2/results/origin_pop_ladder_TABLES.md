# Origin -> popularity ladder: training and held-out test tables

This file records the within-week origin -> popularity experiments
(diffused trip-origins at slot t as input, diffused popularity at the
same slot t as target). Chain dynamics are FROZEN at the geometric-fit
values throughout: SP alpha = 0.0508; TP-new beta = 0.102, rho = 0.101.
Only alpha_s, alpha_l, alpha_ang are tuned per row. No retraining.

Diffusion (both for origins input and popularity target): Laplace
pseudo-count alpha = 0.01 plus one-step graph diffusion with gamma =
0.20, matching pipeline/generate_smoothed.py.

Angle feature per directed adjacency edge (i -> j):
    f_ang(i,j) = 1 + cos(theta(i,j)) + 0.05,
    transition weight gets multiplied by f_ang^alpha_ang.
cos theta is the dot product of the unit direction vectors of links i
and j taken from data/slc_network.xml.

## 1. 840-slot training ladder (sample_pairs seed=7, BPW=120)

All 12 rows below trained on the same 840 (t, t+7d) source slots t. The
diffused popularity at slot t is the target; the input is diffused
origins at the same slot. Loss = mean Ov.MRE over the 840 slots.

| chain  | mode          | alpha_s | alpha_l | alpha_ang | Ov.MRE mean | Ov.MRE median |
|--------|---------------|--------:|--------:|----------:|------------:|--------------:|
| SP     | uniform_prior |     --  |     --  |       --  |      1.2670 |        1.0312 |
| SP     | diffuse_prior |     --  |     --  |       --  |      1.2135 |        0.9989 |
| SP     | lanes_only    |     --  |  +0.744 |       --  |      1.1977 |        0.9873 |
| SP     | speed_only    |  +2.528 |     --  |       --  |      1.1922 |        0.9832 |
| SP     | both          |  +1.980 |  +0.475 |       --  |      1.1862 |        0.9788 |
| SP     | ang_only      |     --  |     --  |    -6.252 |      1.1895 |        1.0180 |
| SP     | speed_ang     |  +5.579 |     --  |    -2.300 |      1.1817 |        1.0088 |
| SP     | both_ang      |  +4.595 |  +0.929 |    -2.447 |      1.1802 |        1.0080 |
| TP-new | uniform_prior |     --  |     --  |       --  |      1.2662 |        1.0307 |
| TP-new | diffuse_prior |     --  |     --  |       --  |      1.2074 |        0.9953 |
| TP-new | lanes_only    |     --  |  +0.730 |       --  |      1.1927 |        0.9846 |
| TP-new | speed_only    |  +2.478 |     --  |       --  |      1.1872 |        0.9810 |
| TP-new | both          |  +1.943 |  +0.455 |       --  |      1.1821 |        0.9769 |
| TP-new | ang_only      |     --  |     --  |    -3.037 |      1.1841 |        1.0113 |
| TP-new | speed_ang     |  +5.176 |     --  |    -2.046 |      1.1736 |        1.0009 |
| TP-new | both_ang      |  +4.372 |  +0.790 |    -2.169 |  **1.1722** |        1.0004 |

Best 840-slot row: TP-new both_ang at mean Ov.MRE = 1.1722.

## 2. Held-out 200-slot test (last 7 days), seed=99 — tuned rows (no straightness)

Test slots: 200 random 5-min bins from
[2018-09-23 19:00:00 .. 2018-09-30 18:55:00], seed=99. Trained alphas
from section 1 are applied with chain dynamics frozen; no retraining.

| chain  | mode       | alpha_s | alpha_l | alpha_ang | train mean | train median | test mean | test median |
|--------|------------|--------:|--------:|----------:|-----------:|-------------:|----------:|------------:|
| SP     | lanes_only |     --  |  +0.744 |       --  |     1.1977 |       0.9873 |    1.2586 |      1.0988 |
| SP     | speed_only |  +2.528 |     --  |       --  |     1.1922 |       0.9832 |    1.2518 |      1.0933 |
| SP     | both       |  +1.980 |  +0.475 |       --  |     1.1862 |       0.9788 |    1.2450 |      1.0875 |
| SP     | ang_only   |     --  |     --  |    -6.252 |     1.1895 |       1.0180 |    1.2453 |      1.1091 |
| TP-new | lanes_only |     --  |  +0.730 |       --  |     1.1927 |       0.9846 |    1.2535 |      1.0966 |
| TP-new | speed_only |  +2.478 |     --  |       --  |     1.1872 |       0.9810 |    1.2468 |      1.0916 |
| TP-new | both       |  +1.943 |  +0.455 |       --  |     1.1821 |       0.9769 |    1.2410 |      1.0867 |
| TP-new | ang_only   |     --  |     --  |    -3.037 |     1.1841 |       1.0113 |**1.2401** |      1.1029 |

Script: analysis/test_last7days_200slots.py.
JSON:   results/origin_pop_ladder/test_last7days_200.json.

## 3. Held-out 200-slot test (last 7 days), seed=99 — STRAIGHTNESS rows

Same 200 slots as section 2 (test_seed=99). These are the six rows that
include the angle parameter alpha_ang.

| chain  | mode       | alpha_s | alpha_l | alpha_ang | train mean | train median | test mean | test median |
|--------|------------|--------:|--------:|----------:|-----------:|-------------:|----------:|------------:|
| SP     | ang_only   |     --  |     --  |    -6.252 |     1.1895 |       1.0180 |    1.2453 |      1.1091 |
| SP     | speed_ang  |  +5.579 |     --  |    -2.300 |     1.1817 |       1.0088 |    1.2371 |      1.1007 |
| SP     | both_ang   |  +4.595 |  +0.929 |    -2.447 |     1.1802 |       1.0080 |    1.2354 |      1.0996 |
| TP-new | ang_only   |     --  |     --  |    -3.037 |     1.1841 |       1.0113 |    1.2401 |      1.1029 |
| TP-new | speed_ang  |  +5.176 |     --  |    -2.046 |     1.1736 |       1.0009 |    1.2288 |      1.0930 |
| TP-new | both_ang   |  +4.372 |  +0.790 |    -2.169 | **1.1722** |   **1.0004** |**1.2272** |  **1.0921** |

TP-new both_ang sweeps every column. Train-to-test gap is ~+0.055 mean
/ +0.09 median, uniform across rows. Angle exponent is consistently
negative; its magnitude shrinks once speed is also tuned, so a portion
of the "favor non-straight" signal is already captured by the speed
weighting.

Script: analysis/test_last7days_200_straightness.py.
JSON:   results/origin_pop_ladder/test_last7days_200_straightness.json.

## 4. Full-corpus train + test (rows finishing on CHPC)

Train slots: every 5-min bin in [2018-08-31 19:00 .. 2018-09-23 18:55]
(6,624 slots). Test slots: every 5-min bin in
[2018-09-23 19:00 .. 2018-09-30 18:55] (2,016 slots). Per row, L-BFGS
trains alpha_s / alpha_l / alpha_ang on the 6,624 train slots and the
trained alphas are then applied to all 2,016 test slots.

Finished so far (4 of 12):

| chain  | mode       | alpha_s | alpha_l | TRAIN mean | TRAIN median | TEST mean | TEST median |
|--------|------------|--------:|--------:|-----------:|-------------:|----------:|------------:|
| SP     | speed_only |  +2.242 |     --  |     1.1096 |       0.9179 |    1.1963 |      1.0109 |
| SP     | lanes_only |     --  |  +0.673 |     1.1136 |       0.9209 |    1.2018 |      1.0153 |
| SP     | both       |  +1.751 |  +0.436 |     1.1045 |       0.9138 |    1.1903 |      1.0057 |
| TP-new | speed_only |  +2.201 |     --  |     1.1049 |       0.9162 |    1.1915 |      1.0084 |

Remaining 8 jobs (still running, ~3.5 h elapsed at last check): TP
lanes_only, TP both, SP/TP lanes_ang, SP/TP speed_ang, SP/TP both_ang.
The angle-aware rows will land last because of the per-edge angle
factor in the kernel build.
