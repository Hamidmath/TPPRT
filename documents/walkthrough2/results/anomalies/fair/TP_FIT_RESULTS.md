# Old-form TP fit on the Fair-anomaly window

Record of all old-form TP optimization runs aimed at fitting the Utah
State Fair anomaly. Each run uses the **diffused -> diffused** protocol
(input = diffused popularity at t, target = diffused popularity at
t+7d, same as Table 10), the **old-form (classical PageRank) TP chain**

```
v_{k+1} = (1-alpha) M_{2N} v_k + alpha E_{2N},  E_{2N} = [E_b, 0]
```

L-BFGS-B with eps=2e-3, ftol=1e-6, gtol=1e-4, multiple starts. Loss is
average Overall MRE over the listed pairs, restricted to the listed
region.

## 1. Round 1: 39-link Fairpark interchange region, 4 params

Region: largest connected component of top-200 lift links
(component 0 of `region_components.json`). Mixed freeway + ramps +
arterial, total length ~33.9 km.

Parameters tuned: alpha, beta, alpha_s, alpha_l. Bounds: alpha, beta
in [0.01, 0.15]; alpha_s, alpha_l unbounded. 5 starts, maxiter=20.

| run | hour | best alpha | best beta | alpha_s | alpha_l | region MRE (mean) | region MRE (median) | full OV |
|---|---|---|---|---|---|---|---|---|
| Sep 11 16:00-17:00 | afternoon Fair-going | 0.010 | 0.010 | -6.11 | +2.39 | **3.329** | 3.506 | 1.395 |
| Sep 12 08:00-09:00 | morning vendors/handlers | 0.010 | 0.014 | -191.77 | -129.37 | **2.158** | 1.385 | 1.700 |

Both bad. The afternoon job had 4/5 starts converge to the same
optimum; the morning job's alpha_s / alpha_l blew up to numerical
drift (~-100s). Region was too broad for the chain to fit under the
smoothed prior.

Scripts: `analysis/tp_old_fair_region_hour.py`,
`tp_old_fair_region_hour_W08.py`.
Outputs: `results/tp_old_fair_region_hour.json`,
`tp_old_fair_region_hour_W08.json`.

## 2. Round 2: 7-link tight sub-region (~1 km radius), 4 params

Region: tight sub-cluster of 7 links found by Dijkstra on the
length-weighted undirected link graph, all within 1 km road-graph
distance of center link 59063. All surface streets (30-50 mph, 2-6
lanes, 115-578 m segments). See `tight_region.json` for the link list
and `analysis/fair_pick_tight_region.py` for the selection script.

Tight links: `59063, 77400, 77395, 85822, 77405, 59055, 3614`.

Same parameters tuned and bounds as Round 1. 5 starts, maxiter=20.

| run | hour | best alpha | best beta | alpha_s | alpha_l | region MRE (mean) | region MRE (median) | full OV |
|---|---|---|---|---|---|---|---|---|
| Sep 11 16:00-17:00 | afternoon | 0.010 | 0.010 | -9.06 | -302 (drift) | **0.634** | 0.568 | 1.617 |
| Sep 12 08:00-09:00 | morning | 0.010 | 0.010 | -1.05 | -10.3 (drift) | **0.581** | 0.540 | 1.523 |

**~5x improvement** vs Round 1 in both cases:
3.329 -> 0.634 (afternoon), 2.158 -> 0.581 (morning).

Observations:

- alpha and beta both pin at the lower bound 0.01 in every start.
  Chain wants minimum damping and minimum up-phase commit.
- alpha_s negative in both: down-weight high-speed links. The
  surviving mass flows through low-speed surface streets, consistent
  with the Fairpark-adjacent street grid.
- alpha_l drifts to large negative values (-10 to -302) in every
  Sep 11 start. All 5 Sep 11 starts return f=0.6340 with alpha_l in
  [-58, -302] -- a flat direction at the optimum. Once alpha_l is
  "negative enough" the loss saturates and L-BFGS-B stops improving.
- Full-network OV stays ~1.5-1.6: the model is overfitting the 7-link
  region at the cost of everything else. Expected, since the loss
  only scores those 7 links.

Scripts: `analysis/tp_old_fair_tight_hour.py`,
`tp_old_fair_tight_hour_W08.py`.
Outputs: `results/tp_old_fair_tight_hour.json`,
`tp_old_fair_tight_hour_W08.json`.

## 3. Round 3: 7-link tight + 5 categorical road weights

Same 7-link region and same Sep 11 16:00-17:00 pairs, but the speed
exponent alpha_s is replaced by **5 unbounded categorical multipliers**
(one per OSM-derived speed bucket):

```
edge weight = exp(w_cat[ category(link) ]) * lanes^alpha_l
```

with the 5 categories:

| cat | OSM class | speed bucket | links in graph |
|---|---|---|---|
| 0 | motorway / freeway | >= 26.82 m/s (60+ mph) | ~447 |
| 1 | arterial / primary | 17.88-26.82 m/s (40-55 mph) | ~5,220 |
| 2 | secondary / collector | 13.41-17.88 m/s (30-35 mph) | ~56,164 |
| 3 | residential | 11.18-13.41 m/s (25 mph) | ~29,173 |
| 4 | local / service | < 11.18 m/s (<= 20 mph) | ~9,379 |

8 parameters total (alpha, beta, alpha_l, w_0, w_1, w_2, w_3, w_4).
Bounds: alpha, beta in [0.01, 0.15], everything else unbounded. 8
starts, maxiter=30.

**Result: region MRE = 0.6349.** Essentially the same as Round 2's
4-param fit (0.6340). The extra 4 parameters did not improve the loss.

Best params (start 5 of 8):

| param | value |
|---|---|
| alpha | 0.0100 (lower bound) |
| beta | 0.0100 (lower bound) |
| alpha_l | -16.18 (drifting, flat direction) |
| w_motorway | -0.08 (no signal) |
| w_arterial | -2.57 (push down 40-55 mph) |
| w_secondary | +0.01 (no signal) |
| w_residential | +1.57 (boost 25 mph streets) |
| w_local | +1.07 (boost <=20 mph streets) |

All 8 starts converge to f in [0.6349, 0.6357] but with wildly
different category-weight values -- the loss surface is nearly flat
across the 5-D w_cat space. The directional signal is consistent
though: down-weight arterials, up-weight residential + local.

Script: `analysis/tp_old_fair_tight_cats.py`.
Output: `results/tp_old_fair_tight_cats.json`.

## 3b. Summary across all three rounds

| run | params | region MRE (mean) | region MRE (median) | full OV |
|---|---|---|---|---|
| Sep 11 16-17, 39-link, 4 params | 4 | 3.329 | 3.506 | 1.395 |
| Sep 11 16-17, 7-link tight, 4 params | 4 | **0.634** | 0.568 | 1.617 |
| Sep 11 16-17, 7-link tight, **categorical** | **8** | **0.635** | 0.569 | 1.596 |
| Sep 12 08-09, 39-link, 4 params | 4 | 2.158 | 1.385 | 1.700 |
| Sep 12 08-09, 7-link tight, 4 params | 4 | **0.581** | 0.540 | 1.523 |

The 5x improvement from 39-link to 7-link is real. The 8-param
categorical did not push further on the tight region.

## 4. Round 4: alpha_len (length exponent) and reference-busy top-20

Edge weight extended to:

```
weight(link) = (speed/mean)^alpha_s * (lanes/mean)^alpha_l * (length/mean)^alpha_len
```

Length is a new third feature. Two regions tested:

- **104-link 1 km area** (`area_1km_all.json`) -- all road links within
  1 km road-graph distance of center 59063.
- **20-link top-busy union** (`region_top20plus7.json`) -- the 20
  busiest links in the 1 km area, ranked by sum of diffused popularity
  over Tue-Thu of W1+W3+W4 (the 3 non-Fair weeks). All 7 anomaly-lift
  links are already in that top 20, so the union is exactly 20.

All runs: Sep 11 16:00-17:00 (12 pairs), diffused -> diffused, old-form
TP, L-BFGS-B with 5-6 starts, eps=2e-3, ftol=1e-6, gtol=1e-4,
maxiter=30. alpha, beta in [0.01, 0.15]; everything else unbounded.

| run | links | params | region MRE (mean) | median | full OV | best params |
|---|---|---|---|---|---|---|
| 104-link area, 4p (baseline)  | 104 | 4 | 1.893 | 1.814 | 1.758 | a_s=+90, a_l=-30 |
| 104-link area, **+alpha_len** | 104 | **5** | **1.623** | 1.585 | 2.128 | a_s=-4.9, a_l=-0.9, **a_len=+13.9** |
| 20-link top-busy, 4p          | 20  | 4 | 0.907 | 0.883 | 1.525 | a_s=+11.8, a_l=-4.8 |
| 20-link top-busy, **+alpha_len** | **20** | **5** | **0.739** | **0.728** | 2.116 | a_s=+5.5, a_l=-12.2, **a_len=+8.75** |

**Key reads from Round 4:**

1. **alpha_len is the first added parameter that helps.** -14% MRE on
   the 104-link area (1.89 -> 1.62), -19% on the 20-link top-busy
   (0.91 -> 0.74). The earlier categorical-cats parameter expansion
   did not.
2. **alpha_len strongly positive** (+14 on 104-link, +8.75 on 20-link):
   the chain wants to up-weight longer links. Long roads in this area
   are arterials / freeway, which physically carry through-traffic, so
   this matches intuition.
3. **alpha_s flips sign when alpha_len is present.** Length and speed
   are correlated (long links tend to be high-speed arterials), so
   alpha_len absorbs much of the work alpha_s was doing. With
   alpha_len present, alpha_s drops from +90 to -4.9 (104-link) and
   from +11.8 to +5.5 (20-link).
4. **Top-20 reference-busy is the right region.** Going 104 -> 20
   drops MRE by ~half (1.62 -> 0.74) at the same parameter count.
   The 84 quiet residential side streets in the 104-link area carry
   mostly noise, not signal.
5. **alpha = beta = 0.01 pinned everywhere.** This finding is now
   bulletproof across every Fair-region run.
6. **Trade-off: full-network OV degrades when alpha_len helps the
   region.** 1.76 -> 2.13 (104), 1.53 -> 2.12 (20). Expected, since
   the loss only scores the region.

## Best Fair-anomaly fit to date

**20-link region + 5 params (alpha, beta, alpha_s, alpha_l, alpha_len),
MRE = 0.739 (mean), 0.728 (median)** on Sep 11 16:00-17:00 against Sep
18 16:00-17:00 (diffused -> diffused). Comparable to the Table 10
ladder's best of 0.856 Overall MRE on the full 840-pair eval, but on
a Fair-anomaly hour and on a small busy region rather than the whole
network.

The alpha_len finding motivates adding the same parameter to the full
Table 10 ladder (next round).

## 4. Reference: Table 10 best on the full 840-pair eval

For scale, on the standard 840-pair Table 10 protocol (whole network,
not Fair-restricted), the best old-form TP achieves Overall MRE
**0.856**. The Round-2 region MRE of 0.58-0.63 on the tight Fair
region is in the same ballpark, but on a much smaller (7-link) loss.

## 5. What this tells us so far

- The diffused-to-diffused protocol can fit a 1 km region of the Fair
  anomaly to ~0.6 MRE if the region is tight and the road-type weights
  are pushed strongly toward low-speed streets.
- It cannot fit a 33 km mixed freeway+arterial region (MRE 2-3).
- alpha and beta both want their lower bound: damping does not help
  here.
- alpha_l is unidentifiable past a saturation threshold; could
  reasonably be bounded for stability without losing fit.

Open questions still being tested:

- Does the categorical-weight model (Round 3) drop MRE further?
- Does the same fit hold for the morning hour, or does that need
  different category weights?
- Would raw-to-raw (skipping diffusion on both sides) recover the
  full Fair-time anomaly better? Not yet attempted.

## Files in this folder relevant to TP fitting

- `region_components.json` -- 39-link region used in Round 1
- `tight_region.json` -- 7-link tight region used in Rounds 2-3
- (parent results dir) `tp_old_fair_region_hour.json` -- Round 1 afternoon
- (parent results dir) `tp_old_fair_region_hour_W08.json` -- Round 1 morning
- (parent results dir) `tp_old_fair_tight_hour.json` -- Round 2 afternoon
- (parent results dir) `tp_old_fair_tight_hour_W08.json` -- Round 2 morning
- (parent results dir) `tp_old_fair_tight_cats.json` -- Round 3 (pending)
