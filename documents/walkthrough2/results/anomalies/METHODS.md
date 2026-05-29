# Week-over-week anomaly detection

Goal: surface 5-min slots whose link-level trip pattern is the most
unlike the same slot one week earlier. We do this for three consecutive
week pairs and report the top-24 most anomalous slots per pair.

## 1. Data

- **Source file**: `data/popularity_results_osm.npz`
  - raw map-matched trip counts (no diffusion, no smoothing)
  - matrix shape: `(8640, 60069)`, `int32`, CSR
  - rows = 5-min bins, columns = OSM link IDs
  - time range: `2018-08-31 19:00:00` to `2018-09-30 18:55:00`
    (30 calendar days)
- **Granularity**: one entry `c_t(i)` = number of map-matched trips
  whose peak link at bin `t` was link `i`. Map-matching uses the
  Newson-Krumm hidden Markov algorithm via the Leuven library.
- We do **not** use the diffused popularity. Diffusion would smear
  zeros across neighbors and hide the very anomalies we want.

## 2. Slot pairing

We define 5-min slots per week as `7 days * 24 h * 12 bins = 2016`.
The 8640-bin corpus contains:

- bins 0..2015  = week 1 (Fri Aug 31 19:00 ... Fri Sep 7 18:55)
- bins 2016..4031 = week 2
- bins 4032..6047 = week 3
- bins 6048..8063 = week 4
- bins 8064..8639 = leftover 2 days (unused)

For each comparison `wb -> wt`, every slot `k in [0, 2016)` is paired:

```
t_base   = times[wb * 2016 + k]
t_target = times[wt * 2016 + k]
```

so `t_target = t_base + 7 days` exactly, and the day-of-week and
time-of-day align by construction.

Three comparisons reported: `W1->W2`, `W2->W3`, `W3->W4`.

## 3. Per-slot MRE

For a single slot pair we compute

```
S(t)   = { links i : c_{t_base}(i) >= FLOOR }       # baseline-active links
MRE(t) = (1 / |S(t)|) * sum_{i in S(t)} |c_{t_target}(i) - c_{t_base}(i)|
                                              / c_{t_base}(i)
```

with `FLOOR = 5` trips. The floor is critical: without it, a link going
from 1 trip to 3 trips gives MRE = 2.0 and dominates the ranking; freeway
links with 1000 -> 1100 give MRE = 0.10 and are ignored. The floor keeps
us at links that carry real traffic in the baseline week.

Notes:
- We mask on the **baseline** week, not the target. A link active in the
  target but zero in the baseline does **not** enter `S(t)`.
- Because `c_{t_base}(i) >= 5`, the denominator is bounded away from
  zero; no epsilon is needed.
- Slots with `|S(t)| = 0` (no baseline-active links at all) are
  excluded from the comparison. These are overnight bins, ~400-540 per
  pair out of 2016.

## 4. Histograms

For each comparison we write two PNGs in this folder.

- `hist_{w1w2,w2w3,w3w4}.png` -- full distribution of valid MRE values
  (one bar per log-spaced bin). Vertical lines: median, p95, p99.
- `hist_{w1w2,w2w3,w3w4}_thick.png` -- same histogram restricted to
  slots with `|S(t)| >= 50` baseline-active links. Adds a purple
  dash-dot line at the top-24 MRE cutoff for that distribution.

The thick variant exists because the full histogram has a hard pile-up
at MRE = 1.0: many slots have only a handful of baseline-active links,
and when all of them drop to zero in the target they tie at 1.0. The
thick filter removes this artifact.

## 5. Top-24 ranking

We rank slots by MRE descending and keep the top 24. We write **two**
rankings to `top24_anomalies_thick.json`:

- `top_unfiltered`: top 24 over all valid slots (subject to ties at 1.0)
- `top_thick`: top 24 over slots with `|S(t)| >= 50` baseline-active
  links. This is the ranking to look at for genuine network-wide events.

The legacy `top24_anomalies.json` contains the unfiltered ranking only,
kept for reference.

## 6. Reproducibility

```bash
python3 analysis/anomaly_week_pairs.py
```

Outputs in `documents/walkthrough2/results/anomalies/`:

```
hist_w1w2.png             hist_w1w2_thick.png
hist_w2w3.png             hist_w2w3_thick.png
hist_w3w4.png             hist_w3w4_thick.png
top24_anomalies.json      top24_anomalies_thick.json
METHODS.md
```

Script knobs (top of `analysis/anomaly_week_pairs.py`):

```
FLOOR             = 5      # min baseline trips per link to enter S(t)
MIN_LINKS_RANKED  = 50     # min |S(t)| to be eligible for top_thick
SLOTS_PER_WEEK    = 2016
TOP_K             = 24
```

Wall time on a single core: about 90 seconds.

## 7. Headline results

| comparison | n_valid | median | p95 | p99 | top-24 cutoff (thick) |
|---|---|---|---|---|---|
| W1 -> W2 | 1474 | 0.467 | 0.906 | 1.000 | 0.531 |
| W2 -> W3 | 1594 | 0.486 | 0.886 | 1.000 | 0.683 |
| W3 -> W4 | 1630 | 0.474 | 0.856 | 1.000 | 0.535 |

W2 -> W3 has the heaviest right tail and the highest top-24 cutoff,
making it the most anomalous comparison of the three.
