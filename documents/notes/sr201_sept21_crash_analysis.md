# SR-201 Sept 21 2018 13:00 westbound crash — running notebook

Last updated: 2026-05-28.

This file records the full state of the WB SR-201 crash intervention so the
discussion can be resumed without losing context. Cross-reference:
documents/notes/professor_discussion_2026_05_15.md (general guidance) and
the recording in
documents/walkthrough2/results/prof_ladder_and_event15_TABLES.md
(background eval and Sep 11 vs Sep 18 event-direction).

## 1. Event

Reported westbound SR-201 crash at "just before 13:00" on Friday 2018-09-21.
Comparison week: Friday 2018-09-28, same hour.

Time window of interest: 13:00 to 14:55 (24 five-minute slots), split into
W1 = 13:00-13:55 (immediate aftermath) and W2 = 14:00-14:55 (recovery).

## 2. SR-201 geography used in scripts

SR-201 is the 2100 South Freeway, lat ~ 40.706 to 40.712, lon -111.91 (I-15)
to -112.13 (Magna). Filters used throughout:

- WB corridor (G1 in scripts): lat 40.700-40.715, lon -112.15 to -111.85,
  freespeed >= 20 m/s, edge orientation x2 < x1.
- WB + EB SR-201 (G2): same bbox without the direction filter.
- corridor + neighbors (G3): lat 40.695-40.725, lon -112.20 to -111.80,
  freespeed >= 15 m/s, any direction.

In the raw popularity file (60,069 OSM links), G1 = 23 WB links.
In city_graph_full.json (99,716 links), G1 = 34 WB links (more on/off
ramps survive).

## 3. Raw-count evidence (sr201_perslot_neighbors.py, sr201_sept21_event.py)

Whole network 13:00-14:55: Sep21 = 86,754, Sep28 = 82,992, delta = +4.5%.
On the 4,028 links with Sep28 >= 5, Overall MRE = 0.364, signed mean -0.016.

By window:

| window | whole net | WB SR-201 | EB SR-201 |
|---|---|---|---|
| W1 (13:00-13:55) | +3.7% | -6.9% (-26)  | +13.7% (+42)  |
| W2 (14:00-14:55) | +5.4% | +15.8% (+60) | +31.7% (+93)  |
| WA (13:00-14:55) | +4.5% | +4.5% (+34)  | +22.5% (+135) |

WB SR-201 going DOWN while the rest of the network is UP is the closure
signature. By W2 the WB corridor has overshot to +16% (queue discharging).

Per-slot WB G1 highlights (delta vs rel):

- 13:00: shock dip -19, -47%.
- 13:45 / 13:50 / 13:55: -21 / -12 / -16; second drop wave.
- 14:05-14:35: queue-release spikes +11 / +31 / +18 / +35 / +30.
- 14:40-14:55: secondary decay -21 / -26 / -10 / -8.

Top WB drop links in W1 (Sep28 >= 5, sorted by |rel|):

| link | lat | lon | fs | sum21 | sum28 | delta | rel |
|---|---|---|---|---|---|---|---|
| 74339 | 40.7084 | -111.9805 | 22.4 | 8 | 17 | -9 | -0.53 |
| 83106 | 40.7039 | -111.9800 | 22.4 | 7 | 13 | -6 | -0.46 |
| 74247 | 40.7116 | -111.9814 | 22.4 | 7 | 13 | -6 | -0.46 |
| 87803 | 40.7024 | -111.8714 | 20.1 | 5 | 9  | -4 | -0.44 |
| 74249 | 40.7150 | -111.9835 | 22.4 | 9 | 16 | -7 | -0.44 |

Closure-candidate filter (Sep21 = 0 in >= 3 of 24 slots where Sep28 >= 2):

- 58979 at (40.7105, -111.9529), fs = 31.3 (WB SR-201 mainline approach to I-15)
- 74249 at (40.7150, -111.9835), fs = 22.4 (Redwood Rd cluster)

## 4. Intervention plan (from professor 2026-05-28 transcript)

Three event categories the paper will cover:

1. Football game = demand change (new E_b), graph unchanged.
2. 9th & 9th street festival = graph change + demand change.
3. Crash (SR-201) = graph change only ("doesn't change why people are driving").

Crash protocol (no retraining; trained dynamics stay fixed):

- L1: last-week-as-is. Predict Sep 21 popularity = Sep 14 popularity.
- L2: chain on ORIGINAL graph, input = Sep 14 diffused popularity,
  compare prediction against Sep 21 actual.
- L3: chain on SURGERY graph (closed links removed from city_graph_full.json),
  same input, same target.

Calibrated parameters used (no retrain):

- SP: alpha = 0.0508, alpha_s = 3.0504 (from sp_prof_ladder.json speed_only row).
- TP-new: beta = 0.102, rho = 0.101, alpha_s = 3.0523
  (from tp_new_prof_ladder.json speed_only row).
- alpha_l = 0, alpha_len = 0 (professor: "lanes also isn't doing very much.
  the speed is basically all we need").

Closure sets tested:

- A focused: ['58979']  (WB SR-201 mainline approach to I-15)
- B broad:   ['58979', '74247', '74249', '31585', '83108']
  (mainline + Redwood Rd cluster)

Windows: W1, W2, WA.
Regions: R1 whole net, R2 WB SR-201 (34 links in city graph),
R3 corridor + neighbors (1,534 links in city graph).

## 5. Surgery results (analysis/sr201_crash_surgery.py)

Headline: surgery reduces WB-corridor MRE by 22-23% in W1 with no
retraining. The improvement is localized to the corridor; R1 and R3
move slightly the wrong way (chain over-spreads on the wider region).

### W1 (13:00-13:55)

| prediction | R1 net | R2 WB | R3 nbr |
|---|---|---|---|
| L1 last-week-as-is | 1.028 | 12.837 | 4.294 |
| L2 SP orig | 1.626 | 12.747 | 5.985 |
| L3 SP surgery-A | 1.640 | 10.903 | 5.976 |
| L3 SP surgery-B | 1.641 | **9.927** | 6.002 |
| L2 TP orig | 1.559 | 12.805 | 5.895 |
| L3 TP surgery-A | 1.570 | 10.896 | 5.885 |
| L3 TP surgery-B | 1.572 | **9.875** | 5.906 |

### W2 (14:00-14:55)

| prediction | R1 net | R2 WB | R3 nbr |
|---|---|---|---|
| L1 last-week-as-is | 1.015 |  9.086 | 4.519 |
| L2 SP orig | 1.600 | 10.038 | 6.114 |
| L3 SP surgery-B | 1.616 |  9.420 | 6.177 |
| L2 TP orig | 1.533 |  9.958 | 6.035 |
| L3 TP surgery-B | 1.547 |  9.321 | 6.092 |

### WA (13:00-14:55)

| prediction | R1 net | R2 WB | R3 nbr |
|---|---|---|---|
| L1 last-week-as-is | 1.022 | 10.962 | 4.406 |
| L2 SP orig | 1.613 | 11.392 | 6.050 |
| L3 SP surgery-B | 1.629 |  9.674 | 6.090 |
| L2 TP orig | 1.546 | 11.382 | 5.965 |
| L3 TP surgery-B | 1.559 |  9.598 | 5.999 |

### What the numbers say

1. Surgery reduces WB-corridor MRE by 22% (SP) and 23% (TP-new) in W1.
   Broader closure (B, 5 links) beats focused (A, 1 link) by ~10%.
2. Surgery effect concentrates in W1; W2 effect is 6-7%.
3. R1 (whole net) and R3 (neighbors) don't benefit. Surgery doesn't degrade
   them by more than 1%, but L1 still beats the chain there.
4. The chain alone (L2) underperforms L1 globally — same finding as the
   840-pair background eval. Chain over-spreads on diffused popularity.

## 6. Open work for next session

- Try surgery-C that ALSO removes the cross-streets / ramps near 58979,
  per professor's "remove that link and the links that cross it".
- Make a 2-panel figure: predicted vs actual heat map, original vs surgery,
  bbox = the WB SR-201 corridor.
- Run the equivalent intervention for the 9th & 9th street festival
  (Sep 15 closure on 900 S between 700 E and 1100 E).
- Decide which window (W1, WA) is the headline for the paper.

## 7. Files involved

Scripts:

- analysis/sr201_sept21_event.py     -- pre/post window comparison + per-slot trace
- analysis/sr201_perslot_neighbors.py -- per-slot trace G1/G2/G3, link-level drops
- analysis/sr201_crash_surgery.py    -- L1/L2/L3 SP/TP comparison across windows and regions

Inputs:

- data/popularity_results_osm.npz                       -- raw per-slot counts (60,069 links)
- data/popularity_results_smoothed_osm_gamma020.npz     -- diffused popularity (chain input/target)
- data/slc_network.xml                                  -- network geometry for masks
- data/city_graph_full.json                             -- adjacency + speed/lane attributes for chain
- results/sp_prof_ladder.json                           -- trained alpha_s for SP
- results/tp_new_prof_ladder.json                       -- trained alpha_s for TP-new
