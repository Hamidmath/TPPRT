# Link-level anomaly attribution to the Utah State Fair 2018

This document records exactly how we move from a per-slot week-over-week
MRE blow-up (W2 -> W3 in `top24_anomalies_thick.json`) to a per-link
attribution of the Fair lift on a small, geographically coherent
region of the road graph. Everything written here is reproducible with
two scripts:

```
python3 analysis/fair_link_anomaly.py
python3 analysis/fair_region_profile.py
```

All outputs land in `documents/walkthrough2/results/anomalies/fair/`.

## 1. Why we focus on the Fair window

The W2 -> W3 anomaly is the strongest in the corpus (top-thick MRE
cutoff 0.683 vs ~0.53 for W1->W2 and W3->W4). The cause is the
**Utah State Fair, Sep 6-16, 2018**, held at the Fairpark adjacent to
I-15 Exit 309 (see [FINDING_W2W3.md](../FINDING_W2W3.md)). All 24
top-thick slots fall on **Tue-Thu Sep 11-13** of W2 (the days the
Fair gates opened at NOON instead of 10 a.m.), comparing to the
same Tue-Thu of W3.

We isolate the Fair signal at the **link** level so we can talk about
*which roads* surged, not just *which time slots*.

## 2. Inputs

| input | path | notes |
|---|---|---|
| raw map-matched counts | `data/popularity_results_osm.npz` | int32 CSR, (8640, 60069) |
| road graph | `data/city_graph_full.json` | per-link speed/lanes/length/direction; OSM adjacency |

The `vector` field in each link record is a 2-D unit direction vector
(not coordinates). It lets us tag a link as roughly NS or EW but does
not pin a geographic position.

## 3. Defining the windows

- **Fair window**: Tue Sep 11, Wed Sep 12, Thu Sep 13, 2018, full 24 h.
  These are exactly the W2 weekdays present in the top-24 anomaly list.
- **Reference weeks**: the same Tue-Wed-Thu of weeks 1, 3, and 4:
  - W1: Sep 4, 5, 6 (Fair had not started yet)
  - W3: Sep 18, 19, 20 (Fair had ended Sun Sep 16)
  - W4: Sep 25, 26, 27 (well after the Fair)

W3 is included as a reference because its same-weekday counts give a
direct read of post-Fair normal traffic. Using the median over three
reference weeks makes the comparison robust to any one week being
unusual (Yom Kippur on W3, for instance).

## 4. Per-link statistics

For each of the 60,069 map-matched links *i*:

```
fair_count(i)     = sum over all 5-min bins t in Fair window of c_t(i)
ref_count_W1(i)   = same sum over W1 reference days
ref_count_W3(i)   = same sum over W3 reference days
ref_count_W4(i)   = same sum over W4 reference days
ref_median(i)     = median { ref_count_W1, ref_count_W3, ref_count_W4 }
lift(i)           = fair_count(i) - ref_median(i)
fold(i)           = fair_count(i) / max(ref_median(i), 1)
```

The full per-link table (60,069 rows) is `per_link.csv`. It also
records `ref_mean`, `ref_min`, `ref_max` so a reader can audit the
spread across reference weeks.

### Why two rankings (lift and fold)

- `lift` (absolute) picks links whose **total trip surplus** is
  largest. These tend to be already-busy roads -- freeways and major
  arterials -- where even a modest percentage lift represents many
  trips.
- `fold` (relative) picks links whose **proportional** surge is
  largest. These tend to be small streets that normally carry a
  trickle and saw a Fair-related spike.

The two rankings answer different questions: "where did the most
Fair traffic go?" vs "which link's day looked the least normal?".

## 5. Floors and filters

We apply minimum reference floors to avoid noise:

- For the **lift histogram** and the lift ranking we require
  `ref_median(i) >= 5`. This drops 43,325 links that are essentially
  off-network in normal weeks (16,744 of 60,069 links remain).
- For the **fold ranking** we additionally require
  `ref_median(i) >= 20`. Without this, a link going from 1 to 5 trips
  would dominate the fold ranking with `fold = 5`, which is not
  meaningful traffic.

No floor is applied to the per-link CSV itself; all 60,069 links are
present and can be re-ranked.

## 6. Outputs at this stage

`fair_link_anomaly.py` produces:

| file | contents |
|---|---|
| `per_link.csv` | full table for all 60,069 links |
| `top_lift.csv` | top 200 by absolute lift (ref_median >= 5) |
| `top_fold.csv` | top 200 by fold change (ref_median >= 20) |
| `hist_lift.png` | distribution of lift over the 16,744 with ref >= 5 |
| `hist_fold.png` | distribution of fold over the same set, log x |
| `summary.json` | counts, thresholds, totals, lift/fold percentiles |

Headline numbers from `summary.json`:

- 60,069 links total; 16,744 pass `ref_median >= 5`.
- Total Fair-window trips: **1,856,177**.
  Reference totals: W1 1,776,736; W3 1,683,761; W4 1,837,302.
- Median per-link lift: **+1.0** trips. p95 lift: **+32**. p99 lift: not
  exposed but the top is +330.
- Median fold: **1.04** (essentially no change). p95 fold: **1.67**.
  p99 fold: **2.33**.

So the typical link sees a tiny +4% Fair lift; the right tail is where
the action is.

## 7. Region detection: connected components on the top-lift set

A handful of links spiking is interesting only if they form a coherent
**corridor**. We test this directly with the graph adjacency.

Steps:

1. Take the top 200 links by absolute lift (`top_lift.csv`).
2. Build the undirected restriction of the road graph adjacency
   to these 200 nodes.
3. Run a BFS-based connected-component search.
4. Keep components with `>= 10` links.

Implemented in `fair_region_profile.py`. Result:

- **Component 0**: 39 links, total Fair trips 65,381, total lift +6,143,
  mean lift +158/link. Mix of orientations: 21 NS-ish + 17 EW-ish.
  Speed range 13.9-31.3 m/s, lanes 2-6, length 115-2104 m. This is the
  largest coherent region; the mixed orientation suggests a freeway
  mainline plus its on/off ramps and a perpendicular arterial -- the
  pattern expected at an interchange.
- **Component 1**: 21 links, 57,851 Fair trips, +3,033 lift.
  12 NS + 6 EW; same speed/lane profile. A second freeway/arterial
  pair.
- **Component 2**: 13 links, 61,202 Fair trips, +1,971 lift. **Purely
  NS-oriented** (13/13), median 6 lanes, median length 1,273 m, all at
  freeway speed. This is the textbook signature of a single
  freeway mainline -- almost certainly I-15 through Salt Lake.

Total: **73 freeway/major-arterial links** in the three large
components carry the bulk of the Fair lift in the top-200. This is
within the user-requested "10-300 link region" target.

The full per-component table with all link IDs is
`region_components.json`; the 39-link largest component (with link IDs
and per-link lift) is `region_top_links.csv`.

## 8. Time-of-day profile for the region

For the 39-link largest component we sum trip counts per 5-min bin
across the component, separately on:

- the 3 Fair days (Tue/Wed/Thu Sep 11-13), averaged per weekday;
- each reference week's 3 Tue-Thu, averaged per weekday;

then take the mean +/- 1 standard deviation across the three reference
weeks. Plot is `tod_profile.png`; underlying data is `tod_profile.csv`.

The Fair curve sits above the reference band **throughout the daytime**
with two clear peaks. The 12 largest per-bin lifts (per weekday) are:

| time | Fair trips/day | reference trips/day | lift |
|---|---|---|---|
| 08:25 | 200.0 | 102.9 | **+97.1** |
| 14:45 | 168.7 | 106.3 | +62.3 |
| 16:55 | 180.3 | 119.8 | +60.6 |
| 08:30 | 182.0 | 128.9 | +53.1 |
| 11:20 | 163.3 | 110.3 | +53.0 |
| 13:50 | 156.0 | 104.1 | +51.9 |
| 10:55 | 154.3 | 103.1 | +51.2 |
| 15:35 | 161.0 | 113.0 | +48.0 |
| 12:30 | 145.0 | 97.1 | +47.9 |
| 08:20 | 159.7 | 111.9 | +47.8 |
| 15:55 | 137.7 | 91.0 | +46.7 |
| 09:25 | 165.3 | 119.0 | +46.3 |

The 08:00-09:00 spike is consistent with **vendors, exhibitors, and
livestock handlers arriving before the Fair opens**. The 14:00-17:00
spike is consistent with **public attendees arriving for the
noon-opening day**, which Sep 11/12/13 uniquely featured (the rest of
the Fair opened at 10 a.m.). The lift remains positive between the
peaks, reflecting steady afternoon traffic on the corridor.

## 9. Outputs at this stage

`fair_region_profile.py` produces:

| file | contents |
|---|---|
| `region_components.json` | all 59 components from top-200 lift links, sorted by size |
| `region_top_links.csv` | the 39 links in the largest component with per-link lift/fold |
| `tod_profile.png` | time-of-day plot, Fair vs reference, for the 39-link region |
| `tod_profile.csv` | the 288 per-bin numbers behind the plot |

## 10. Reproducibility

```
python3 analysis/fair_link_anomaly.py     # ~30 s, writes per_link.csv etc.
python3 analysis/fair_region_profile.py   # ~20 s, writes tod_profile.png etc.
```

Both scripts are deterministic. No random seeds are involved at any
point (no sampling, no optimization), so the outputs are
bit-for-bit reproducible from the same `popularity_results_osm.npz`
and `city_graph_full.json`.

## 11. What we cannot say from this alone

- **Which freeway and which exit.** Component 2 is almost certainly
  I-15 mainline; Component 0 is almost certainly an interchange around
  Exit 309 (the Fairpark exit). But the road graph stores only
  direction vectors, not coordinates or street names. Mapping these
  link IDs to a geographic basemap would settle it.
- **Whether the local-street spikes in the fold ranking are
  Fair parking access.** The top-fold links (47879, 81261, 73962,
  21068, ...) are 1-3 lane city streets that went from ~20 trips on
  normal Tue/Wed/Thu to ~50+ during the Fair. Without coordinates we
  cannot confirm they are within walking distance of the Fairpark,
  though their attributes (low speed, short, single-lane) are exactly
  what residential or commercial side streets look like.

Both questions become trivial if we have access to a geographic
mapping for the OSM link IDs in `city_graph_full.json`.
