# Fair anomaly attribution: summary report

This is the report Claude produced after running
`fair_link_anomaly.py` + `fair_region_profile.py` and writing
`METHODS_FAIR.md`. Kept here so the chain of reasoning is preserved
even outside the chat transcript.

## The Fair anomaly region: three connected components, 73 freeway/arterial links

After ranking all 60,069 links by **lift = Fair-window count - median(W1, W3, W4 same Tue-Thu)**,
the top 200 break into 3 large connected components (using the graph
adjacency restricted to that set):

| component | n links | total Fair trips | total lift | mean lift/link | profile |
|---|---|---|---|---|---|
| 0 | **39** | 65,381 | **+6,143** | +158 | 21 NS + 17 EW, lanes 2-6, speeds 13.9-31.3 m/s -> freeway mainline + interchange ramps + perpendicular arterial |
| 1 | 21 | 57,851 | +3,033 | +144 | 12 NS + 6 EW -> second freeway + arterial pair |
| 2 | 13 | 61,202 | +1,971 | +152 | **all 13 NS, median 6 lanes, median length 1,273 m, all at 70 mph -> textbook I-15 mainline** |

The 39-link component is the largest single Fair-attributable region
(within the user's 10-300 link target). Its mixed orientation (NS + EW)
and span of speeds (some at 31 mph, most at 70 mph) is the pattern of
an **interchange**: freeway mainline + on/off ramps + perpendicular
arterial. Given that the Utah State Fair was reached via I-15 Exit
309, this region almost certainly is the Fairpark interchange.

## Time-of-day profile of the 39-link region (Fair vs reference)

The Fair curve sits above the reference band throughout the daytime,
with two clean peaks:

- **08:00-09:00**: +97 trips/day vs reference. Consistent with vendors,
  exhibitors, and livestock handlers arriving before the Fair opens.
- **14:00-17:00**: +47 to +62 trips/day. Consistent with public
  attendees on the noon-opening days (Sep 11/12/13 were the only days
  the Fair opened at noon; the rest opened at 10 a.m.).

## Two rankings, two stories

- **`top_lift.csv`** (top 200 by absolute lift) is mostly freeway-class
  links (70 mph, 4-6 lanes) -- the *spillover* of Fair traffic onto
  I-15 and its surroundings.
- **`top_fold.csv`** (top 200 by fold change, requiring ref >= 20) is
  mostly **city streets** (8.9-20.1 m/s = 20-45 mph, 1-3 lanes, short
  segments). Link 47879 went from a median 20 trips on a normal Tue-Thu
  to 53 during the Fair (2.65x). These look like Fair parking access /
  side-street parking.

## Files in this folder

- `METHODS_FAIR.md` -- full methodology (sections 1-11), reproducibility
  command, definitions of lift/fold, why floors were applied, what we
  can and cannot say from the data alone.
- `per_link.csv` -- every one of 60,069 links with fair / refs / lift /
  fold / attributes.
- `top_lift.csv`, `top_fold.csv` -- top 200 of each ranking.
- `region_components.json` -- all 59 connected components in the
  top-200, sorted by size.
- `region_top_links.csv` -- the 39-link largest component with link IDs.
- `tod_profile.png` + `tod_profile.csv` -- the time-of-day plot, plus
  the underlying numbers.
- `hist_lift.png`, `hist_fold.png`, `summary.json` -- global
  distributions of lift and fold over the 16,744 links passing
  `ref_median >= 5`.

## Next-step options not yet executed

- Generate the same TOD profile for components 1 and 2 for direct
  comparison.
- Cross-check the city-street fold winners by looking at the
  orientation/length of each, to confirm they are residential side
  streets near the Fairpark rather than something else.
- Map the link IDs to geographic coordinates (if a coordinate file
  exists for `city_graph_full.json`) and overlay on a basemap to
  visually confirm the I-15 / Exit 309 / Fairpark hypothesis.
