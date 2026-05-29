# Hour-restricted comparison: Sep 11 16:00-17:00 vs Sep 18 16:00-17:00

A focused check of the Fair anomaly: on the 39-link region (largest
connected component of the top-200 lift links), restrict to one hour
(12 consecutive 5-min slots) on Tuesday Sep 11, 2018, and compare to
the same hour on Tuesday Sep 18, 2018.

## Setup

- Region: 39 link IDs from `region_components.json` component 0
  (the Fairpark-interchange candidate; freeway mainline + on/off
  ramps + perpendicular arterial; speeds 13.9-31.3 m/s, lanes 2-6,
  mixed NS+EW orientations).
- Slots: 16:00, 16:05, 16:10, ..., 16:55 -- 12 consecutive 5-min bins.
- Source: raw map-matched counts (`data/popularity_results_osm.npz`).

## Numbers

| metric | Sep 11 16:00-17:00 (Fair day) | Sep 18 16:00-17:00 (no Fair) | delta |
|---|---|---|---|
| region total trips | **1,735** | **398** | **+1,337 (+336%)** |
| links with positive delta | 39 / 39 | -- | -- |
| links with negative delta | 0 / 39 | -- | -- |

Every link in the region is elevated during the Fair hour. The top 12
contributing links each add +41 to +52 trips of difference. The signal
is not a one-link spike; it is the whole interchange working harder.

## Top 12 contributing links

```
link_id   Sep11   Sep18   delta
59038       60       8    +52
59236       61       9    +52
59022       56       4    +52
59310       78      27    +51
58911       71      22    +49
58824       76      28    +48
59016       54       7    +47
59312       70      25    +45
64541       70      26    +44
59147       49       6    +43
58931       46       4    +42
59309       68      27    +41
```

These 12 alone account for ~42% of the positive delta. The remaining
27 links carry the other 58%.

## Reproducibility

The numbers above were produced by a one-off block of Python in the
chat session (not yet committed as a script). To regenerate:

```python
import sys, json, numpy as np
sys.path.insert(0, '.')
from core.io import load_popularity_npz

b = load_popularity_npz('data/popularity_results_osm.npz')
M = b['matrix']
times = [str(t) for t in b['times']]
link_ids = [str(x) for x in b['link_ids']]
lid_to_idx = {lid: i for i, lid in enumerate(link_ids)}
t2idx = {t: i for i, t in enumerate(times)}

comps = json.load(open('documents/walkthrough2/results/anomalies/fair/region_components.json'))
region_idxs = [lid_to_idx[l] for l in comps[0]['link_ids'] if l in lid_to_idx]

def slot_range(date, h0, h1):
    return [t2idx[f"{date} {h:02d}:{m:02d}:00"]
            for h in range(h0, h1) for m in range(0, 60, 5)
            if f"{date} {h:02d}:{m:02d}:00" in t2idx]

idx_w2 = slot_range('2018-09-11', 16, 17)
idx_w3 = slot_range('2018-09-18', 16, 17)
c11 = M[idx_w2][:, region_idxs].toarray().sum(axis=0)
c18 = M[idx_w3][:, region_idxs].toarray().sum(axis=0)
print(c11.sum(), c18.sum(), c11.sum() - c18.sum())
```

## Why this matters

The earlier per-link analysis aggregated over the full 3-day Fair
window. This restriction to a single hour confirms that the Fair signal
is also visible at the **intra-day** resolution: the surge in that
specific hour, on that specific region, is 4.4x normal. That gives us
exactly the kind of localized event the chain models should be able to
predict, and is what motivates the targeted optimization step (see
upcoming run).
