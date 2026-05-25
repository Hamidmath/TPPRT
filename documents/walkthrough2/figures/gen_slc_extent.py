"""Two-panel figure: where the raw GPS fixes fall vs the SLC
bounding box used for preprocessing.

(a) wide-area scatter (~west US), shows the spatial spread.
(b) zoom into the SLC bounding box, shows inside vs outside in
    the local neighborhood.

Sampled from data/compressed_routes.parquet, 924M fixes total.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent.parent.parent
PQ = ROOT / "data" / "compressed_routes.parquet"
OUT = Path(__file__).resolve().parent / "slc_extent.pdf"

# Same generous SLC metro bounding box used in the preprocessing
LAT_MIN, LAT_MAX = 40.4, 40.95
LON_MIN, LON_MAX = -112.15, -111.55

# Sampling: draw N random fixes total
N_SAMPLE = 200_000
SEED = 7

print(f"loading {PQ}")
p = pq.ParquetFile(str(PQ))
total = p.metadata.num_rows
print(f"  total fixes: {total:,}")

# Sample uniformly across batches
rng = np.random.default_rng(SEED)
samples_lat, samples_lon = [], []
per_batch_target = N_SAMPLE // p.num_row_groups + 1
for batch in p.iter_batches(batch_size=2_000_000, columns=['lat', 'lon']):
    lat = batch['lat'].to_numpy(); lon = batch['lon'].to_numpy()
    n = len(lat)
    take = min(per_batch_target, n)
    if take < n:
        idx = rng.choice(n, take, replace=False)
        samples_lat.append(lat[idx]); samples_lon.append(lon[idx])
    else:
        samples_lat.append(lat); samples_lon.append(lon)
lat = np.concatenate(samples_lat); lon = np.concatenate(samples_lon)
print(f"  sampled fixes: {len(lat):,}")
inside = (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)
print(f"  inside SLC bbox: {inside.sum():,} ({100*inside.sum()/len(lat):.1f}%)")

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

# Plot frame
lat_lo, lat_hi = 40.25, 41.25
lon_lo, lon_hi = -112.5, -111.0
mask_zoom = (lat >= lat_lo) & (lat <= lat_hi) & (lon >= lon_lo) & (lon <= lon_hi)
lat_z = lat[mask_zoom]; lon_z = lon[mask_zoom]
in_z = (lat_z >= LAT_MIN) & (lat_z <= LAT_MAX) & (lon_z >= LON_MIN) & (lon_z <= LON_MAX)
ax.scatter(lon_z[~in_z], lat_z[~in_z], s=0.6, alpha=0.30,
            color="#888", label="outside box", rasterized=True)
ax.scatter(lon_z[in_z], lat_z[in_z], s=0.6, alpha=0.60,
            color="#c0392b", label="inside box", rasterized=True)
rect = mpatches.Rectangle((LON_MIN, LAT_MIN), LON_MAX - LON_MIN,
                             LAT_MAX - LAT_MIN, fill=False, lw=1.5,
                             edgecolor="#000", linestyle="--")
ax.add_patch(rect)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
ax.set_xlim(lon_lo, lon_hi); ax.set_ylim(lat_lo, lat_hi)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", markerscale=4, framealpha=0.85)

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight", dpi=200)
print(f"saved {OUT}")
