# Sigma calibration (GPS noise scale for map-matching)

Picks the HMM observation-noise scale sigma by self-consistency: the declared
sigma should equal the standard deviation of the matched-fix residuals. This
reproduces the sigma-choice table in the paper (`tab:sigma-choice`).

Chain:

```
build_network.py                build the OSM/MATSim network
match_routes_osm_chunked.py     map-match the corpus at a given obs_noise (sigma), in chunks
aggregate_chunks.py             combine the per-chunk matched JSON
measure_residual_full_generic.py  std of perpendicular fix-to-subedge residuals (drop > 30 m)
                                  -> results/sigma_residual/residual_full_noise{TAG}.json
make_sigma_table.py             read the result JSONs -> the sigma-choice table
```

`compare_obs_noise_long_links.py` is a supporting analysis of how match
coverage on long links changes with obs_noise.

The matching stages need the raw GPS (not shipped). The residual result JSONs
for sigma in {1, 1.75, 3, 3.5, 3.75, 4, 5} are included under
`results/sigma_residual/`, so the table reproduces directly:

```
python analysis/sigma_calibration/make_sigma_table.py
```

sigma_measured is the standard deviation of the residuals after dropping
residuals over 30 m (`stats_after_filter.sigma_std`). The chosen value is the
smallest declared sigma that exceeds the measured residual std (sigma = 3.5).
