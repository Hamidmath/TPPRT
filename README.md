# Two-Phase PageRank

Link-level traffic prediction on the Salt Lake City road network from INRIX probe trajectories. Raw GPS is map-matched to the road graph and binned into per-link popularity, then a two-phase PageRank chain over a graph-diffused prior forecasts heavy-traffic links.

```
            raw GPS CSVs
                 │
                 ▼   compress_routes.py      clean + segment into trips
         compressed_routes.parquet
                 │
                 ▼   match_routes.py         HMM map-matching (σ = 3.5 m)
            matched_routes.json
                 │
                 ▼   analyze_popularity.py   time-binned per-link counts
           popularity_results.npz
                 │
                 ▼   generate_smoothed.py    graph diffusion (γ = 0.20)
          diffused popularity prior
                 │
                 ▼   core/pagerank.py        two-phase PageRank chain
           per-link traffic forecast
```
