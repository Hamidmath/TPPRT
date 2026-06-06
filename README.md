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

## Event case studies

We stress-test the model on three Salt Lake City events:

- **I-15 crash** — a lane closure. We cut the closed link from the network, re-solve the chain, and compare the eval-box error with and without this surgery.
- **Stadium game** — a pure demand surge, no closures. We learn a prior mass-transfer ψ that shifts demand onto the stadium box, fit on a held-out split of the box links.
- **9th & 9th street festival** — a road closure plus a demand shift. We combine surgery (remove the closed streets) with the learned ψ, chosen by 5-fold cross-validation.
