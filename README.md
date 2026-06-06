# Two-Phase PageRank

Link-level traffic prediction on the Salt Lake City road network from INRIX probe trajectories. Raw GPS is map-matched to the road graph and binned into per-link popularity, then a two-phase PageRank chain over a graph-diffused prior forecasts heavy-traffic links, with event case studies (a crash, a stadium game, and a street festival). The SIGSPATIAL paper lives in `documents/walkthrough2/main.tex`.
