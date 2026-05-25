One follow-up on why I restrict the event-window evaluation to the top-K busiest links near the stadium rather than computing the error over the full network.

The graph has roughly 99,716 directed road links, but the football game can only perturb traffic on a small subset of them — the corridors that feed into Rice-Eccles. If I evaluate the d2m over the entire network, the per-link improvement on those affected corridors gets averaged together with tens of thousands of links that the event does not touch. The signal is diluted by the bulk of the graph, and the network-wide d2m barely moves between the default and the tuned chain; the two-phase vs single-phase difference also disappears at that scale.

Restricting to the top-K cluster inside a 1 km radius of the stadium isolates the region where the intervention is physically present. On that cluster the tuned chain reduces the d2m by 73-80% relative to its own default, and the gap between single-phase and two-phase becomes visible (5-10 percentage points on every cluster, as in the previous message). The same numbers, computed network-wide, are within rounding distance of each other.

In short, the top-K cluster acts as a magnifying glass on the affected region; it is not chosen to make the chain look good, but to compute the event-window error where the event actually happens instead of washing it out across the rest of the city.

Best,
Hamid
