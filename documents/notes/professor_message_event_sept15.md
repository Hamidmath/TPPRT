Hi Prof,

I finished the analysis on the Utah vs Washington event (Sept 15, 2018, kickoff at 8:00 PM at Rice-Eccles Stadium).

**Cluster definition.** I identified the top-10, top-20, and top-30 busiest links within a 1 km radius of the stadium. The ranking is built from the empirical popularity averaged over the three clean baseline Saturdays in the corpus (Sept 1, 8, and 22), restricted to the 19:00-19:55 window, i.e. the hour leading up to kickoff on a normal Saturday.

**Setup.** I ran both the single-phase and the two-phase PageRank chains on the Sept 8 to Sept 15 event window (12 five-minute bins, 19:00-19:55). For each chain I fit its free parameters by L-BFGS-B gradient descent against the per-cluster Top-K d2m, with the teleport/commit parameters constrained to [0.01, 0.15] (so at each step the chain restarts to the ground-truth prior with probability at most 15% and moves along the graph transition with probability at least 85%). The speed and lane weighting exponents alpha_s, alpha_l were left unbounded.

**Result.** Tuning reduced the d2m error by roughly 73-76% for the single-phase chain and 76-79% for the two-phase chain, each relative to its own default parameters. In absolute terms the tuned two-phase d2m is 0.728 / 0.731 / 0.802 on top-10 / top-20 / top-30, versus the tuned single-phase d2m of 0.806 / 0.811 / 0.849. The two-phase chain is therefore 5-10 percentage points below the single-phase chain on every cluster, even after both chains are given full freedom to fit the event window.

I have the per-cluster optimal parameter values and the head-to-head comparison table written up in the appendix of the draft. Happy to share if helpful.

Best,
Hamid
