"""Plot per-bin Stadium MRE time series for Sept 1->8 (control) and
Sept 8->15 (event). Reveals whether the event has a time-specific
signature that the 84-bin average washed out.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project/results/event_sept15/results.json")
OUT = Path("/home/hamid/Downloads/new/TwoPhase_PageRank_Project/results/event_sept15")

with open(RES) as f:
    data = json.load(f)
exp = data["experiment_1_default_params"]

ctrl_top   = np.array(exp["control_top100_per_bin"])
ctrl_ovr   = np.array(exp["control_overall_per_bin"])
ctrl_cl    = np.array(exp["control_cluster_per_bin"])
evt_top    = np.array(exp["event_top100_per_bin"])
evt_ovr    = np.array(exp["event_overall_per_bin"])
evt_cl     = np.array(exp["event_cluster_per_bin"])
bin_labels = exp["event_bins"]
times = [b[11:16] for b in bin_labels]
n = len(times)
x = np.arange(n)

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, (lbl, ctrl, evt) in zip(axes, [
    ("Top-100 MRE", ctrl_top, evt_top),
    ("Overall MRE", ctrl_ovr, evt_ovr),
    (f"Stadium MRE (1,913 links within 2km)", ctrl_cl, evt_cl),
]):
    ax.plot(x, ctrl, "o-", color="#1f6fb4", lw=1.4, ms=4,
            label="control (Sept 1 -> Sept 8)")
    ax.plot(x, evt, "s-", color="#c0392b", lw=1.4, ms=4,
            label="event (Sept 8 -> Sept 15)")
    ax.axvline(np.searchsorted(times, "20:00"), color="black", ls="--", lw=0.8,
               label="kickoff 20:00")
    ax.axvline(np.searchsorted(times, "23:00"), color="black", ls=":", lw=0.8,
               label="post-game ~23:00")
    ax.set_ylabel(lbl)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xticks(x[::6])
axes[-1].set_xticklabels(times[::6], rotation=45)
axes[-1].set_xlabel("bin (Sept 15 evening, MDT)")
fig.suptitle("Per-bin forecast MRE: control Saturday vs event Saturday "
             "(default beta=0.102, rho=0.101)")
fig.tight_layout()
fig.savefig(OUT / "fig_per_bin_mre.png", dpi=130)
plt.close(fig)
print(f"saved {OUT/'fig_per_bin_mre.png'}")

# Also print: did event-cluster MRE peak around kickoff or post-game?
i_kick = np.searchsorted(times, "20:00")
i_post = np.searchsorted(times, "23:00")
print()
print(f"kickoff bin index   = {i_kick} (time={times[i_kick]})")
print(f"post-game bin index = {i_post} (time={times[i_post]})")
print()
print("Stadium MRE control vs event at key bins:")
for i in [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 83]:
    if i < n:
        print(f"  bin {i:3d} ({times[i]})  control={ctrl_cl[i]:.3f}  event={evt_cl[i]:.3f}  ratio={evt_cl[i]/max(ctrl_cl[i],1e-6):.2f}x")
