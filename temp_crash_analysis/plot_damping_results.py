"""
Generate a publication-quality 3-panel figure showing damping optimization results
for the Two-Phase PageRank model.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Data  –  Panel A
# ---------------------------------------------------------------------------
damping_A = np.array([0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96])

t100_mean = np.array([0.0312, 0.0356, 0.0409, 0.0474, 0.0558, 0.0668, 0.0820, 0.1050, 0.1443])
t100_err  = np.array([0.0122, 0.0137, 0.0155, 0.0176, 0.0201, 0.0232, 0.0271, 0.0321, 0.0389])

ov_mean   = np.array([0.0464, 0.0528, 0.0605, 0.0699, 0.0817, 0.0970, 0.1177, 0.1475, 0.1947])
ov_err    = np.array([0.0056, 0.0064, 0.0074, 0.0086, 0.0101, 0.0122, 0.0150, 0.0193, 0.0264])

# ---------------------------------------------------------------------------
# Data  –  Panel B
# ---------------------------------------------------------------------------
gammas = np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])

damping_labels_B = ["d=0.91", "d=0.92", "d=0.93", "d=0.94", "d=0.96"]
mre_B = {
    "d=0.91": [0.1066, 0.0714, 0.0458, 0.0284, 0.0171, 0.0100, 0.0058],
    "d=0.92": [0.1177, 0.0796, 0.0515, 0.0321, 0.0194, 0.0115, 0.0067],
    "d=0.93": [0.1311, 0.0896, 0.0585, 0.0367, 0.0223, 0.0133, 0.0077],
    "d=0.94": [0.1475, 0.1020, 0.0673, 0.0426, 0.0261, 0.0156, 0.0091],
    "d=0.96": [0.1947, 0.1389, 0.0944, 0.0613, 0.0383, 0.0232, 0.0138],
}
cmap_B = plt.cm.viridis(np.linspace(0.15, 0.90, len(damping_labels_B)))

# ---------------------------------------------------------------------------
# Data  –  Panel C
# ---------------------------------------------------------------------------
config_labels = [
    "Original\nd=0.80, \u03b3=0",
    "Baseline\nd=0.90, \u03b3=0",
    "Optimized\nd=0.94, \u03b3=0.15",
    "Optimized\nd=0.96, \u03b3=0.20",
]
ov_C  = [0.0464, 0.0970, 0.0426, 0.0383]
t100_C = [0.0312, 0.0668, 0.0315, 0.0296]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ========================  PANEL A  ========================================
ax = axes[0]
ax.errorbar(damping_A, t100_mean, yerr=t100_err, fmt="o-", color="tab:red",
            capsize=4, linewidth=1.8, markersize=6, label="Top-100 MRE", zorder=3)
ax.errorbar(damping_A, ov_mean, yerr=ov_err, fmt="s-", color="tab:blue",
            capsize=4, linewidth=1.8, markersize=6, label="Overall MRE", zorder=3)

ax.axhline(0.05, color="red", linestyle="--", linewidth=1.0, alpha=0.7, label="Target")

# Mark best point (d=0.80)
ax.annotate("best (d=0.80)",
            xy=(0.80, t100_mean[0]), xytext=(0.83, t100_mean[0] - 0.015),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=10, ha="left", va="top")

ax.set_xlabel("Damping factor (d)")
ax.set_ylabel("MRE")
ax.set_title("A.  Baseline: Damping vs MRE\n(no demand weighting)", fontweight="bold")
ax.set_xticks(damping_A)
ax.set_xticklabels([f"{d:.2f}" for d in damping_A], rotation=45, ha="right")
ax.legend(loc="upper left", frameon=True, edgecolor="0.8")
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)

# ========================  PANEL B  ========================================
ax = axes[1]

# Green shaded target zone
ax.axhspan(0, 0.05, color="green", alpha=0.08, zorder=0)
ax.text(0.28, 0.025, "Target zone", color="green", fontsize=10,
        ha="right", va="center", fontstyle="italic", alpha=0.8)

for idx, label in enumerate(damping_labels_B):
    ax.plot(gammas, mre_B[label], "o-", color=cmap_B[idx],
            linewidth=1.8, markersize=6, label=label, zorder=3)

ax.axhline(0.05, color="red", linestyle="--", linewidth=1.0, alpha=0.7)

ax.set_xlabel("Demand-weight exponent (\u03b3)")
ax.set_ylabel("Overall MRE")
ax.set_title("B.  Demand-Weighted: \u03b3 vs Overall MRE\n(49 timeframes)", fontweight="bold")
ax.set_xticks(gammas)
ax.legend(loc="upper right", frameon=True, edgecolor="0.8", ncol=1)
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.3)

# ========================  PANEL C  ========================================
ax = axes[2]

x_pos = np.arange(len(config_labels))
bar_w = 0.35

bars_ov  = ax.bar(x_pos - bar_w / 2, ov_C,  bar_w, color="tab:blue",
                  edgecolor="white", linewidth=0.5, label="Overall MRE", zorder=3)
bars_t100 = ax.bar(x_pos + bar_w / 2, t100_C, bar_w, color="tab:red",
                   edgecolor="white", linewidth=0.5, label="Top-100 MRE", zorder=3)

ax.axhline(0.05, color="red", linestyle="--", linewidth=1.0, alpha=0.7, label="Target")

# Annotate bar values
for bar_group in (bars_ov, bars_t100):
    for bar in bar_group:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.4f}", ha="center", va="bottom", fontsize=8.5, rotation=0)

ax.set_xticks(x_pos)
ax.set_xticklabels(config_labels, fontsize=10)
ax.set_ylabel("MRE")
ax.set_title("C.  Comparison: Original vs Optimized", fontweight="bold")
ax.legend(loc="upper right", frameon=True, edgecolor="0.8")
ax.set_ylim(0, max(ov_C) * 1.25)
ax.grid(axis="y", alpha=0.3)

# ---------------------------------------------------------------------------
# Suptitle & save
# ---------------------------------------------------------------------------
fig.suptitle("Damping Factor Optimization with Demand-Weighted Transitions",
             fontsize=16, fontweight="bold", y=1.02)

fig.tight_layout(rect=[0, 0, 1, 0.97])

out_path = "/home/hamid/Downloads/new/TwoPhase_PageRank_Project/temp_crash_analysis/fig_damping_final.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Saved figure to {out_path}")
