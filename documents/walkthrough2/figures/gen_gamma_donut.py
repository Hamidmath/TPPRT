"""Donut chart showing how active cells distribute across
displacement bands at gamma=0.20 under the GLOBAL criterion.

Bands (D / tau_global):
  [0, 0.5]       91.05%
  (0.5, 0.8]     4.55%
  (0.8, 1.0]    1.40%
  (1.0, 1.5]    1.70%
  (1.5, 2.0]    0.70%
  (2.0, inf)    0.60%

Numbers come from the violator-tail mini-table (paper); the
complement sums to 100%.

Output: figures/gamma_donut_violators.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "gamma_donut_violators.pdf"

violator_total = 4.55 + 1.40 + 1.70 + 0.70 + 0.60   # ~8.95
pass_size = 100.0 - violator_total

fig, ax = plt.subplots(figsize=(6.5, 5.0))

sizes = [pass_size, violator_total]
labels = [
    rf"$\leq 0.5\,\tau$" + f"\n(pass)\n{pass_size:.2f}\\%",
    rf"$>0.5\tau$  {violator_total:.2f}\%",
]
colors = ["#5b8fd6", "#c0392b"]

wedges, _ = ax.pie(
    sizes, colors=colors,
    startangle=90, counterclock=False,
    wedgeprops=dict(edgecolor="white", linewidth=1.2),
)
# Pass label inside the large wedge
ax.text(0.2, 0.0, labels[0], ha="center", va="center",
         fontsize=12, weight="bold", color="white")
# Violator label INSIDE the small wedge.
# The wedge is ~32 degrees wide centered around 74 deg from +x; the text is
# laid out radially (rotated to point along the spoke) so it fits inside.
violator_center_angle = 90 - (violator_total / 2.0) * 3.6  # degrees
rad = np.deg2rad(violator_center_angle)
rx = 0.62 * np.cos(rad); ry = 0.62 * np.sin(rad)
ax.text(rx, ry, labels[1], ha="center", va="center",
         fontsize=8.5, weight="bold", color="white",
         rotation=violator_center_angle - 90)
ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.2, 1.3)
ax.set_aspect("equal"); ax.axis("off")
ax.text(0, -1.35,
        r"$\gamma = 0.20$, global criterion, "
        r"$n_{\mathrm{active}} = 5.95\times 10^{6}$",
        ha="center", va="center", fontsize=10, color="#444")
plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"saved {OUT}")
