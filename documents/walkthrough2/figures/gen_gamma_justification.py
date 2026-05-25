"""Two figures that visualize Table 3 (γ noise-floor justification).

The data are taken from the published table; this script just
plots them in two ways:

  Panel A — quantile curves vs γ (median and 95th-percentile),
            with the 0.5 criterion line marked. One curve per
            criterion (per-cell vs global).
  Panel B — fraction of cells exceeding 0.5σ^NF vs γ, for both
            criteria.

Both panels mark γ=0.20 with a vertical line.

Outputs:
  figures/gamma_just_quantiles.pdf
  figures/gamma_just_violators.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

# Table 3 numbers (paper)
gamma = np.array([0.00, 0.05, 0.10, 0.12, 0.13, 0.14, 0.15,
                  0.20, 0.25, 0.30, 0.40, 0.50, 0.70])
# per-cell
pc_med = np.array([0.000, 0.038, 0.069, 0.081, 0.087, 0.093, 0.099,
                   0.130, 0.160, 0.190, 0.248, 0.306, 0.418])
pc_p95 = np.array([0.000, 0.100, 0.186, 0.219, 0.236, 0.253, 0.270,
                   0.354, 0.436, 0.518, 0.680, 0.838, 1.145])
pc_frac = np.array([0.00, 0.42, 1.00, 1.27, 1.41, 1.56, 1.73,
                    2.62, 3.63, 5.39, 9.42, 17.29, 32.10])  # percent
# global
gl_med = np.array([0.000, 0.034, 0.062, 0.073, 0.079, 0.084, 0.090,
                   0.117, 0.144, 0.171, 0.224, 0.276, 0.376])
gl_p95 = np.array([0.000, 0.215, 0.393, 0.463, 0.498, 0.533, 0.568,
                   0.741, 0.912, 1.081, 1.413, 1.737, 2.364])
gl_frac = np.array([0.00, 0.90, 3.31, 4.40, 4.96, 5.53, 6.10,
                    8.95, 11.87, 14.81, 20.64, 26.59, 37.83])  # percent

GAMMA_USED = 0.20
THR = 0.5

# Figure A — 95th-percentile curves only (medians dropped at user request)
fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
ax.plot(gamma, pc_p95, "o-", color="#1f6fb4", lw=1.8,
        label="per-cell, 95th pct")
ax.plot(gamma, gl_p95, "s-", color="#c0392b", lw=1.8,
        label="global, 95th pct")
ax.axhline(THR, color="black", ls=":", lw=1.0,
           label=f"criterion: ratio $\\leq$ {THR}")
ax.axvline(GAMMA_USED, color="gray", ls="-.", lw=1.0,
           label=f"$\\gamma$ used = {GAMMA_USED}")
ax.set_xlabel(r"diffusion strength $\gamma$")
ax.set_ylabel(r"$95$th percentile of $D_{b,i}/\tau$")
ax.set_title("95th-percentile of per-cell and global ratio vs $\\gamma$")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig(OUT_DIR / "gamma_just_quantiles.pdf", bbox_inches="tight",
            dpi=150)
plt.close(fig)
print("saved gamma_just_quantiles.pdf")

# Figure B — fraction > 0.5
fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
ax.plot(gamma, pc_frac, "o-", color="#1f6fb4", lw=1.8,
        label="per-cell criterion")
ax.plot(gamma, gl_frac, "s-", color="#c0392b", lw=1.8,
        label="global criterion")
ax.axvline(GAMMA_USED, color="gray", ls="-.", lw=1.0,
           label=f"$\\gamma$ used = {GAMMA_USED}")
ax.set_xlabel(r"diffusion strength $\gamma$")
ax.set_ylabel(r"fraction of cells with $D_{b,i}/\tau > 0.5$ (%)")
ax.set_title("Cells exceeding the half-noise tolerance vs $\\gamma$")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig(OUT_DIR / "gamma_just_violators.pdf", bbox_inches="tight",
            dpi=150)
plt.close(fig)
print("saved gamma_just_violators.pdf")
