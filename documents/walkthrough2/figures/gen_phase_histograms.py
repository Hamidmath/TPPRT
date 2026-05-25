"""Generate the empirical L_up / L_down histogram for the walkthrough.

Reads the per-trip phase split saved by the route-distribution study and
produces a two-panel histogram with the geometric reference overlay and
the resulting (beta, rho) annotations.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_NPZ = os.path.normpath(
    os.path.join(
        HERE,
        os.pardir,
        os.pardir,
        os.pardir,
        "results",
        "route_distribution_study",
        "empirical_phase_split.npz",
    )
)


def panel(ax, x, color, label, label_phase, beta_or_rho_symbol):
    mean = x.mean()
    p = 1.0 / (1.0 + mean)
    xmax = int(np.percentile(x, 99.5)) + 1
    bins = np.arange(0, xmax + 1)
    ax.hist(
        x,
        bins=bins,
        density=True,
        color=color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.3,
        label=f"empirical ({label})",
    )
    k = np.arange(0, xmax + 1)
    geom = (1.0 - p) ** k * p
    ax.plot(k + 0.5, geom, color="black", linewidth=1.4,
            label=f"Geom(${beta_or_rho_symbol}={p:.3f}$)")
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_ylim(1e-4, 0.4)
    ax.set_xlabel(f"${label_phase}$ (links)")
    ax.set_ylabel("probability")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize=9)


def main():
    data = np.load(SPLIT_NPZ)
    Lu = data["L_up"].astype(np.int64)
    Ld = data["L_down"].astype(np.int64)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    panel(axes[0], Lu, "#4c8c4a", "up phase",
          "L_{\\uparrow}", "\\beta")
    panel(axes[1], Ld, "#7e57c2", "down phase",
          "L_{\\downarrow}", "\\rho")

    out_pdf = os.path.join(HERE, "phase_histograms.pdf")
    out_png = os.path.join(HERE, "phase_histograms.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print(f"N trips with >=3 links = {len(Lu)}")
    print(f"E[L_up]   = {Lu.mean():.4f}  -> beta = {1/(1+Lu.mean()):.4f}")
    print(f"E[L_down] = {Ld.mean():.4f}  -> rho  = {1/(1+Ld.mean()):.4f}")


if __name__ == "__main__":
    main()
