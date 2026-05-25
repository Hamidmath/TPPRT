"""Plot the up-phase / down-phase length histograms at OSM sub-edge
granularity, comparing them against the MATSim-link-aggregated version.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def panel(ax, x, color, label, label_phase):
    mean = float(x.mean())
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
            label=f"Geom($p={p:.3f}$)")
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_ylim(1e-4, 0.4)
    ax.set_xlabel(f"${label_phase}$ (sub-edges)")
    ax.set_ylabel("probability")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize=9)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    npz = np.load(
        os.path.join(
            root, "results/route_distribution_study/"
            "empirical_phase_split_osm_subedge.npz"
        )
    )
    Lu = npz["L_up"].astype(np.int64)
    Ld = npz["L_down"].astype(np.int64)
    print(f"N (K>=3 matsim) trips = {len(Lu)}")
    print(f"E[L_up]   = {Lu.mean():.4f}  -> beta = {1/(1+Lu.mean()):.4f}")
    print(f"E[L_down] = {Ld.mean():.4f}  -> rho  = {1/(1+Ld.mean()):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    panel(axes[0], Lu, "#1f77b4", "up", "L_\\mathrm{up}")
    panel(axes[1], Ld, "#d62728", "down", "L_\\mathrm{down}")
    out_pdf = os.path.join(
        root, "documents/walkthrough2/figures/phase_histograms_osm_subedge.pdf"
    )
    out_png = out_pdf.replace(".pdf", ".png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
