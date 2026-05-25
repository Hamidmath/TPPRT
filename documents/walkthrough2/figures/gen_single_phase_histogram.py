"""Generate the single-phase histogram: empirical distribution of the
per-trip total link count K_total = L_up + 1 + L_down, with the
matched geometric overlay used by the single-phase PageRank baseline.

Mean and p match the KS analysis ("single (K_total)" row):
    E[K_total] = mean,
    p          = 1 / (1 + E[K_total]).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_NPZ = os.path.normpath(
    os.path.join(
        HERE, os.pardir, os.pardir, os.pardir,
        "results", "route_distribution_study", "empirical_phase_split.npz",
    )
)


def main():
    z = np.load(SPLIT_NPZ)
    K = z["K"].astype(np.int64)
    n = len(K)
    mean = K.mean()
    p = 1.0 / (1.0 + mean)

    xmax = int(np.percentile(K, 99.5)) + 1
    bins = np.arange(0, xmax + 1)

    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    ax.hist(
        K, bins=bins, density=True,
        color="#1f6fb4", alpha=0.85,
        edgecolor="white", linewidth=0.3,
        label=f"empirical $K_{{\\mathrm{{total}}}}$ ($N={n:,}$ trips)",
    )
    k = np.arange(0, xmax + 1)
    geom = (1.0 - p) ** k * p
    ax.plot(
        k + 0.5, geom, color="black", linewidth=1.4,
        label=f"Geom($p = {p:.4f}$), $1/(1+E[K])$",
    )
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_ylim(1e-5, 0.2)
    ax.set_xlabel(r"$K_{\mathrm{total}}$ (total links per route)")
    ax.set_ylabel("probability")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax.set_title(
        f"Single-phase route-length distribution  "
        f"(all-K, $E[K]={mean:.2f}$, $p={p:.4f}$)",
        fontsize=11,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    out_pdf = os.path.join(HERE, "single_phase_histogram.pdf")
    out_png = os.path.join(HERE, "single_phase_histogram.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print(f"N = {n:,}")
    print(f"E[K_total] = {mean:.4f}")
    print(f"p          = 1/(1+E[K]) = {p:.6f}")
    print(f"damping d  = 1 - p      = {1 - p:.6f}")


if __name__ == "__main__":
    main()
