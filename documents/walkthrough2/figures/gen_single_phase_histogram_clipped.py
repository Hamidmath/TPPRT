"""Same plot range as the cut version (K = 0..82), but every trip with
K > 82 is *clipped* to K = 82 instead of being dropped. The last bar
then represents the full mass of "K >= 82" (a "tail spike").
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
    K_raw = z["K"].astype(np.int64)
    n = len(K_raw)
    mean = K_raw.mean()                    # statistics on the FULL data
    p = 1.0 / (1.0 + mean)

    xmax = int(np.percentile(K_raw, 99.5)) + 1   # = 82
    # Keep K <= 83 only; clip K = 83 down to 82. Trips with K >= 84
    # are dropped. So the last bin lumps K = 82 and K = 83 together.
    K = K_raw[K_raw <= 83]
    K = np.minimum(K, xmax)
    n_clipped = int((K_raw == 83).sum())
    n_dropped = int((K_raw >= 84).sum())

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
        f"(K=83 clipped to {xmax}; K>=84 dropped;  "
        f"$E[K]={mean:.2f}$, $p={p:.4f}$)",
        fontsize=10,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    # Annotate the lumped tail
    ax.annotate(f"K=83 ({n_clipped:,} trips) lumped onto {xmax}",
                 xy=(xmax - 0.4, 0.003), xytext=(xmax - 28, 0.025),
                 fontsize=8, color="black",
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))

    out_pdf = os.path.join(HERE, "single_phase_histogram_clipped.pdf")
    out_png = os.path.join(HERE, "single_phase_histogram_clipped.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print(f"N = {n:,}")
    print(f"E[K] = {mean:.4f}  (on full data, not on clipped)")
    print(f"K = 83 clipped to {xmax}: {n_clipped:,} trips "
          f"({100*n_clipped/n:.3f}%)")
    print(f"K >= 84 dropped: {n_dropped:,} trips "
          f"({100*n_dropped/n:.3f}%)")


if __name__ == "__main__":
    main()
