"""Same as gen_single_phase_histogram.py but with the x-axis cut where
the geometric reference drops below 10^-5 (instead of 10^-6).
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

    PROB_FLOOR = 1e-5
    BIN_W = 1
    xmax_data = int(K.max()) + 1
    bins = np.arange(0, xmax_data + BIN_W, BIN_W)
    xmax_view = int(np.ceil(np.log(PROB_FLOOR / p) / np.log(1.0 - p)))

    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    ax.hist(
        K, bins=bins, density=True,
        color="#777777", alpha=1.0,
        edgecolor="none", linewidth=0,
        label=(f"$N={n:,}$\n"
                f"$E[K]={mean:.2f}$"),
    )
    bin_left = np.arange(0, xmax_data, BIN_W)
    bin_prob = sum(((1.0 - p) ** (bin_left + offset) * p)
                    for offset in range(BIN_W))
    geom_density = bin_prob / BIN_W
    ax.plot(
        bin_left + BIN_W / 2, geom_density, color="black", linewidth=1.4,
        label=f"Geom($p = {p:.4f}$)",
    )
    ax.set_yscale("log")
    ax.set_xlim(0, xmax_view)
    ax.set_ylim(PROB_FLOOR, 0.2)
    ax.set_xlabel(r"$K_{\mathrm{total}}$ (total links per route)")
    ax.set_ylabel("probability")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax.set_title("Single-phase route-length distribution",
                  fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    out_pdf = os.path.join(HERE, "single_phase_histogram_1e5.pdf")
    out_png = os.path.join(HERE, "single_phase_histogram_1e5.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print(f"N = {n:,}")
    print(f"K range = [{K.min()}, {K.max()}]")
    print(f"E[K_total] = {mean:.4f}")
    print(f"p          = 1/(1+E[K]) = {p:.6f}")
    print(f"xmax_view (geom < {PROB_FLOOR:g}) = {xmax_view}")


if __name__ == "__main__":
    main()
