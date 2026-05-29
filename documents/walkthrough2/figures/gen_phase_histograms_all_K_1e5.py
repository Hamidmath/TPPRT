"""Same as gen_phase_histograms_all_K.py but with the x-axis cut where
the geometric reference drops below 10^-5 (instead of 10^-6).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_NPZ = os.path.join(HERE, "phase_split_all_K.npz")


def panel(ax, x, color, label_phase, geom_symbol):
    n = len(x)
    mean = x.mean()
    p = 1.0 / (1.0 + mean)
    PROB_FLOOR = 1e-5
    BIN_W = 1
    xmax_data = int(x.max()) + 1
    bins = np.arange(0, xmax_data + BIN_W, BIN_W)
    xmax_view = int(np.ceil(np.log(PROB_FLOOR / p) / np.log(1.0 - p)))

    ax.hist(
        x, bins=bins, density=True,
        color=color, alpha=1.0,
        edgecolor="none", linewidth=0,
        label=(f"$N={n:,}$\n"
                f"$E[{label_phase}]={mean:.2f}$"),
    )
    bin_left = np.arange(0, xmax_data, BIN_W)
    bin_prob = sum(((1.0 - p) ** (bin_left + offset) * p)
                    for offset in range(BIN_W))
    geom_density = bin_prob / BIN_W
    ax.plot(bin_left + BIN_W / 2, geom_density,
             color="black", linewidth=1.4,
             label=f"Geom(${geom_symbol}={p:.3f}$)")

    ax.set_yscale("log")
    ax.set_xlim(0, xmax_view)
    ax.set_ylim(PROB_FLOOR, 0.4)
    ax.set_xlabel(f"${label_phase}$ (links)")
    ax.set_ylabel("probability")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


def main():
    data = np.load(SPLIT_NPZ)
    Lu = data["L_up"].astype(np.int64)
    Ld = data["L_down"].astype(np.int64)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True)
    panel(axes[0], Lu, "#3a8533", "L_{\\uparrow}",  "\\beta")
    panel(axes[1], Ld, "#553099", "L_{\\downarrow}", "\\rho")

    out_pdf = os.path.join(HERE, "phase_histograms_all_K_1e5.pdf")
    out_png = os.path.join(HERE, "phase_histograms_all_K_1e5.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")
    print(f"N = {len(Lu):,}")
    print(f"L_up: range [0, {Lu.max()}], mean = {Lu.mean():.4f}, "
          f"beta = {1/(1+Lu.mean()):.4f}")
    print(f"L_down: range [0, {Ld.max()}], mean = {Ld.mean():.4f}, "
          f"rho  = {1/(1+Ld.mean()):.4f}")


if __name__ == "__main__":
    main()
