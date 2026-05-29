"""Build two single-column case-study figures:

  detect_hist.pdf    histogram of per-slot MRE for three consecutive
                     week-pair comparisons.
  region_bars.pdf    per-5-min trip counts on the 20-link region in
                     the event hour: Sep 18 vs the average of
                     Sep 4, 11, 25.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from core.io import load_popularity_npz

RAW_NPZ = str(ROOT / "data/popularity_results_osm.npz")
OUT_DETECT = Path(__file__).parent / "detect_hist.pdf"
OUT_REGION = Path(__file__).parent / "region_bars.pdf"

REGION = ["58905","58889","58864","58870","58898","58888","58869","58862",
          "59159","59261","59071","59157","58874","58923","59260"]

WEEKS = {
    "W1": ["2018-09-01","2018-09-02","2018-09-03","2018-09-04",
            "2018-09-05","2018-09-06","2018-09-07"],
    "W2": ["2018-09-08","2018-09-09","2018-09-10","2018-09-11",
            "2018-09-12","2018-09-13","2018-09-14"],
    "W3": ["2018-09-15","2018-09-16","2018-09-17","2018-09-18",
            "2018-09-19","2018-09-20","2018-09-21"],
    "W4": ["2018-09-22","2018-09-23","2018-09-24","2018-09-25",
            "2018-09-26","2018-09-27","2018-09-28"],
}
FAIR_DAYS = ["2018-09-11","2018-09-12","2018-09-13"]
REF_DAYS = [["2018-09-04","2018-09-05","2018-09-06"],
            ["2018-09-18","2018-09-19","2018-09-20"],
            ["2018-09-25","2018-09-26","2018-09-27"]]
FLOOR = 5
TOP_K = 24

# Column-width target for acmart sigconf, ~3.3 inches.
COL_W = 3.35


def week_pair_mres(M, t2idx, week_a, week_b):
    mres, n_links = [], []
    for da, db in zip(week_a, week_b):
        for h in range(24):
            for mm in range(0, 60, 5):
                ta = f"{da} {h:02d}:{mm:02d}:00"
                tb = f"{db} {h:02d}:{mm:02d}:00"
                if ta not in t2idx or tb not in t2idx:
                    continue
                base = M.getrow(t2idx[ta]).toarray().ravel().astype(np.float64)
                tgt  = M.getrow(t2idx[tb]).toarray().ravel().astype(np.float64)
                mask = base >= FLOOR
                n = int(mask.sum())
                if n == 0:
                    continue
                mres.append(float((np.abs(tgt[mask] - base[mask]) / base[mask]).mean()))
                n_links.append(n)
    return np.array(mres), np.array(n_links)


def main():
    b = load_popularity_npz(RAW_NPZ)
    M = b["matrix"]
    times = [str(t) for t in b["times"]]
    t2idx = {t: i for i, t in enumerate(times)}
    link_ids = [str(x) for x in b["link_ids"]]
    lid2col = {lid: i for i, lid in enumerate(link_ids)}

    # ---- DETECTION FIGURE: single-panel histogram ----
    print("computing W1->W2 ...")
    mre_12, n_12 = week_pair_mres(M, t2idx, WEEKS["W1"], WEEKS["W2"])
    print("computing W2->W3 ...")
    mre_23, n_23 = week_pair_mres(M, t2idx, WEEKS["W2"], WEEKS["W3"])
    print("computing W3->W4 ...")
    mre_34, n_34 = week_pair_mres(M, t2idx, WEEKS["W3"], WEEKS["W4"])

    eligible_23 = n_23 >= 50
    elig_idx = np.where(eligible_23)[0]
    top_idx = elig_idx[np.argsort(mre_23[elig_idx])[::-1][:TOP_K]]
    cutoff_23 = mre_23[top_idx].min()

    cutoff_12 = np.sort(mre_12[n_12 >= 50])[::-1][:TOP_K].min()
    cutoff_34 = np.sort(mre_34[n_34 >= 50])[::-1][:TOP_K].min()

    bins = np.linspace(0, 1.2, 36)
    samples = [
        (mre_12[n_12 >= 50], "steelblue", r"W1$\to$W2", cutoff_12),
        (mre_23[n_23 >= 50], "crimson",   r"W2$\to$W3", cutoff_23),
        (mre_34[n_34 >= 50], "seagreen",  r"W3$\to$W4", cutoff_34),
    ]
    ymax = max(np.histogram(s, bins=bins)[0].max() for s, _, _, _ in samples)

    fig, axs = plt.subplots(1, 3, figsize=(COL_W, 1.95), sharey=True)
    for ax, (s, color, label, cut) in zip(axs, samples):
        ax.hist(s, bins=bins, color=color, alpha=0.75,
                 histtype="stepfilled", edgecolor=color)
        ax.axvline(cut, color="black", lw=0.7, ls="--")
        # Print the cutoff value high on the dashed line.
        ax.text(cut, ymax * 0.85, f"{cut:.2f}", fontsize=6,
                 color="black", ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.15",
                            facecolor="white", edgecolor="none",
                            alpha=0.85))
        ax.set_xlabel("per-slot MRE", fontsize=7)
        ax.set_title(label, fontsize=7.5, pad=2)
        ax.tick_params(axis="both", labelsize=6)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlim(0, 1.2)
        ax.set_ylim(0, ymax * 1.10)
        ax.grid(True, alpha=0.3)
    axs[0].set_ylabel("number of 5-min slots", fontsize=7)
    fig.tight_layout(pad=0.2, w_pad=0.3)
    # Header above the panels indicating the dashed line meaning.
    fig.subplots_adjust(top=0.80)
    fig.text(0.5, 0.93, "$--$ top-$24$ cut-off",
              ha="center", va="bottom", fontsize=7.5)
    fig.savefig(OUT_DETECT, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT_DETECT}  (top-24 cutoffs: "
          f"W1W2={cutoff_12:.3f}  W2W3={cutoff_23:.3f}  W3W4={cutoff_34:.3f})")

    # ---- REGION FIGURE: zoom on the 12-bin experiment window ----
    cols = np.array([lid2col[l] for l in REGION if l in lid2col], dtype=int)

    HOUR = 16
    times_window = [f"2018-09-{d:02d} {HOUR:02d}:{mm:02d}:00"
                     for d in (4, 11)
                     for mm in range(0, 60, 5)]

    def hour_series(day):
        ys = []
        for mm in range(0, 60, 5):
            t = f"2018-09-{day:02d} {HOUR:02d}:{mm:02d}:00"
            if t not in t2idx:
                ys.append(np.nan); continue
            row = M.getrow(t2idx[t]).toarray().ravel()
            ys.append(float(row[cols].sum()))
        return np.array(ys)

    y_ref_stack = np.stack(
        [hour_series(4), hour_series(11), hour_series(25)], axis=0)
    y_ref_mean = np.nanmean(y_ref_stack, axis=0)
    y_sep18 = hour_series(18)
    x_min = np.arange(0, 60, 5)

    fig2, ax = plt.subplots(figsize=(COL_W, 2.1))
    width = 2.0
    ax.bar(x_min - width/2, y_ref_mean, width, color="steelblue",
            alpha=0.85, label="avg of Sep 4, 11, 25")
    ax.bar(x_min + width/2, y_sep18,    width, color="crimson",
            alpha=0.85, label="Sep 18")
    ax.set_xticks(x_min)
    ax.set_xticklabels([f"{HOUR}:{mm:02d}" for mm in x_min],
                        rotation=45, ha="right", fontsize=6)
    ax.set_xlabel("5-min slot", fontsize=7)
    ax.set_ylabel("trips per 5-min bin", fontsize=7)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout(pad=0.3)
    fig2.savefig(OUT_REGION, bbox_inches="tight")
    plt.close(fig2)
    print(f"saved {OUT_REGION}  "
          f"(ref.avg total={float(np.nansum(y_ref_mean)):.0f}, "
          f"Sep 18 total={int(np.nansum(y_sep18))})")


if __name__ == "__main__":
    main()
