"""Quantify how well the per-route link-count distributions fit a
geometric, with a Kolmogorov-Smirnov statistic and a chi-square
goodness-of-fit statistic.

Three variants per matching version (MATSim, OSM-link, OSM-subedge):
    - single-phase: K_total = L_up + L_down + 1
    - up-phase:     L_up
    - down-phase:   L_down

For each variant we fit the geometric on support {0, 1, 2, ...} by
moment matching (p = 1 / (1 + mean)) and report:
    - mean, var, geom_var = mean * (1 + mean)
    - KS statistic D = max_k |F_emp(k) - F_geom(k)|
    - chi-square statistic (Pearson) over bins with expected >= 5

The argument we want to make for the paper: the SPLIT KS values are
smaller than the SINGLE KS value, and OSM is at least as good as
MATSim. The aim is one number per row, plus a comparison.
"""
import numpy as np
from scipy import stats


DATA = {
    "MATSim XML matching, MATSim links": (
        "results/route_distribution_study/empirical_phase_split.bak.npz",
        "old_matsim",
    ),
    "OSM matching, MATSim links": (
        "results/route_distribution_study/empirical_phase_split.npz",
        "new_matsim",
    ),
    "OSM matching, OSM sub-edges": (
        "results/route_distribution_study/empirical_phase_split_osm_subedge.npz",
        "osm_sub",
    ),
}


def geom_cdf(k, p):
    """CDF of geometric on {0,1,2,...}: F(k) = 1 - (1-p)^(k+1)."""
    return 1.0 - (1.0 - p) ** (np.asarray(k, dtype=np.float64) + 1.0)


def ks_geom(data, p):
    """Discrete KS statistic between empirical CDF of data and
    geometric(p) on {0,1,2,...}."""
    data = np.asarray(data, dtype=np.int64)
    n = data.size
    max_k = int(data.max())
    counts = np.bincount(data, minlength=max_k + 2)
    emp_cdf = np.cumsum(counts) / n
    ks = np.arange(max_k + 2)
    theo_cdf = geom_cdf(ks, p)
    return float(np.max(np.abs(emp_cdf - theo_cdf)))


def chi2_geom(data, p, min_expected=5.0):
    """Pearson chi-square goodness-of-fit vs geometric(p), pooling
    upper tail bins so each expected count >= min_expected.
    Returns (chi2_stat, dof, p_value)."""
    data = np.asarray(data, dtype=np.int64)
    n = data.size
    max_k = int(data.max())
    counts = np.bincount(data, minlength=max_k + 1).astype(np.float64)
    ks = np.arange(max_k + 1)
    p_k = (1.0 - p) ** ks * p
    expected = p_k * n

    # Pool tail bins from the right until expected >= min_expected
    obs, exp = [], []
    acc_o = 0.0
    acc_e = 0.0
    pooling_tail = False
    for o, e in zip(counts, expected):
        acc_o += o
        acc_e += e
        if acc_e >= min_expected:
            obs.append(acc_o)
            exp.append(acc_e)
            acc_o = 0.0
            acc_e = 0.0
    # Anything left = the tail; add the missing tail mass of the geometric
    tail_geom = n - sum(expected)
    acc_e += tail_geom
    if acc_e > 0:
        obs.append(acc_o)
        exp.append(acc_e)

    obs = np.asarray(obs)
    exp = np.asarray(exp)
    # Re-normalise expected to match observed total (defensive)
    exp *= obs.sum() / exp.sum()
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    # dof = bins - 1 (free) - 1 (fitted parameter p)
    dof = max(len(obs) - 2, 1)
    pval = 1.0 - stats.chi2.cdf(chi2, dof)
    return chi2, dof, pval, len(obs)


def report_variant(name, data):
    n = data.size
    mean = float(data.mean())
    var = float(data.var(ddof=1))
    p = 1.0 / (1.0 + mean)
    geom_var = mean * (1.0 + mean)
    ks = ks_geom(data, p)
    chi2, dof, pval, nbins = chi2_geom(data, p)
    print(f"  {name:18s}  n={n:>7d}  mean={mean:6.3f}  "
          f"var={var:7.2f} (geom {geom_var:7.2f})  "
          f"p={p:.4f}  KS={ks:.4f}  "
          f"chi2={chi2:9.1f}/dof={dof:3d}  "
          f"chi2/dof={chi2/dof:6.2f}  bins={nbins}")
    return {
        "name": name,
        "n": n,
        "mean": mean,
        "var": var,
        "geom_var": geom_var,
        "p": p,
        "ks": ks,
        "chi2": chi2,
        "dof": dof,
        "chi2_per_dof": chi2 / dof,
        "pval": pval,
    }


def main():
    print("=" * 110)
    print("  GEOMETRIC GOODNESS-OF-FIT (per-route link-count distributions)")
    print("=" * 110)
    print()

    rows = []
    for label, (path, tag) in DATA.items():
        print(f"== {label} ==")
        z = np.load(path)
        Lu = z["L_up"].astype(np.int64)
        Ld = z["L_down"].astype(np.int64)
        Kt = Lu + Ld + 1
        s = report_variant("single (K_total)", Kt)
        s["dataset"] = tag; s["variant"] = "single"
        rows.append(s)
        s = report_variant("up   (L_up)", Lu)
        s["dataset"] = tag; s["variant"] = "up"
        rows.append(s)
        s = report_variant("down (L_down)", Ld)
        s["dataset"] = tag; s["variant"] = "down"
        rows.append(s)
        print()

    print()
    print("=" * 110)
    print("  AXIS A: single-phase vs split-phase (KS statistic, smaller = better)")
    print("=" * 110)
    print(f"  {'dataset':42s}  {'single':>10s}  {'up':>10s}  {'down':>10s}"
          f"  {'split avg':>10s}  {'improvement':>12s}")
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], {})[r["variant"]] = r
    rows_order = [
        ("old_matsim", "MATSim XML matching  ->  MATSim links"),
        ("new_matsim", "OSM matching         ->  MATSim links"),
        ("osm_sub",    "OSM matching         ->  OSM sub-edges"),
    ]
    for ds, label in rows_order:
        s = by_ds[ds]["single"]["ks"]
        u = by_ds[ds]["up"]["ks"]
        d = by_ds[ds]["down"]["ks"]
        avg = 0.5 * (u + d)
        impv = (s - avg) / s * 100.0
        print(f"  {label:42s}  {s:10.4f}  {u:10.4f}  {d:10.4f}"
              f"  {avg:10.4f}  {impv:11.1f}%")

    print()
    print("=" * 110)
    print("  AXIS B: matching change (single-phase KS, MATSim-link granularity, "
          "before vs after)")
    print("=" * 110)
    s_old = by_ds["old_matsim"]["single"]["ks"]
    s_new = by_ds["new_matsim"]["single"]["ks"]
    u_old = by_ds["old_matsim"]["up"]["ks"]
    u_new = by_ds["new_matsim"]["up"]["ks"]
    d_old = by_ds["old_matsim"]["down"]["ks"]
    d_new = by_ds["new_matsim"]["down"]["ks"]
    print(f"  single:  {s_old:.4f} -> {s_new:.4f}  "
          f"({(s_old - s_new)/s_old * 100:+.1f}% improvement)")
    print(f"  up:      {u_old:.4f} -> {u_new:.4f}  "
          f"({(u_old - u_new)/u_old * 100:+.1f}% improvement)")
    print(f"  down:    {d_old:.4f} -> {d_new:.4f}  "
          f"({(d_old - d_new)/d_old * 100:+.1f}% improvement)")

    print()
    print("=" * 110)
    print("  SUMMARY: chi^2 / dof (smaller = better fit; "
          "1.0 = perfect match of variance)")
    print("=" * 110)
    print(f"  {'dataset':42s}  {'single':>10s}  {'up':>10s}  {'down':>10s}"
          f"  {'split avg':>10s}  {'improvement':>12s}")
    for ds, label in rows_order:
        s = by_ds[ds]["single"]["chi2_per_dof"]
        u = by_ds[ds]["up"]["chi2_per_dof"]
        d = by_ds[ds]["down"]["chi2_per_dof"]
        avg = 0.5 * (u + d)
        impv = (s - avg) / s * 100.0
        print(f"  {label:42s}  {s:10.2f}  {u:10.2f}  {d:10.2f}"
              f"  {avg:10.2f}  {impv:11.1f}%")


if __name__ == "__main__":
    main()
