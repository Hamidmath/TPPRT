import json
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results", "sigma_residual")


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "residual_full_noise*.json"))):
        d = json.load(open(p))
        s = d["declared_sigma"]
        m = d["stats_after_filter"]["sigma_std"]
        rows.append((s, m, m - s))
    rows.sort()
    chosen = next((i for i, (s, m, g) in enumerate(rows) if g < 0), None)

    print("declared  measured  gap     verdict")
    for i, (s, m, g) in enumerate(rows):
        if i == chosen:
            v = "first sigma > sigma_std (chosen)"
        elif g > 0.5:
            v = "too tight"
        elif g > 0:
            v = "slightly too tight"
        else:
            v = "too loose"
        print(f"{s:<8}  {m:.2f}      {g:+.2f}   {v}")

    print("\nLaTeX rows:")
    for s, m, g in rows:
        print(f"{s:.2f} & {m:.2f} & ${g:+.2f}$ \\\\")


if __name__ == "__main__":
    main()
