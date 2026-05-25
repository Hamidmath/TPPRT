"""Verify Table 10 reproducibility artifacts.

Checks:
  1. Input data files have the SHA256 hashes recorded in REPRODUCE_TABLE10.md.
  2. Each result JSON is parseable and exposes the expected key.
  3. The Overall MRE at that key matches the value in the paper to 1e-3.

Run from the project root:
    python verify_table10.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RES  = ROOT / "results"

EXPECTED_HASH = {
    "data/city_graph_full.json":
        "2f9023ce7b7543b7d4ca1489996867549e4431c2b28ff86ffa376a9038d8af3b",
    "data/popularity_results_smoothed_osm_gamma020.npz":
        "7e8fd4221dd3ba21c1805ca626a6d4cd17d8e321d7112bb3b2488e7da83af1a5",
}

# (row, label, json_path, key_path_as_list, expected_ov_mre)
ROWS = [
    (1,  "SP vanilla alpha=0.05",
     "results/vanilla_pr_two_alphas.json",
     ["results", "alpha_0.05", "overall_mean"], 1.072),
    (2,  "SP vanilla alpha=0.15",
     "results/vanilla_pr_two_alphas.json",
     ["results", "alpha_0.15", "overall_mean"], 0.923),
    (3,  "SP + lanes only",
     "results/sp_tune_three_ov.json",
     ["variants", "lanes_only", "overall_mean"], 0.916),
    (4,  "SP + speed only",
     "results/sp_tune_three_ov.json",
     ["variants", "speed_only", "overall_mean"], 0.870),
    (5,  "SP + both",
     "results/sp_tune_three_ov.json",
     ["variants", "both", "overall_mean"], 0.869),
    (6,  "SP + both, alpha=0.05 fixed",
     "results/sp_tune_alpha005_fixed.json",
     ["overall_mean"], 1.002),
    (7,  "TP vanilla beta=0.102 rho=0.101",
     "results/tp_eval_calibrated.json",
     ["overall_mean"], 1.041),
    (8,  "TP vanilla beta=rho=0.15",
     "results/tp_two_betas.json",
     ["results", "beta_0.15", "overall_mean"], 0.984),
    (9,  "TP + lanes only",
     "results/tp_tune_three_ov.json",
     ["variants", "lanes_only", "overall_mean"], 0.962),
    (12, "TP + both, (beta,rho)=(0.102,0.101) fixed",
     "results/tp_tune_calibrated_fixed.json",
     ["overall_mean"], 0.970),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dig(d, parts):
    cur = d
    for part in parts:
        cur = cur[part]
    return cur


def main() -> int:
    fail = 0

    print("=== Input data hashes ===")
    for rel, expected in EXPECTED_HASH.items():
        p = ROOT / rel
        if not p.exists():
            print(f"  [MISS] {rel}: file not found")
            fail += 1
            continue
        got = sha256(p)
        ok = got == expected
        print(f"  [{'OK ' if ok else 'BAD'}] {rel}")
        if not ok:
            print(f"    expected {expected}")
            print(f"    got      {got}")
            fail += 1

    print()
    print("=== Result JSON values ===")
    for row, label, rel_json, key, expected in ROWS:
        p = ROOT / rel_json
        if not p.exists():
            print(f"  [MISS] row {row:2d} {label}: {rel_json} not found")
            fail += 1
            continue
        try:
            with open(p) as f:
                d = json.load(f)
            got = float(dig(d, key))
        except KeyError:
            print(f"  [BAD ] row {row:2d} {label}: key {key} not in {rel_json}")
            fail += 1
            continue
        except Exception as e:
            print(f"  [BAD ] row {row:2d} {label}: {e}")
            fail += 1
            continue
        diff = abs(got - expected)
        ok = diff < 1e-3
        flag = "OK " if ok else "BAD"
        print(f"  [{flag}] row {row:2d} {label:42s}  paper={expected:.3f}  json={got:.4f}  |diff|={diff:.4f}")
        if not ok:
            fail += 1

    print()
    if fail == 0:
        print("All checks passed.")
        return 0
    print(f"{fail} check(s) failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
