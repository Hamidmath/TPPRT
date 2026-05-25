# Pipeline: update paper numbers when a new `obs_noise` matched corpus arrives

This is the exact sequence used to bring the paper to noise=4. Repeat it for
any new noise level (`{NOISE}` = `3p75`, `5`, etc.) once all CHPC chunks
finish.

The user-visible end product is `documents/walkthrough2/main.pdf` with every
data-dependent number, figure, and table rewritten against the new corpus.

---

## 0. Inputs you need locally

| File | What it is |
|---|---|
| `data/matched_routes_osm_noise{NOISE}_chunk*.json` | All chunks pulled from CHPC into `data/` |
| `data/slc_network.xml` | MATSim XML network (already local) |
| `data/network_osm.pkl` | High-res OSM pickle (already local) |
| `data/compressed_routes.parquet` | Raw GPS (already local) |

If chunks are still on CHPC, pull them first:
```bash
rsync -ah notchpeak.chpc.utah.edu:'tppr/results/matched_routes_osm_noise{NOISE}_chunk*.json' \
    data/ --exclude='*.partial.json'
```

## 1. Aggregate chunks into one combined JSON

```bash
python3 analysis/aggregate_match_chunks.py --noise {NOISE}
```
Output: `data/matched_routes_osm_noise{NOISE}_combined.json`.
Then (optional) delete the chunks if disk-pressed:
```bash
rm data/matched_routes_osm_noise{NOISE}_chunk*.json
```

## 2. Recompute the per-trip phase split (drives Fig 6, Fig 7, Table 1, beta/rho)

Backup the current canonical file before overwriting:
```bash
cp results/route_distribution_study/empirical_phase_split.npz \
   results/route_distribution_study/empirical_phase_split.bak_prev.npz
```
Run:
```bash
python3 analysis/compute_phase_split.py \
    --matched data/matched_routes_osm_noise{NOISE}_combined.json \
    --xml data/slc_network.xml \
    --out results/route_distribution_study/empirical_phase_split.npz \
    --out-all-k results/route_distribution_study/empirical_phase_split_all_K.npz
```
Read off from the stdout:
- `E[L_up]`, `E[L_down]`
- `beta = 1/(1+E[L_up])`, `rho = 1/(1+E[L_down])`
- Trips kept (K>=3) and trips skipped (K<3)

## 3. Recompute KS goodness-of-fit (drives Table 1 row 2)

```bash
python3 analysis/quantify_geometric_fit.py
```
Read off the row labelled "OSM matching, MATSim links":
- `n`, `D_single`, `D_up`, `D_down`, `variance`, `geom_variance`
- "AXIS A" `improvement` percentage (split-vs-single)
- "AXIS B" `improvement` percentages (low-res to high-res, for D_up and D_down)
- Overdispersion ratio = var / geom_var (paper text references this)

## 4. Regenerate Fig 6 and Fig 7

```bash
cd documents/walkthrough2/figures
python3 gen_phase_histograms.py
cp ../../../results/route_distribution_study/empirical_phase_split_all_K.npz phase_split_all_K.npz
python3 gen_phase_histograms_all_K.py
cd ../../..
```

## 5. Rebuild the popularity matrix (raw)

```bash
cp data/popularity_results.npz data/popularity_results.bak_prev.npz
python3 -c "
import sys, time
from pathlib import Path
sys.path.insert(0, '.')
import config
config.MATCHED_ROUTES = Path('data/matched_routes_osm_noise{NOISE}_combined.json')
config.POPULARITY_RAW_NPZ = Path('data/popularity_results.npz')
from pipeline.analyze_popularity import collect_trip_sets, save_popularity_matrix
trips = collect_trip_sets(str(config.MATCHED_ROUTES))
save_popularity_matrix(trips, str(config.POPULARITY_RAW_NPZ))
"
```

## 6. Install the new popularity as the OSM-tagged canonical

```bash
cp data/popularity_results_osm.npz data/popularity_results_osm.bak_prev.npz
cp data/popularity_results_smoothed_osm.npz data/popularity_results_smoothed_osm.bak_prev.npz
cp data/popularity_results.npz data/popularity_results_osm.npz
```

## 7. Regenerate the smoothed popularity matrix

```bash
python3 run_smoothed_osm.py
```

## 8. Run the contract evaluation with the new beta and rho

Edit `analysis/run_contract_eval.py` so its top-of-file constants match:
- `BETA = {new_beta}`
- `RHO  = {new_rho}`

Then:
```bash
python3 analysis/run_contract_eval.py
```
The script prints a contract-style table (3 configs x 3 comparisons x
{T-100, Ov}) and saves `results/contract_eval/contract_eval_noise{NOISE}.json`.

## 9. Edit the paper

Open `documents/walkthrough2/main.tex` and apply the following replacements
(use **string** matches with `Edit`):

| Where | Old value | New value |
|---|---|---|
| Dataset section ("survive map matching") | `485{,}931` | new total matched |
| §"What C does and does not measure" | `485{,}931`-trip | new total matched |
| §"Calibrating beta and rho" K<3 line | `88{,}234` or prev | new K<3 |
| same line, K>=3 line | `637{,}914` or prev | new K>=3 |
| same section, `E[L_up]` | `10.26` or prev | new |
| `E[L_down]` | `10.32` or prev | new |
| `\beta = ... 0.089` (search all occurrences) | `0.089` | new beta |
| `\rho = ... 0.088` (search all occurrences) | `0.088` | new rho |
| Fig 6 caption (`...-trip calibration sample`) | `637{,}914` | new K>=3 |
| Fig 7 caption (`... matched trips`) | `726,148` | new total matched |
| Table 1 row 2 | `637{,}914 & 0.127 & 0.034 & 0.032` | new from step 3 |
| Two-findings paragraph: `74\%`, `74\%`, `69\%` | step 3 improvements | new % |
| Overdispersion: `2.07 \to 0.91` | step 3 ratio | new |
| Table 2 (all 18 MRE values) | from step 8 | new |
| "Reconstruction is faithful" paragraph | `~6%`, `~9%`, `~0.60` | new from step 8 |
| "Diffusion makes the overall MRE meaningful" range | `0.22--1.03` | new from step 8 |
| Conclusion `Top-100 MRE` phrase | `~9%`, `~6%` | new from step 8 |

The β/ρ value occurs in multiple places. Use Edit's exact-string mode for
each unique surrounding context.

If you also have a comparison appendix at end of paper, append a new
old-vs-new column for this run.

## 10. Rebuild the PDF

```bash
cd documents/walkthrough2
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
Should compile to 15 pages, no errors. Verify Fig 6 legend reads
`Geom(beta = NEW)` and `Geom(rho = NEW)`.

---

## Notes / gotchas

- The `analyze_popularity.py` step loads the entire combined JSON into
  memory (~5-10 GB peak for 700k routes). Should fit on 16 GB local. If it
  OOMs, refactor to stream line by line.
- The phase-split script also `json.load`s the file at once (same caveat).
- `quantify_geometric_fit.py` reads three npz files. Row 1 (low-res
  matching) is fixed and reflects the original MATSim-XML matched corpus.
  Row 3 (OSM sub-edges) is the old noise=1 sub-edge analysis; if you want
  it updated for the new noise level, you also need to re-run
  `analysis/compute_phase_split_osm_subedge.py` first.
- Backups (`*.bak_prev.npz`) are kept so an undo is trivial: just copy
  the bak back over.
- The contract eval samples 84 random bins with seed 42; the sample is
  the same across all noise levels, so the comparison is apples-to-apples.
