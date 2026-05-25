"""Run evaluate_v1_forecast.py with the noise=3.5 OSM corpus
(beta=0.102, rho=0.101) against the OSM-derived smoothed popularity matrix.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

import sys as _sys
_USE_RAW = "--raw" in _sys.argv
if _USE_RAW:
    _sys.argv = [a for a in _sys.argv if a != "--raw"]
    config.POPULARITY_NPZ = config.DATA_DIR / "popularity_results_osm.npz"
else:
    config.POPULARITY_NPZ = config.DATA_DIR / "popularity_results_smoothed_osm.npz"

import analysis.evaluate_v1_forecast as ev1f
import numpy as _np

ev1f.BETA = 0.102
ev1f.RHO = 0.101

# Match the PDF table denominator floor (eps=1e-6, per EVALUATION_CONTRACT.md).
_EPS = 0.0 if "--eps0" in _sys.argv else 1e-6
if "--eps0" in _sys.argv:
    _sys.argv = [a for a in _sys.argv if a != "--eps0"]

def _mre_eps(truth, pred, top_k=100):
    if _EPS == 0.0:
        # Skip cells with truth==0 to avoid div-by-zero.
        mask = truth > 0
        if mask.sum() == 0:
            return 0.0, 0.0
        rel = _np.abs(truth[mask] - pred[mask]) / truth[mask]
        overall = float(_np.mean(rel))
        # top-k computed over the unmasked array's argsort, then filtered:
        top = _np.argsort(truth)[::-1][:top_k]
        rel_full = _np.abs(truth - pred) / _np.where(truth > 0, truth, 1.0)
        top_mre = float(_np.mean(rel_full[top]))
    else:
        rel = _np.abs(truth - pred) / (truth + _EPS)
        overall = float(_np.mean(rel))
        top = _np.argsort(truth)[::-1][:top_k]
        top_mre = float(_np.mean(rel[top]))
    return overall, top_mre

ev1f.mre = _mre_eps

# Set sys.argv to control argparse defaults explicitly.
sys.argv = ["run_eval_osm.py", "--num-frames", "49", "--seed", "42"]
ev1f.main()
