"""Generate smoothed popularity matrix from the OSM raw popularity npz.

Reads:  data/popularity_results_osm.npz
Writes: data/popularity_results_smoothed_osm.npz
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

config.POPULARITY_RAW_NPZ = config.DATA_DIR / "popularity_results_osm.npz"
config.POPULARITY_NPZ = config.DATA_DIR / "popularity_results_smoothed_osm.npz"

from pipeline.generate_smoothed import main

main()
