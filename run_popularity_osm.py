"""Run pipeline/analyze_popularity.py against the OSM-matched JSON.

Outputs data/popularity_results_osm.npz. Same matrix format as the
existing data/popularity_results.npz so downstream code can swap
between them by changing the file path.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

config.MATCHED_ROUTES = config.DATA_DIR / "map-match" / "noise1.json"
config.POPULARITY_RAW_NPZ = config.DATA_DIR / "popularity_results_osm.npz"

from pipeline.analyze_popularity import main

main()
