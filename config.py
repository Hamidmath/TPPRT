from pathlib import Path

# Project root is wherever this file lives
PROJECT_ROOT = Path(__file__).resolve().parent

# Data paths (absolute, resolved from project root)
DATA_DIR = PROJECT_ROOT / 'data'
GRAPH_FILE = DATA_DIR / 'city_graph_full.json'
POPULARITY_NPZ = DATA_DIR / 'popularity_results_smoothed.npz'
POPULARITY_NPZ_026 = DATA_DIR / 'popularity_results_smoothed_026.npz'
POPULARITY_RAW_NPZ = DATA_DIR / 'popularity_results.npz'
OUTPUT_FILE = DATA_DIR / 'two_phase_pagerank_vector.json'
NETWORK_XML = DATA_DIR / 'slc_network.xml'
MATCHED_ROUTES = DATA_DIR / 'matched_routes.json'

# Output directories
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = PROJECT_ROOT / 'figures'

# Default hyperparameters
DEFAULT_PARAMS = {
    'alpha_s': 0.0,
    'alpha_l': 0.0,
    'beta': 0.9,
    'damping': 0.80,
    'mu': 20.0,
    'tau': 1.0,
    'top_k': 100,
    'top_k_boost': 1.0,
    'max_iters': 100,
    'tol': 1e-6,
}
