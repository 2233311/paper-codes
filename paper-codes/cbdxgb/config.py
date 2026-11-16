"""Central configuration. All paths are relative to the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic"
RESULTS_DIR = ROOT / "results"
TABLE_DIR = RESULTS_DIR / "tables"

WELLS = ["Well1", "Well2", "Well3", "Well4"]
TRAIN_WELLS = ["Well1", "Well2", "Well3"]
BLIND_WELL = "Well4"

# Calibrated CBD thresholds (inches) used by the paper workflow.
CBD_THRESHOLDS = {
    "Well1": 1.25,
    "Well2": 1.50,
    "Well3": 1.00,
    "Well4": 0.50,
}

# Enlarged-interval label expansion: +/-3 depth samples, never across a gap
# larger than MAX_GAP_M.
EXPAND_HALF = 3
MAX_GAP_M = 0.5

# Five input features and a 5-sample depth window -> 25 dimensions.
BASE_FEATURES = ["GR", "CNCF", "LG_RD", "LG_RS", "R_ratio"]
RISKY_PREFIXES = ("CNCF", "LG_RS")
WINDOW_HALF = 2

SEED = 42

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": SEED,
    "n_jobs": 4,
}

PHYS = {
    "rho_ma": 2.65,      # matrix density, g/cm3
    "rho_sh": 2.70,      # shale density, g/cm3
    "rho_f": 1.00,       # fluid density, g/cm3
    "dtc_f": 189.0,      # fluid transit time, us/ft
    "dtc_ma_default": 64.4,
    "gr_clean": 30.0,    # GR endpoint for Vsh = 0
    "gr_shale": 150.0,   # GR endpoint for Vsh = 1
}

TRAIN_FRACTION = 0.8
NORMAL_GR_BINS = [(0, 50), (50, 80), (80, 110), (110, 140), (140, 200)]
MULTIPLIERS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00]
BLIND_THRESHOLD_SCAN = [0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]
