# config.py

from pathlib import Path

TIME_COL = "TIME"
PDCSAP_FLUX_COL = "PDCSAP_FLUX"
PDCSAP_FLUX_ERR_COL = "PDCSAP_FLUX_ERR"
QUALITY_COL = "QUALITY"

VALID_QUALITY = 0

DETREND_METHOD = "biweight"
WINDOW_LENGTH = 0.5

FIG_SIZE = (10, 5)
MARKER_SIZE = 2

PROCESSED_DATA_DIR = Path("data/processed")
DETRENDED_DATA_DIR = Path("data/detrended")
RAW_DATA_DIR = Path("data/raw")

# ------------------------------------------------------------------
# Detection
# ------------------------------------------------------------------
BLS_POWER_THRESHOLD = 0.003

# ------------------------------------------------------------------
# Classification — Directories
# ------------------------------------------------------------------
MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"

# ------------------------------------------------------------------
# Classification — Model & Data
# ------------------------------------------------------------------
TARGET_SEQUENCE_LENGTH = 2048
NUM_CLASSES = 5
CLASS_NAMES = ["Transit", "EclipsingBinary", "Blend", "FalsePositive", "Other"]

# ------------------------------------------------------------------
# Classification — Training Hyperparameters
# ------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
LR_REDUCE_PATIENCE = 7
LR_REDUCE_FACTOR = 0.5
MIN_SAMPLES_PER_CLASS = 100
RANDOM_SEED = 42