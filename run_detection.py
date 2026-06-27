"""
Run the full detection pipeline on all detrended light curves.

Usage:
    python run_detection.py
"""

from detection.batch_runner import run_batch
from config import DETRENDED_DATA_DIR, PROCESSED_DATA_DIR

if __name__ == "__main__":
    run_batch(DETRENDED_DATA_DIR, PROCESSED_DATA_DIR)
