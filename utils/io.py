# io.py

from pathlib import Path

import numpy as np

from config import DETRENDED_DATA_DIR, PROCESSED_DATA_DIR


def save_processed_data(file_name, time, flux, flux_err):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(file_name).name.replace(".fits", "")
    output_path = PROCESSED_DATA_DIR / f"{stem}_processed.npz"
    np.savez_compressed(output_path, time=time, flux=flux, flux_err=flux_err)


def save_detrended_data(file_name, time, detrended_flux, trend):
    DETRENDED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(file_name).name.replace(".fits", "")
    output_path = DETRENDED_DATA_DIR / f"{stem}_detrended.npz"
    np.savez_compressed(output_path, time=time, flux=detrended_flux, trend=trend)
