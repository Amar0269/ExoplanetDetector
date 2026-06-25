# cleaner.py

import numpy as np

from config import VALID_QUALITY


def clean_lightcurve(time, flux, flux_err, quality):
    time_array = np.asarray(time)
    flux_array = np.asarray(flux)
    flux_err_array = np.asarray(flux_err)
    quality_array = np.asarray(quality)

    valid_mask = (
        np.isfinite(time_array)
        & np.isfinite(flux_array)
        & np.isfinite(flux_err_array)
        & (quality_array == VALID_QUALITY)
    )

    clean_time = time_array[valid_mask].copy()
    clean_flux = flux_array[valid_mask].copy()
    clean_flux_err = flux_err_array[valid_mask].copy()

    return clean_time, clean_flux, clean_flux_err