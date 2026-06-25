# normalizer.py

import numpy as np


def normalize_flux(flux):
    flux_array = np.asarray(flux)

    median_flux = np.median(flux_array)
    if median_flux == 0 or not np.isfinite(median_flux):
        raise ValueError("Flux median must be finite and nonzero.")

    normalized_flux = flux_array / median_flux

    return normalized_flux