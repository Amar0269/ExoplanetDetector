# detrender.py

import numpy as np
from wotan import flatten

from config import DETREND_METHOD, WINDOW_LENGTH


def detrend_lightcurve(time, flux):
    time_array = np.asarray(time)
    flux_array = np.asarray(flux)

    if len(time_array) != len(flux_array):
        raise ValueError("time and flux must have the same length.")

    if len(time_array) == 0:
        raise ValueError("time and flux arrays must not be empty.")

    detrended_flux, trend = flatten(
        time_array,
        flux_array,
        method=DETREND_METHOD,
        window_length=WINDOW_LENGTH,
        return_trend=True,
    )

    return detrended_flux, trend