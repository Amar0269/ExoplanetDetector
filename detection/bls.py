import numpy as np
from astropy.timeseries import BoxLeastSquares


def run_bls(time, flux):
    """
    Run Box Least Squares on a detrended light curve.

    Parameters
    ----------
    time : ndarray
        Observation times.

    flux : ndarray
        Detrended normalized flux.

    Returns
    -------
    dict
        Dictionary containing the best transit candidate
        and the full BLS periodogram.
    """

    # Candidate periods (days)
    periods = np.linspace(0.5, 20, 5000)

    # Candidate transit durations (days)
    durations = np.linspace(0.01, 0.15, 20)

    # Create BLS object
    bls = BoxLeastSquares(time, flux)

    # Compute periodogram
    result = bls.power(periods, durations)

    # Strongest peak
    best_index = np.argmax(result.power)

    return {
        "period": result.period[best_index],
        "duration": result.duration[best_index],
        "transit_time": result.transit_time[best_index],
        "power": result.power[best_index],
        "periods": result.period,
        "powers": result.power,
    }