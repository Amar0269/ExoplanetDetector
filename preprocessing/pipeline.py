# pipeline.py

from .cleaner import clean_lightcurve
from .detrender import detrend_lightcurve
from .load_data import load_lightcurve
from .normalizer import normalize_flux

from utils.io import save_detrended_data, save_processed_data


def preprocess(file_path):
    time, flux, flux_err, quality = load_lightcurve(file_path)

    clean_time, clean_flux, clean_flux_err = clean_lightcurve(
        time,
        flux,
        flux_err,
        quality,
    )

    normalized_flux = normalize_flux(clean_flux)

    save_processed_data(file_path, clean_time, normalized_flux, clean_flux_err)

    detrended_flux, trend = detrend_lightcurve(clean_time, normalized_flux)

    save_detrended_data(file_path, clean_time, detrended_flux, trend)

    return clean_time, detrended_flux, trend