# load_data.py

from pathlib import Path

import numpy as np
from astropy.io import fits

from config import PDCSAP_FLUX_COL, PDCSAP_FLUX_ERR_COL, QUALITY_COL, TIME_COL


LIGHTCURVE_EXTENSION = "LIGHTCURVE"


def load_lightcurve(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Light curve file not found: {file_path}")

    required_columns = (
        TIME_COL,
        PDCSAP_FLUX_COL,
        PDCSAP_FLUX_ERR_COL,
        QUALITY_COL,
    )

    with fits.open(path) as hdul:
        data = hdul[LIGHTCURVE_EXTENSION].data

        missing_columns = [
            column for column in required_columns if column not in data.columns.names
        ]
        if missing_columns:
            raise KeyError(f"Missing required FITS column(s): {', '.join(missing_columns)}")

        time = np.asarray(data[TIME_COL])
        pdcsap_flux = np.asarray(data[PDCSAP_FLUX_COL])
        pdcsap_flux_err = np.asarray(data[PDCSAP_FLUX_ERR_COL])
        quality = np.asarray(data[QUALITY_COL])

    return time, pdcsap_flux, pdcsap_flux_err, quality