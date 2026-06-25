from pathlib import Path

from preprocessing.pipeline import preprocess
from preprocessing.visualize import plot_lightcurve

file = next(Path("data/raw").glob("*.fits"))

time, detrended_flux, trend = preprocess(file)

plot_lightcurve(time, detrended_flux, "Detrended Light Curve")