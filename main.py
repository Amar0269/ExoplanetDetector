from pathlib import Path
import numpy as np

from preprocessing.pipeline import preprocess
from preprocessing.visualize import plot_lightcurve

from detection.bls import run_bls
from detection.candidate import select_candidate
from detection.parameters import estimate_parameters
from detection.confidence import estimate_confidence
from detection.visualize import plot_periodogram

# Load first FITS file
file = next(Path("data/raw").glob("*.fits"))

# ----------------------------
# Preprocessing
# ----------------------------
time, detrended_flux, trend = preprocess(file)

plot_lightcurve(
    time,
    detrended_flux,
    "Detrended Light Curve"
)

# ----------------------------
# Exoplanet Detection
# ----------------------------
bls_output = run_bls(time, detrended_flux)

candidate = select_candidate(bls_output)

parameters = estimate_parameters(candidate)

confidence = estimate_confidence(candidate)

print("\n========== BLS RESULT ==========")
print(f"Best Period     : {bls_output['period']:.4f} days")
print(f"Transit Duration: {bls_output['duration']:.4f} days")
print(f"Transit Epoch   : {bls_output['transit_time']:.4f}")
print(f"Detection Power : {bls_output['power']:.4f}")

print("\n========== CANDIDATE ==========")
print(candidate)

print("\n========== PARAMETERS ==========")
print(parameters)

print("\n========== CONFIDENCE ==========")
print(confidence)

plot_periodogram(bls_output)
