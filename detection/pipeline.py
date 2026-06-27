import numpy as np

from detection.bls import run_bls
from detection.candidate import select_candidate
from detection.parameters import estimate_parameters
from detection.confidence import estimate_confidence


def run_pipeline(file_path):
    """
    Complete exoplanet detection pipeline.

    Parameters
    ----------
    file_path : str
        Path to detrended light curve (.npz)

    Returns
    -------
    dict
        Dictionary containing all detection results.
    """

    # Load data
    data = np.load(file_path)

    time = data["time"]
    flux = data["flux"]

    # Step 1: Run BLS
    bls_result = run_bls(time, flux)

    # Step 2: Select best candidate
    candidate = select_candidate(bls_result)

    # Step 3: Estimate parameters
    parameters = estimate_parameters(candidate)

    # Step 4: Estimate confidence
    confidence = estimate_confidence(candidate)

    return {
        "bls": bls_result,
        "candidate": candidate,
        "parameters": parameters,
        "confidence": confidence
    }