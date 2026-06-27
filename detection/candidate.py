"""
candidate.py

Filters and validates transit candidates detected by BLS.
"""

def select_candidate(bls_output, power_threshold=0.003):
    """
    Select the best transit candidate from the BLS output.

    Parameters
    ----------
    bls_output : dict
        Output dictionary returned by run_bls().

    power_threshold : float
        Minimum detection power required.

    Returns
    -------
    dict or None
        Candidate information if detection is significant,
        otherwise None.
    """

    if bls_output["power"] < power_threshold:
        return None

    candidate = {
    "period": float(bls_output["period"]),
    "duration": float(bls_output["duration"]),
    "transit_time": float(bls_output["transit_time"]),
    "power": float(bls_output["power"]),
}

    return candidate