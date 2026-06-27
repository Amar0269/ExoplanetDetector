"""
confidence.py

Estimate confidence of a detected transit.
"""


def estimate_confidence(candidate):
    """
    Estimate confidence score of a detected transit.

    Parameters
    ----------
    candidate : dict

    Returns
    -------
    dict
    """

    if candidate is None:
        return None

    power = candidate["power"]

    # Normalize BLS power to percentage
    confidence = min(power / 0.01, 1.0) * 100

    return {
        "confidence": confidence,
        "power": power
    }