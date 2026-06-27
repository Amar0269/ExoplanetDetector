"""
parameters.py

Extracts transit parameters from a selected candidate.
"""


def estimate_parameters(candidate):
    """
    Estimate transit parameters.

    Parameters
    ----------
    candidate : dict
        Candidate returned by candidate.select_candidate()

    Returns
    -------
    dict
        Estimated transit parameters.
    """

    if candidate is None:
        return None

    parameters = {
        "orbital_period": candidate["period"],
        "transit_duration": candidate["duration"],
        "transit_epoch": candidate["transit_time"],
        "transit_depth": candidate["power"],   # Placeholder
    }

    return parameters