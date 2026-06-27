"""
utils.py

Shared utility functions for the classification pipeline.
"""

import re
import random
import numpy as np
from scipy.interpolate import interp1d


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def extract_tic_id(filename):
    """
    Extract the TIC ID from a TESS filename.

    Parameters
    ----------
    filename : str
        TESS filename (e.g., 'tess2018...-s0001-0000000025245855-0120-s_lc_detrended.npz').

    Returns
    -------
    int or None
        TIC ID as integer, or None if extraction fails.
    """
    match = re.search(r's\d{4}-(\d+)-', str(filename))
    if match:
        return int(match.group(1))
    return None


def resample_flux(time, flux, target_length):
    """
    Resample a light curve to a fixed number of points using linear interpolation.

    Parameters
    ----------
    time : ndarray
        Original time array.
    flux : ndarray
        Original flux array.
    target_length : int
        Desired output length.

    Returns
    -------
    ndarray
        Resampled flux array of shape (target_length,).
    """
    if len(flux) == target_length:
        return flux.copy()

    # Create uniform time grid
    t_new = np.linspace(time[0], time[-1], target_length)

    # Interpolate
    interpolator = interp1d(time, flux, kind='linear', fill_value='extrapolate')
    return interpolator(t_new).astype(np.float32)


def normalize_flux(flux):
    """
    Normalize flux to zero-mean, unit-variance.

    Parameters
    ----------
    flux : ndarray
        Raw or detrended flux values.

    Returns
    -------
    ndarray
        Normalized flux.
    """
    mean = np.mean(flux)
    std = np.std(flux)
    if std < 1e-10:
        return flux - mean
    return ((flux - mean) / std).astype(np.float32)


def phase_fold(time, flux, period, epoch):
    """
    Phase-fold a light curve at a given period and epoch.

    Parameters
    ----------
    time : ndarray
        Observation times.
    flux : ndarray
        Flux values.
    period : float
        Folding period in days.
    epoch : float
        Transit epoch (time of first transit).

    Returns
    -------
    phase : ndarray
        Phase values in [-0.5, 0.5].
    folded_flux : ndarray
        Flux sorted by phase.
    """
    phase = ((time - epoch) / period) % 1.0
    phase[phase > 0.5] -= 1.0

    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]


def bin_phase_folded(phase, flux, num_bins=200):
    """
    Bin a phase-folded light curve for cleaner visualization.

    Parameters
    ----------
    phase : ndarray
        Phase values.
    flux : ndarray
        Flux values.
    num_bins : int
        Number of phase bins.

    Returns
    -------
    bin_centers : ndarray
        Center of each bin.
    bin_means : ndarray
        Mean flux in each bin.
    """
    bin_edges = np.linspace(phase.min(), phase.max(), num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = np.full(num_bins, np.nan)

    for i in range(num_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means[i] = np.nanmean(flux[mask])

    # Fill any remaining NaN bins with neighbors
    valid = ~np.isnan(bin_means)
    if valid.sum() > 0:
        bin_means = np.interp(
            bin_centers, bin_centers[valid], bin_means[valid]
        )

    return bin_centers, bin_means
