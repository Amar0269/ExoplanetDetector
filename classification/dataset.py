"""
dataset.py

Dataset preparation for the exoplanet classification pipeline.

Handles:
- Loading detrended .npz light curves
- Label acquisition (TOI catalog cross-match + heuristics)
- Synthetic data augmentation for class balancing
- Normalization and resampling to fixed sequence length
- Stratified train/validation/test splitting
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from classification.utils import (
    extract_tic_id,
    resample_flux,
    normalize_flux,
    set_seed,
)
from config import (
    DETRENDED_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TARGET_SEQUENCE_LENGTH,
    CLASS_NAMES,
    MIN_SAMPLES_PER_CLASS,
    RANDOM_SEED,
)


# ================================================================
# Label Acquisition
# ================================================================

def fetch_toi_labels(tic_ids):
    """
    Query the NASA Exoplanet Archive for TOI dispositions.

    Parameters
    ----------
    tic_ids : list of int
        TIC IDs to look up.

    Returns
    -------
    dict
        Mapping of {tic_id: class_label} for TIC IDs found in the TOI catalog.
    """
    labels = {}

    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

        # Query TOI catalog for our TIC IDs
        tic_str = ",".join(str(t) for t in tic_ids)
        query = (
            f"SELECT tid, tfopwg_disp, toi "
            f"FROM toi "
            f"WHERE tid IN ({tic_str})"
        )

        try:
            result = NasaExoplanetArchive.query_criteria(
                table="toi",
                select="tid,tfopwg_disp,toi",
                where=f"tid in ({tic_str})"
            )

            if result is not None and len(result) > 0:
                for row in result:
                    tic_id = int(row["tid"])
                    disp = str(row["tfopwg_disp"]).strip().upper()

                    if disp in ("KP", "CP"):
                        labels[tic_id] = "Transit"
                    elif disp == "PC":
                        labels[tic_id] = "Transit"
                    elif disp in ("FP", "FA"):
                        labels[tic_id] = "FalsePositive"
                    elif disp == "EB":
                        labels[tic_id] = "EclipsingBinary"
                    else:
                        labels[tic_id] = "Other"

                print(f"  TOI catalog: found {len(labels)} matches")
        except Exception as e:
            print(f"  TOI catalog query failed: {e}")

    except ImportError:
        print("  astroquery not installed — skipping TOI catalog lookup")

    return labels


def assign_heuristic_labels(tic_ids, detection_results_path=None):
    """
    Assign heuristic labels based on BLS detection results and
    stellar parameters.

    Parameters
    ----------
    tic_ids : list of int
        All TIC IDs in the dataset.
    detection_results_path : str or Path, optional
        Path to detection_results.csv.

    Returns
    -------
    dict
        Mapping of {tic_id: class_label}.
    """
    labels = {}

    # Try to load detection results
    det_df = None
    if detection_results_path and Path(detection_results_path).exists():
        det_df = pd.read_csv(detection_results_path)
        det_df = det_df.set_index("tic_id")

    for tic_id in tic_ids:
        if det_df is not None and tic_id in det_df.index:
            row = det_df.loc[tic_id]
            # Handle case where multiple rows exist for same TIC ID
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            is_candidate = bool(row.get("is_candidate", False))
            power = float(row.get("detection_power", 0))
            period = float(row.get("best_period", 0))
            confidence = float(row.get("confidence_score", 0))

            if not is_candidate:
                # Below detection threshold
                labels[tic_id] = "Other"
            elif period < 1.0 and power > 0.005:
                # Very short period + strong signal → likely EB
                labels[tic_id] = "EclipsingBinary"
            elif confidence >= 50.0:
                # High confidence detection
                labels[tic_id] = "Transit"
            elif confidence >= 20.0:
                # Moderate confidence — could be blend or weak transit
                labels[tic_id] = "Blend"
            else:
                # Low confidence detection
                labels[tic_id] = "FalsePositive"
        else:
            labels[tic_id] = "Other"

    return labels


# ================================================================
# Synthetic Data Augmentation
# ================================================================

def _inject_transit(time, flux, rng):
    """Inject a synthetic box-shaped transit signal."""
    augmented = flux.copy()
    period = rng.uniform(1.0, 15.0)
    depth = rng.uniform(0.001, 0.02)
    duration_frac = rng.uniform(0.01, 0.05)
    epoch = time[0] + rng.uniform(0, period)

    phase = ((time - epoch) / period) % 1.0
    in_transit = phase < duration_frac
    augmented[in_transit] -= depth

    # Add small noise
    augmented += rng.normal(0, 0.0002, len(augmented))
    return augmented


def _inject_eclipsing_binary(time, flux, rng):
    """Inject a synthetic eclipsing binary signal (sinusoidal + eclipse)."""
    augmented = flux.copy()
    period = rng.uniform(0.3, 2.0)  # Short period
    amplitude = rng.uniform(0.005, 0.05)
    epoch = time[0] + rng.uniform(0, period)

    phase = ((time - epoch) / period) % 1.0
    # Primary eclipse
    primary_mask = phase < 0.03
    augmented[primary_mask] -= amplitude
    # Secondary eclipse (shallower)
    secondary_mask = (phase > 0.47) & (phase < 0.53)
    augmented[secondary_mask] -= amplitude * 0.3
    # Ellipsoidal variation
    augmented += amplitude * 0.1 * np.cos(4 * np.pi * phase)
    augmented += rng.normal(0, 0.0003, len(augmented))
    return augmented


def _inject_blend(time, flux, rng):
    """Create a blended source by superimposing two signals."""
    augmented = flux.copy()
    # Signal 1: diluted transit
    p1 = rng.uniform(2.0, 10.0)
    depth1 = rng.uniform(0.0005, 0.005)  # Shallow due to dilution
    epoch1 = time[0] + rng.uniform(0, p1)
    phase1 = ((time - epoch1) / p1) % 1.0
    augmented[phase1 < 0.02] -= depth1

    # Signal 2: another variability source
    p2 = rng.uniform(0.5, 3.0)
    amp2 = rng.uniform(0.001, 0.003)
    augmented += amp2 * np.sin(2 * np.pi * time / p2)
    augmented += rng.normal(0, 0.0003, len(augmented))
    return augmented


def _inject_false_positive(time, flux, rng):
    """Inject systematic artifacts mimicking false positives."""
    augmented = flux.copy()
    # Momentum dump-like feature
    n_artifacts = rng.integers(2, 6)
    for _ in range(n_artifacts):
        idx = rng.integers(100, len(augmented) - 100)
        width = rng.integers(5, 50)
        depth = rng.uniform(0.002, 0.01)
        augmented[idx:idx + width] -= depth
        # Recovery ramp
        ramp_len = rng.integers(10, 50)
        if idx + width + ramp_len < len(augmented):
            ramp = np.linspace(-depth * 0.5, 0, ramp_len)
            augmented[idx + width:idx + width + ramp_len] += ramp
    augmented += rng.normal(0, 0.0005, len(augmented))
    return augmented


def _inject_other_variability(time, flux, rng):
    """Inject stellar variability (pulsation, rotation)."""
    augmented = flux.copy()
    # Rotational modulation
    rot_period = rng.uniform(1.0, 15.0)
    rot_amp = rng.uniform(0.001, 0.01)
    augmented += rot_amp * np.sin(2 * np.pi * time / rot_period)

    # Add harmonics
    if rng.random() > 0.5:
        augmented += rot_amp * 0.3 * np.sin(4 * np.pi * time / rot_period)

    augmented += rng.normal(0, 0.0003, len(augmented))
    return augmented


_AUGMENTATION_FNS = {
    "Transit": _inject_transit,
    "EclipsingBinary": _inject_eclipsing_binary,
    "Blend": _inject_blend,
    "FalsePositive": _inject_false_positive,
    "Other": _inject_other_variability,
}


def generate_augmented_samples(
    time_arrays, flux_arrays, labels, class_names, min_per_class, rng
):
    """
    Augment the dataset so every class has at least min_per_class samples.

    Parameters
    ----------
    time_arrays : list of ndarray
        Original time arrays (pre-resampling).
    flux_arrays : list of ndarray
        Original flux arrays (pre-resampling).
    labels : list of str
        Class label for each sample.
    class_names : list of str
        All class names.
    min_per_class : int
        Minimum samples per class.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    aug_times, aug_fluxes, aug_labels : lists
        Augmented data arrays and labels.
    """
    aug_times = list(time_arrays)
    aug_fluxes = list(flux_arrays)
    aug_labels = list(labels)

    for cls in class_names:
        count = sum(1 for l in aug_labels if l == cls)
        needed = max(0, min_per_class - count)

        if needed == 0:
            continue

        print(f"    Augmenting class '{cls}': {count} → {count + needed}")

        # Source light curves to augment from
        # Prefer same-class originals, fall back to all
        cls_indices = [i for i, l in enumerate(labels) if l == cls]
        if not cls_indices:
            cls_indices = list(range(len(time_arrays)))

        aug_fn = _AUGMENTATION_FNS[cls]

        for _ in range(needed):
            src_idx = rng.choice(cls_indices)
            t = time_arrays[src_idx]
            f = flux_arrays[src_idx]
            aug_f = aug_fn(t, f, rng)
            aug_times.append(t)
            aug_fluxes.append(aug_f)
            aug_labels.append(cls)

    return aug_times, aug_fluxes, aug_labels


# ================================================================
# Dataset Preparation
# ================================================================

def load_and_preprocess(file_path, target_length=TARGET_SEQUENCE_LENGTH):
    """
    Load a .npz light curve and preprocess it.

    Parameters
    ----------
    file_path : str or Path
        Path to the detrended .npz file.
    target_length : int
        Desired output sequence length.

    Returns
    -------
    flux : ndarray
        Preprocessed flux array of shape (target_length,).
    time : ndarray
        Original time array (for phase folding etc.).
    raw_flux : ndarray
        Original flux array (before normalization).
    """
    data = np.load(file_path)
    time = data["time"]
    flux = data["flux"]

    # Resample to fixed length
    resampled = resample_flux(time, flux, target_length)

    # Normalize
    normalized = normalize_flux(resampled)

    return normalized, time, flux


def prepare_dataset(
    detrended_dir=None,
    processed_dir=None,
    target_length=TARGET_SEQUENCE_LENGTH,
    min_per_class=MIN_SAMPLES_PER_CLASS,
    seed=RANDOM_SEED,
):
    """
    Master function: load all light curves, acquire labels, augment,
    and prepare arrays for training.

    Parameters
    ----------
    detrended_dir : Path or str, optional
        Directory with detrended .npz files.
    processed_dir : Path or str, optional
        Directory with detection CSVs.
    target_length : int
        Sequence length for the model.
    min_per_class : int
        Minimum samples per class after augmentation.
    seed : int
        Random seed.

    Returns
    -------
    X : ndarray
        Shape (N, target_length, 1) — input sequences.
    y : ndarray
        Shape (N,) — integer class labels.
    label_encoder : LabelEncoder
        Fitted label encoder.
    filenames : list of str
        Filename for each sample (empty string for augmented samples).
    """
    if detrended_dir is None:
        detrended_dir = DETRENDED_DATA_DIR
    if processed_dir is None:
        processed_dir = PROCESSED_DATA_DIR

    detrended_dir = Path(detrended_dir)
    processed_dir = Path(processed_dir)
    rng = np.random.default_rng(seed)

    print("\n=== DATASET PREPARATION ===\n")

    # 1. Load all .npz files
    npz_files = sorted(detrended_dir.glob("*.npz"))
    print(f"  Found {len(npz_files)} detrended light curves")

    if not npz_files:
        raise FileNotFoundError(f"No .npz files in {detrended_dir}")

    time_arrays = []
    flux_arrays = []
    tic_ids = []
    fnames = []

    for f in npz_files:
        data = np.load(f)
        time_arrays.append(data["time"])
        flux_arrays.append(data["flux"])
        tic_ids.append(extract_tic_id(f.name))
        fnames.append(f.name)

    # 2. Acquire labels
    print("\n  Acquiring labels...")

    # Try TOI catalog first
    toi_labels = fetch_toi_labels(tic_ids)

    # Fill remaining with heuristics
    det_results_path = processed_dir / "detection_results.csv"
    heuristic_labels = assign_heuristic_labels(tic_ids, det_results_path)

    # Merge: TOI takes priority over heuristic
    labels = []
    for tic_id in tic_ids:
        if tic_id in toi_labels:
            labels.append(toi_labels[tic_id])
        elif tic_id in heuristic_labels:
            labels.append(heuristic_labels[tic_id])
        else:
            labels.append("Other")

    # Report distribution
    from collections import Counter
    dist = Counter(labels)
    print(f"\n  Label distribution (before augmentation):")
    for cls in CLASS_NAMES:
        print(f"    {cls:20s}: {dist.get(cls, 0)}")

    # 3. Augment to balance classes
    print(f"\n  Augmenting to {min_per_class} samples per class...")
    aug_times, aug_fluxes, aug_labels = generate_augmented_samples(
        time_arrays, flux_arrays, labels, CLASS_NAMES, min_per_class, rng
    )

    # Track which are original vs augmented
    aug_fnames = list(fnames) + [""] * (len(aug_labels) - len(fnames))

    # Report augmented distribution
    aug_dist = Counter(aug_labels)
    print(f"\n  Label distribution (after augmentation):")
    total = 0
    for cls in CLASS_NAMES:
        c = aug_dist.get(cls, 0)
        total += c
        print(f"    {cls:20s}: {c}")
    print(f"    {'TOTAL':20s}: {total}")

    # 4. Preprocess all sequences
    print(f"\n  Resampling to {target_length} points + normalizing...")
    X_list = []
    for t, f in zip(aug_times, aug_fluxes):
        resampled = resample_flux(t, f, target_length)
        normalized = normalize_flux(resampled)
        X_list.append(normalized)

    X = np.array(X_list, dtype=np.float32)
    X = X.reshape(-1, target_length, 1)  # (N, seq_len, 1) for Conv1D

    # 5. Encode labels
    le = LabelEncoder()
    le.fit(CLASS_NAMES)  # Fixed order
    y = le.transform(aug_labels)

    print(f"\n  Final dataset: X={X.shape}, y={y.shape}")
    print("  Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    print("\n=== DATASET READY ===\n")

    return X, y, le, aug_fnames


def split_dataset(X, y, val_frac=0.15, test_frac=0.15, seed=RANDOM_SEED):
    """
    Stratified train/validation/test split.

    Parameters
    ----------
    X : ndarray
        Input features.
    y : ndarray
        Integer labels.
    val_frac : float
        Fraction for validation.
    test_frac : float
        Fraction for test.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'.
    """
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_frac,
        random_state=seed,
        stratify=y,
    )

    # Second split: train vs val
    val_relative = val_frac / (1 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative,
        random_state=seed,
        stratify=y_trainval,
    )

    print(f"  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def get_class_names():
    """Return the ordered list of class names."""
    return CLASS_NAMES
