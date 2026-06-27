"""
predict.py

Batch classification of all transit candidates.

Usage:
    python predict.py

Automatically:
1. Loads the trained model from model/best_model.pth
2. Reads detection_results.csv and candidates.csv from data/processed/
3. Loads every detrended light curve from data/detrended/
4. Classifies each light curve
5. Generates per-candidate visualizations
6. Saves classified_results.csv

Output:
    results/classified_results.csv
    results/figures/TIC_*_classification.png
    results/classification_summary.png
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from classification.model import build_model
from classification.utils import (
    extract_tic_id,
    resample_flux,
    normalize_flux,
    set_seed,
)
from classification.visualize import plot_candidate_result, plot_batch_summary
from config import (
    DETRENDED_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    RESULTS_DIR,
    RESULTS_FIGURES_DIR,
    TARGET_SEQUENCE_LENGTH,
    NUM_CLASSES,
    CLASS_NAMES,
    RANDOM_SEED,
)


def load_trained_model(model_path=None, device=None):
    """
    Load the trained model from disk.

    Parameters
    ----------
    model_path : Path or str, optional
        Path to the saved model checkpoint.
    device : torch.device, optional
        Device to load model to.

    Returns
    -------
    model : ExoplanetClassifier
        Loaded model in eval mode.
    class_names : list of str
        Class names from the checkpoint.
    """
    if model_path is None:
        model_path = MODEL_DIR / "best_model.pth"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    input_length = checkpoint.get("input_length", TARGET_SEQUENCE_LENGTH)
    num_classes = checkpoint.get("num_classes", NUM_CLASSES)
    class_names = checkpoint.get("class_names", CLASS_NAMES)

    model = build_model(input_length, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names


def classify_light_curve(model, time, flux, target_length, device):
    """
    Classify a single light curve.

    Parameters
    ----------
    model : ExoplanetClassifier
        Trained model in eval mode.
    time : ndarray
        Time array.
    flux : ndarray
        Flux array.
    target_length : int
        Expected sequence length.
    device : torch.device
        Computation device.

    Returns
    -------
    predicted_class_idx : int
        Index of the predicted class.
    confidence : float
        Confidence score (max probability).
    probabilities : ndarray
        Probability for each class.
    """
    # Preprocess
    resampled = resample_flux(time, flux, target_length)
    normalized = normalize_flux(resampled)

    # Convert to tensor
    x = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
    x = x.to(device)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    predicted_idx = int(np.argmax(probs))
    confidence = float(probs[predicted_idx])

    return predicted_idx, confidence, probs


def main():
    """Main prediction pipeline."""
    print("=" * 60)
    print("  EXOPLANET SIGNAL CLASSIFIER — PREDICTION")
    print("=" * 60)

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # --- 1. Load Model ---
    model_path = MODEL_DIR / "best_model.pth"
    if not model_path.exists():
        print(f"\n  ERROR: No trained model found at {model_path}")
        print("  Run 'python train.py' first.")
        return

    print(f"  Loading model from {model_path}...")
    model, class_names = load_trained_model(model_path, device)
    print(f"  Model loaded. Classes: {class_names}")

    # --- 2. Load Detection Results ---
    det_results_path = PROCESSED_DATA_DIR / "detection_results.csv"
    candidates_path = PROCESSED_DATA_DIR / "candidates.csv"

    det_df = None
    if det_results_path.exists():
        det_df = pd.read_csv(det_results_path)
        print(f"  Loaded detection_results.csv ({len(det_df)} rows)")
    else:
        print(f"  Warning: {det_results_path} not found")

    cand_df = None
    if candidates_path.exists():
        cand_df = pd.read_csv(candidates_path)
        print(f"  Loaded candidates.csv ({len(cand_df)} rows)")

    # --- 3. Process All Light Curves ---
    npz_files = sorted(DETRENDED_DATA_DIR.glob("*.npz"))
    print(f"\n  Found {len(npz_files)} detrended light curves")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"\n  Classifying all light curves...\n")

    for i, npz_path in enumerate(npz_files, 1):
        tic_id = extract_tic_id(npz_path.name)
        data = np.load(npz_path)
        time = data["time"]
        flux = data["flux"]

        # Classify
        pred_idx, confidence, probs = classify_light_curve(
            model, time, flux, TARGET_SEQUENCE_LENGTH, device
        )
        predicted_class = class_names[pred_idx]

        # Get detection info if available
        period = None
        epoch = None
        det_power = None
        is_candidate = False

        if det_df is not None:
            det_row = det_df[det_df["tic_id"] == tic_id]
            if len(det_row) > 0:
                det_row = det_row.iloc[0]
                period = det_row.get("best_period")
                epoch = det_row.get("transit_epoch")
                det_power = det_row.get("detection_power")
                is_candidate = bool(det_row.get("is_candidate", False))

        # Build result row
        row = {
            "filename": npz_path.name,
            "tic_id": tic_id,
            "predicted_class": predicted_class,
            "confidence_score": confidence,
        }
        for j, cls in enumerate(class_names):
            col_name = f"prob_{cls.lower()}"
            row[col_name] = float(probs[j])

        row["is_detection_candidate"] = is_candidate
        row["detection_power"] = det_power
        row["best_period"] = period

        results.append(row)

        status = "★" if is_candidate else " "
        print(
            f"  [{i:3d}/{len(npz_files)}] {status} TIC {tic_id:>12d} → "
            f"{predicted_class:20s} ({confidence:.1%})"
        )

        # Generate visualization for each light curve
        try:
            plot_candidate_result(
                time=time,
                flux=flux,
                tic_id=tic_id,
                predicted_class=predicted_class,
                confidence=confidence,
                probabilities=probs,
                class_names=class_names,
                period=period if period and not np.isnan(period) else None,
                epoch=epoch if epoch and not np.isnan(epoch) else None,
                output_dir=RESULTS_FIGURES_DIR,
            )
        except Exception as e:
            print(f"    Warning: Could not generate plot for TIC {tic_id}: {e}")

    # --- 4. Save Results ---
    results_df = pd.DataFrame(results)
    output_path = RESULTS_DIR / "classified_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n  Saved classified_results.csv ({len(results_df)} rows) → {output_path}")

    # --- 5. Generate Batch Summary ---
    try:
        plot_batch_summary(results_df, class_names, RESULTS_DIR)
    except Exception as e:
        print(f"  Warning: Could not generate batch summary plot: {e}")

    # --- 6. Print Summary ---
    print(f"\n{'='*60}")
    print(f"  CLASSIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Class distribution:")
    for cls in class_names:
        count = (results_df["predicted_class"] == cls).sum()
        print(f"    {cls:20s}: {count}")

    det_candidates = results_df[results_df["is_detection_candidate"] == True]
    if len(det_candidates) > 0:
        print(f"\n  Detection candidates ({len(det_candidates)}):")
        for _, row in det_candidates.iterrows():
            print(
                f"    TIC {row['tic_id']:>12d}: {row['predicted_class']:20s} "
                f"({row['confidence_score']:.1%})"
            )

    print(f"\n  Output files:")
    print(f"    {output_path}")
    print(f"    {RESULTS_FIGURES_DIR}/ ({len(npz_files)} figures)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
