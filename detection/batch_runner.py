"""
batch_runner.py

Batch detection pipeline — processes all detrended .npz files and
generates CSV outputs for the classification stage.
"""

import re
import os
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits

from detection.pipeline import run_pipeline
from config import DETRENDED_DATA_DIR, PROCESSED_DATA_DIR


def extract_tic_id(filename):
    """Extract the TIC ID from a TESS filename."""
    match = re.search(r's\d{4}-(\d+)-', str(filename))
    if match:
        return int(match.group(1))
    return None


def extract_metadata_from_fits(tic_id, raw_dir="data/raw"):
    """
    Extract stellar metadata from the matching raw FITS file.

    Parameters
    ----------
    tic_id : int
        TIC ID to look up.
    raw_dir : str
        Path to raw FITS directory.

    Returns
    -------
    dict
        Stellar metadata from the FITS header.
    """
    raw_path = Path(raw_dir)
    tic_padded = str(tic_id).zfill(16)

    # Find matching FITS file
    fits_files = list(raw_path.glob(f"*{tic_padded}*"))
    if not fits_files:
        return {}

    try:
        with fits.open(fits_files[0]) as hdul:
            hdr = hdul[0].header
            return {
                "tessmag": hdr.get("TESSMAG"),
                "teff": hdr.get("TEFF"),
                "logg": hdr.get("LOGG"),
                "radius": hdr.get("RADIUS"),
                "ra": hdr.get("RA_OBJ"),
                "dec": hdr.get("DEC_OBJ"),
                "sector": hdr.get("SECTOR"),
                "camera": hdr.get("CAMERA"),
                "ccd": hdr.get("CCD"),
            }
    except Exception:
        return {}


def process_single_file(file_path):
    """
    Run the detection pipeline on a single .npz file with error handling.

    Parameters
    ----------
    file_path : str or Path
        Path to the detrended .npz file.

    Returns
    -------
    dict
        Flat dictionary of detection results.
    """
    file_path = Path(file_path)
    filename = file_path.name
    tic_id = extract_tic_id(filename)

    try:
        result = run_pipeline(str(file_path))

        # Load data for metadata
        data = np.load(file_path)
        num_obs = len(data["time"])
        time_baseline = float(data["time"][-1] - data["time"][0])

        # BLS results (always present)
        bls = result["bls"]
        row = {
            "tic_id": tic_id,
            "filename": filename,
            "best_period": float(bls["period"]),
            "transit_duration": float(bls["duration"]),
            "transit_epoch": float(bls["transit_time"]),
            "detection_power": float(bls["power"]),
            "num_observations": num_obs,
            "time_baseline": time_baseline,
        }

        # Candidate info (may be None)
        candidate = result["candidate"]
        if candidate is not None:
            row["is_candidate"] = True
            row["orbital_period"] = candidate["period"]
            row["transit_depth"] = candidate["power"]
        else:
            row["is_candidate"] = False
            row["orbital_period"] = float("nan")
            row["transit_depth"] = float("nan")

        # Parameters (may be None)
        params = result["parameters"]
        if params is not None:
            row["est_orbital_period"] = params["orbital_period"]
            row["est_transit_duration"] = params["transit_duration"]
            row["est_transit_epoch"] = params["transit_epoch"]
        else:
            row["est_orbital_period"] = float("nan")
            row["est_transit_duration"] = float("nan")
            row["est_transit_epoch"] = float("nan")

        # Confidence (may be None)
        conf = result["confidence"]
        if conf is not None:
            row["confidence_score"] = conf["confidence"]
        else:
            row["confidence_score"] = 0.0

        return row

    except Exception as e:
        print(f"  ERROR processing {filename}: {e}")
        return {
            "tic_id": tic_id,
            "filename": filename,
            "best_period": float("nan"),
            "transit_duration": float("nan"),
            "transit_epoch": float("nan"),
            "detection_power": float("nan"),
            "is_candidate": False,
            "orbital_period": float("nan"),
            "transit_depth": float("nan"),
            "est_orbital_period": float("nan"),
            "est_transit_duration": float("nan"),
            "est_transit_epoch": float("nan"),
            "confidence_score": 0.0,
            "num_observations": 0,
            "time_baseline": 0.0,
        }


def run_batch(input_dir=None, output_dir=None):
    """
    Run detection on all detrended .npz files and save CSV outputs.

    Parameters
    ----------
    input_dir : Path or str, optional
        Directory with detrended .npz files. Defaults to config value.
    output_dir : Path or str, optional
        Output directory for CSV files. Defaults to config value.
    """
    if input_dir is None:
        input_dir = DETRENDED_DATA_DIR
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .npz files
    npz_files = sorted(input_dir.glob("*.npz"))
    print(f"\n{'='*60}")
    print(f"  BATCH DETECTION PIPELINE")
    print(f"  Processing {len(npz_files)} light curves")
    print(f"{'='*60}\n")

    if not npz_files:
        print("No .npz files found. Exiting.")
        return

    # Process each file
    all_results = []
    for i, f in enumerate(npz_files, 1):
        tic_id = extract_tic_id(f.name)
        print(f"[{i:3d}/{len(npz_files)}] TIC {tic_id}...", end=" ", flush=True)
        row = process_single_file(f)
        all_results.append(row)
        status = "CANDIDATE" if row["is_candidate"] else "no detection"
        print(f"{status} (power={row['detection_power']:.6f})")

    # Build DataFrames
    results_df = pd.DataFrame(all_results)

    # --- detection_results.csv ---
    results_df.to_csv(output_dir / "detection_results.csv", index=False)
    print(f"\nSaved detection_results.csv ({len(results_df)} rows)")

    # --- candidates.csv ---
    candidates_df = results_df[results_df["is_candidate"] == True].copy()
    candidates_df.to_csv(output_dir / "candidates.csv", index=False)
    print(f"Saved candidates.csv ({len(candidates_df)} rows)")

    # --- metadata.csv ---
    metadata_rows = []
    for _, row in results_df.iterrows():
        tic_id = row["tic_id"]
        meta = extract_metadata_from_fits(tic_id)
        meta["tic_id"] = tic_id
        meta["num_observations"] = row["num_observations"]
        meta["time_baseline"] = row["time_baseline"]
        metadata_rows.append(meta)

    metadata_df = pd.DataFrame(metadata_rows)
    # Reorder columns
    cols = ["tic_id", "ra", "dec", "tessmag", "teff", "logg", "radius",
            "sector", "camera", "ccd", "num_observations", "time_baseline"]
    for c in cols:
        if c not in metadata_df.columns:
            metadata_df[c] = float("nan")
    metadata_df = metadata_df[cols]
    metadata_df.to_csv(output_dir / "metadata.csv", index=False)
    print(f"Saved metadata.csv ({len(metadata_df)} rows)")

    print(f"\n{'='*60}")
    print(f"  BATCH DETECTION COMPLETE")
    print(f"  Candidates: {len(candidates_df)} / {len(results_df)}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    return results_df


if __name__ == "__main__":
    run_batch()
