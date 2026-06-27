"""
visualize.py

Visualization of classification results for individual candidates
and batch summaries.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from classification.utils import phase_fold, bin_phase_folded
from config import RESULTS_FIGURES_DIR, CLASS_NAMES


def plot_candidate_result(
    time,
    flux,
    tic_id,
    predicted_class,
    confidence,
    probabilities,
    class_names=None,
    period=None,
    epoch=None,
    output_dir=None,
):
    """
    Generate a multi-panel figure for a single classified candidate.

    Panel 1: Detrended light curve
    Panel 2: Phase-folded light curve (if period/epoch available)
    Panel 3: Class probability bar chart

    Parameters
    ----------
    time : ndarray
        Time array.
    flux : ndarray
        Flux array.
    tic_id : int
        TIC ID.
    predicted_class : str
        Predicted class name.
    confidence : float
        Confidence score (0–1).
    probabilities : ndarray
        Probability for each class.
    class_names : list of str, optional
        Class names.
    period : float, optional
        BLS period for phase folding.
    epoch : float, optional
        BLS transit epoch for phase folding.
    output_dir : Path, optional
        Output directory.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if output_dir is None:
        output_dir = RESULTS_FIGURES_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_folded = period is not None and epoch is not None and period > 0
    n_panels = 3 if has_folded else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels))

    fig.suptitle(
        f"TIC {tic_id}  —  {predicted_class}  ({confidence:.1%} confidence)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # --- Panel 1: Detrended Light Curve ---
    ax = axes[0]
    ax.scatter(time, flux, s=0.5, alpha=0.5, color="steelblue", rasterized=True)
    ax.set_xlabel("Time (BTJD)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title("Detrended Light Curve")
    ax.grid(True, alpha=0.2)

    # --- Panel 2 (optional): Phase-Folded Light Curve ---
    if has_folded:
        ax = axes[1]
        phase, folded_flux = phase_fold(time, flux, period, epoch)
        ax.scatter(phase, folded_flux, s=0.5, alpha=0.3, color="gray", rasterized=True)

        # Binned version
        bin_phase, bin_flux = bin_phase_folded(phase, folded_flux, num_bins=200)
        ax.plot(bin_phase, bin_flux, "r-", linewidth=1.5, label="Binned")

        ax.set_xlabel("Phase")
        ax.set_ylabel("Normalized Flux")
        ax.set_title(f"Phase-Folded (P = {period:.4f} days)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    # --- Final Panel: Class Probabilities ---
    ax = axes[-1]
    colors = []
    cmap = plt.cm.Set2
    for i, (cls, prob) in enumerate(zip(class_names, probabilities)):
        c = cmap(i / len(class_names))
        if cls == predicted_class:
            c = "coral"
        colors.append(c)

    bars = ax.barh(class_names, probabilities, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_xlabel("Probability")
    ax.set_title("Predicted Class Probabilities")
    ax.set_xlim(0, 1.05)

    # Annotate bar values
    for bar, prob in zip(bars, probabilities):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{prob:.3f}", va="center", fontsize=9
        )

    ax.grid(True, alpha=0.2, axis="x")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = output_dir / f"TIC_{tic_id}_classification.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_batch_summary(results_df, class_names=None, output_dir=None):
    """
    Generate a summary visualization of the batch classification results.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with 'predicted_class' column.
    class_names : list of str, optional
        Ordered class names.
    output_dir : Path, optional
        Output directory.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if output_dir is None:
        output_dir = Path("results")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: Class Distribution ---
    ax = axes[0]
    counts = results_df["predicted_class"].value_counts()
    # Ensure all classes shown
    all_counts = [counts.get(cls, 0) for cls in class_names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(class_names)))
    bars = ax.bar(class_names, all_counts, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_ylabel("Count")
    ax.set_title("Classification Distribution", fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate counts
    for bar, c in zip(bars, all_counts):
        if c > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(c), ha="center", fontsize=10, fontweight="bold"
            )

    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    # --- Panel 2: Confidence Distribution ---
    ax = axes[1]
    if "confidence_score" in results_df.columns:
        for i, cls in enumerate(class_names):
            mask = results_df["predicted_class"] == cls
            if mask.sum() > 0:
                scores = results_df.loc[mask, "confidence_score"]
                ax.hist(
                    scores, bins=20, alpha=0.6,
                    label=cls, color=colors[i],
                    edgecolor="gray", linewidth=0.5
                )
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Count")
        ax.set_title("Confidence Distribution by Class", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_dir / "classification_summary.png", dpi=150)
    plt.close(fig)
    print(f"  Saved classification_summary.png")
