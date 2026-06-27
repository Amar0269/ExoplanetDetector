"""
evaluate.py

Model evaluation — metrics computation and visualization.

Generates:
- metrics.json (accuracy, precision, recall, F1, ROC-AUC)
- loss_curve.png
- accuracy_curve.png
- confusion_matrix.png
- roc_curve.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from config import RESULTS_DIR, MODEL_DIR, CLASS_NAMES


def compute_metrics(y_true, y_pred, y_prob, class_names=None):
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_true : ndarray
        True integer labels.
    y_pred : ndarray
        Predicted integer labels.
    y_prob : ndarray
        Predicted probabilities, shape (N, num_classes).
    class_names : list of str, optional
        Class name for each index.

    Returns
    -------
    dict
        Dictionary of all metrics.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    num_classes = len(class_names)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Per-class metrics
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics["per_class"] = {}
    for i, cls in enumerate(class_names):
        if i < len(per_class_precision):
            metrics["per_class"][cls] = {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
            }

    # ROC-AUC (one-vs-rest)
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        roc_auc_scores = {}
        for i, cls in enumerate(class_names):
            if y_true_bin[:, i].sum() > 0:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc_scores[cls] = float(auc(fpr, tpr))
        metrics["roc_auc_per_class"] = roc_auc_scores
        if roc_auc_scores:
            metrics["roc_auc_macro"] = float(np.mean(list(roc_auc_scores.values())))
    except Exception:
        metrics["roc_auc_per_class"] = {}
        metrics["roc_auc_macro"] = None

    return metrics


def save_metrics(metrics_dict, output_path=None):
    """
    Save metrics to JSON file.

    Parameters
    ----------
    metrics_dict : dict
        Metrics to save.
    output_path : str or Path, optional
        Output file path. Defaults to model/metrics.json.
    """
    if output_path is None:
        output_path = MODEL_DIR / "metrics.json"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"  Saved metrics → {output_path}")


def plot_training_curves(history, output_dir=None):
    """
    Plot training and validation loss/accuracy curves.

    Parameters
    ----------
    history : dict
        Training history with keys: 'train_loss', 'val_loss',
        'train_acc', 'val_acc'.
    output_dir : Path or str, optional
        Output directory for plots.
    """
    if output_dir is None:
        output_dir = RESULTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # --- Loss Curve ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], "b-", linewidth=2, label="Training Loss")
    ax.plot(epochs, history["val_loss"], "r-", linewidth=2, label="Validation Loss")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved loss_curve.png")

    # --- Accuracy Curve ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_acc"], "b-", linewidth=2, label="Training Accuracy")
    ax.plot(epochs, history["val_acc"], "r-", linewidth=2, label="Validation Accuracy")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_curve.png", dpi=150)
    plt.close(fig)
    print(f"  Saved accuracy_curve.png")


def plot_confusion_matrix(y_true, y_pred, class_names=None, output_path=None):
    """
    Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    y_true : ndarray
        True labels.
    y_pred : ndarray
        Predicted labels.
    class_names : list of str, optional
        Class names.
    output_path : str or Path, optional
        Output file path.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if output_path is None:
        output_path = RESULTS_DIR / "confusion_matrix.png"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion_matrix.png")


def plot_roc_curves(y_true, y_prob, class_names=None, output_path=None):
    """
    Plot one-vs-rest ROC curves with AUC values.

    Parameters
    ----------
    y_true : ndarray
        True integer labels.
    y_prob : ndarray
        Predicted probabilities, shape (N, num_classes).
    class_names : list of str, optional
        Class names.
    output_path : str or Path, optional
        Output file path.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if output_path is None:
        output_path = RESULTS_DIR / "roc_curve.png"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set2(np.linspace(0, 1, num_classes))

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        if y_true_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            color=color,
            linewidth=2,
            label=f"{cls} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved roc_curve.png")


def print_classification_report(y_true, y_pred, class_names=None):
    """Print a formatted classification report."""
    if class_names is None:
        class_names = CLASS_NAMES
    print("\n" + classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    ))
