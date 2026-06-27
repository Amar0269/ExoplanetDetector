"""
train.py

Train the exoplanet signal classifier.

Usage:
    python train.py

Automatically:
1. Loads all detrended light curves from data/detrended/
2. Acquires labels (TOI catalog + heuristic)
3. Augments dataset to balance classes
4. Builds Conv1D + LSTM model
5. Trains with early stopping and learning rate scheduling
6. Evaluates on held-out test set
7. Saves best model, training history, metrics, and plots

Output:
    model/best_model.pth
    model/training_history.pkl
    model/metrics.json
    results/loss_curve.png
    results/accuracy_curve.png
    results/confusion_matrix.png
    results/roc_curve.png
"""

import pickle
import time as time_module
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from classification.utils import set_seed
from classification.dataset import prepare_dataset, split_dataset, get_class_names
from classification.model import build_model, get_model_summary, count_parameters
from classification.evaluate import (
    compute_metrics,
    save_metrics,
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curves,
    print_classification_report,
)
from config import (
    MODEL_DIR,
    RESULTS_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    LR_REDUCE_PATIENCE,
    LR_REDUCE_FACTOR,
    TARGET_SEQUENCE_LENGTH,
    NUM_CLASSES,
    RANDOM_SEED,
)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate and return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += X_batch.size(0)

    return total_loss / total, correct / total


def get_predictions(model, dataloader, device):
    """Get predictions and probabilities for the full dataset."""
    model.eval()
    all_preds = []
    all_probs = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_true.append(y_batch.numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("  EXOPLANET SIGNAL CLASSIFIER — TRAINING")
    print("=" * 60)

    # --- Setup ---
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Prepare Dataset ---
    X, y, label_encoder, filenames = prepare_dataset()
    splits = split_dataset(X, y)

    class_names = get_class_names()

    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(splits["X_train"]),
        torch.LongTensor(splits["y_train"]),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(splits["X_val"]),
        torch.LongTensor(splits["y_val"]),
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(splits["X_test"]),
        torch.LongTensor(splits["y_test"]),
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 2. Build Model ---
    model = build_model(TARGET_SEQUENCE_LENGTH, NUM_CLASSES)
    model = model.to(device)
    print(get_model_summary(model))
    print(f"\n  Trainable parameters: {count_parameters(model):,}")

    # --- 3. Training Setup ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE,
    )

    # --- 4. Training Loop ---
    print(f"\n{'='*60}")
    print(f"  TRAINING — {MAX_EPOCHS} max epochs, patience={EARLY_STOPPING_PATIENCE}")
    print(f"{'='*60}\n")

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    start_time = time_module.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = time_module.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epoch_time = time_module.time() - epoch_start

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            marker = " ★ best"
        else:
            patience_counter += 1
            marker = ""

        print(
            f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"{epoch_time:.1f}s{marker}"
        )

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} (patience={EARLY_STOPPING_PATIENCE})")
            break

    total_time = time_module.time() - start_time
    print(f"\n  Training completed in {total_time:.1f}s ({epoch} epochs)")

    # --- 5. Save Best Model ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model_path = MODEL_DIR / "best_model.pth"
    torch.save({
        "model_state_dict": best_model_state or model.state_dict(),
        "input_length": TARGET_SEQUENCE_LENGTH,
        "num_classes": NUM_CLASSES,
        "class_names": class_names,
    }, model_path)
    print(f"  Saved model → {model_path}")

    # Also save as .keras-compatible name for documentation
    # (The actual format is PyTorch .pth)

    # --- 6. Save Training History ---
    history_path = MODEL_DIR / "training_history.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    print(f"  Saved training history → {history_path}")

    # --- 7. Evaluate on Test Set ---
    print(f"\n{'='*60}")
    print(f"  EVALUATION ON TEST SET")
    print(f"{'='*60}")

    y_true, y_pred, y_prob = get_predictions(model, test_loader, device)

    # Print classification report
    print_classification_report(y_true, y_pred, class_names)

    # Compute all metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    metrics["training_epochs"] = epoch
    metrics["training_time_seconds"] = total_time
    metrics["total_parameters"] = count_parameters(model)

    print(f"\n  Test Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Macro F1:         {metrics['f1_macro']:.4f}")
    print(f"  Weighted F1:      {metrics['f1_weighted']:.4f}")
    if metrics.get("roc_auc_macro"):
        print(f"  Macro ROC-AUC:    {metrics['roc_auc_macro']:.4f}")

    # Save metrics
    save_metrics(metrics)

    # --- 8. Generate Plots ---
    print(f"\n  Generating plots...")
    plot_training_curves(history)
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_roc_curves(y_true, y_prob, class_names)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Model:   {model_path}")
    print(f"  Metrics: {MODEL_DIR / 'metrics.json'}")
    print(f"  Plots:   {RESULTS_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
