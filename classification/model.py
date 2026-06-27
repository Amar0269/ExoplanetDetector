"""
model.py

Conv1D + LSTM deep learning model for exoplanet signal classification.

Architecture:
    Conv1D(64) → BatchNorm → ReLU → MaxPool(2)
    Conv1D(128) → BatchNorm → ReLU → MaxPool(2)
    Conv1D(256) → BatchNorm → ReLU → MaxPool(2)
    Bidirectional LSTM(128)
    Dropout(0.4)
    Dense(128) → ReLU → Dropout(0.3)
    Dense(5) → Softmax
"""

import torch
import torch.nn as nn


class ExoplanetClassifier(nn.Module):
    """
    Conv1D + Bidirectional LSTM classifier for astrophysical signal
    classification from TESS light curves.

    Parameters
    ----------
    input_length : int
        Length of the input sequence (default: 2048).
    num_classes : int
        Number of output classes (default: 5).
    """

    def __init__(self, input_length=2048, num_classes=5):
        super().__init__()

        self.input_length = input_length
        self.num_classes = num_classes

        # --- Convolutional Feature Extractor ---

        # Block 1: Conv1D(64, k=7) → BN → ReLU → MaxPool(2)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Block 2: Conv1D(128, k=5) → BN → ReLU → MaxPool(2)
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # Block 3: Conv1D(256, k=3) → BN → ReLU → MaxPool(2)
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # --- Temporal Modeling ---
        # After 3x MaxPool(2): seq_len = input_length / 8
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # --- Classifier Head ---
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(128 * 2, 128)  # *2 for bidirectional
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, 1).

        Returns
        -------
        torch.Tensor
            Shape (batch, num_classes) — log probabilities (after log_softmax)
            or raw logits depending on use.
        """
        # Conv1D expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        # Feature extraction
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # LSTM expects (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        # LSTM — take the final hidden state
        lstm_out, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # Concatenate forward and backward final hidden states
        h_forward = h_n[0]
        h_backward = h_n[1]
        h_combined = torch.cat([h_forward, h_backward], dim=1)

        # Classifier
        out = self.dropout1(h_combined)
        out = self.fc1(out)
        out = self.relu_fc(out)
        out = self.dropout2(out)
        out = self.fc2(out)

        return out  # Raw logits — CrossEntropyLoss applies softmax internally


def build_model(input_length=2048, num_classes=5):
    """
    Build and return the ExoplanetClassifier model.

    Parameters
    ----------
    input_length : int
        Sequence length.
    num_classes : int
        Number of output classes.

    Returns
    -------
    ExoplanetClassifier
        The constructed model.
    """
    model = ExoplanetClassifier(input_length, num_classes)
    return model


def get_model_summary(model):
    """
    Return a string summary of the model architecture.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model.

    Returns
    -------
    str
        Human-readable model summary.
    """
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  ExoplanetClassifier — Model Summary")
    lines.append(f"{'='*60}")

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    lines.append(f"\n  Architecture:")
    for name, module in model.named_modules():
        if name:
            lines.append(f"    {name}: {module.__class__.__name__}")

    lines.append(f"\n  Total parameters:     {total_params:,}")
    lines.append(f"  Trainable parameters: {trainable_params:,}")
    lines.append(f"{'='*60}")

    return "\n".join(lines)


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
