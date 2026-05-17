# models/cnn_spec.py
"""
Lightweight 2-D CNN for Mel-spectrogram-based emotion recognition.

Architecture
------------
* Block 1 : Conv2d(1→32) × 2  + BN + ReLU + MaxPool + Dropout2d
* Block 2 : Conv2d(32→64) × 2 + BN + ReLU + MaxPool + Dropout2d
* Block 3 : Conv2d(64→128)    + BN + ReLU + AdaptiveAvgPool(4×4)
* Head    : Flatten → Linear(2048→256) → ReLU → Dropout → Linear(256→C)

Input shape : (B, 1, N_MELS, MAX_FRAMES)

Usage
-----
    from models.cnn_spec import EmotionCNN, run_epoch

    cnn  = EmotionCNN().to(device)
    loss = nn.CrossEntropyLoss()
    opt  = torch.optim.AdamW(cnn.parameters(), lr=1e-3)

    train_loss, train_acc = run_epoch(cnn, train_loader, loss, opt)
    val_loss,   val_acc   = run_epoch(cnn, val_loader,   loss)
"""

import torch
import torch.nn as nn

from config import NUM_CLASSES


class EmotionCNN(nn.Module):
    """
    Lightweight 2-D CNN for Mel-spectrogram input.

    Parameters
    ----------
    num_classes : int
        Number of emotion categories.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # ── Block 1 ──────────────────────────────────────────────────────
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # ── Block 2 ──────────────────────────────────────────────────────
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # ── Block 3 ──────────────────────────────────────────────────────
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer=None,
    device=None,
) -> tuple:
    """
    One forward pass (and optional backward pass) over *loader*.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
        Yields ``(mel_tensor, label_tensor)`` batches.
    criterion : nn.Module
        Loss function.
    optimizer : torch.optim.Optimizer or None
        If *None*, run in eval mode (no gradient computation).
    device : torch.device or None
        Defaults to CUDA if available.

    Returns
    -------
    (avg_loss, accuracy) : (float, float)
    """
    if device is None:
        from config import device as _device
        device = _device

    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, n = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out  = model(xb)
            loss = criterion(out, yb)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(yb)
            correct    += (out.argmax(1) == yb).sum().item()
            n          += len(yb)

    return total_loss / n, correct / n
