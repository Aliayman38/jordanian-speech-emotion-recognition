# evaluation/plots.py
"""
Plotting utilities for the SER project.

Functions
---------
plot_confusion_matrix  — heatmap confusion matrix (saved as PNG)
plot_training_curves   — loss and accuracy curves (saved as PNG)
plot_model_comparison  — bar chart comparing multiple models (saved as PNG)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import EMOTION_NAMES


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true,
    y_pred,
    title:     str,
    save_path: str,
    figsize:   tuple = (6, 5),
    dpi:       int   = 200,
) -> None:
    """
    Plot and save a heatmap confusion matrix.

    Parameters
    ----------
    y_true, y_pred : array-like  — ground-truth and predicted labels
    title     : str              — plot title
    save_path : str              — full path for the output PNG
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    history_df: pd.DataFrame,
    best_epoch: int,
    save_path:  str,
    title:      str = "Training Curves",
    figsize:    tuple = (12, 4),
    dpi:        int   = 200,
) -> None:
    """
    Plot loss and accuracy curves with a vertical line at *best_epoch*.

    Parameters
    ----------
    history_df : pd.DataFrame
        Must have columns: epoch, train_loss, val_loss, train_acc, val_acc.
    best_epoch : int
    save_path  : str
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ep = history_df["epoch"]

    # Loss
    ax1.plot(ep, history_df["train_loss"], label="Train Loss")
    ax1.plot(ep, history_df["val_loss"],   label="Val Loss")
    ax1.axvline(best_epoch, color="red", linestyle="--", alpha=0.5,
                label=f"Best epoch {best_epoch}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} — Loss", fontweight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(ep, history_df["train_acc"], label="Train Acc")
    ax2.plot(ep, history_df["val_acc"],   label="Val Acc")
    ax2.axvline(best_epoch, color="red", linestyle="--", alpha=0.5,
                label=f"Best epoch {best_epoch}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title} — Accuracy", fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")


# ── Model comparison bar chart ────────────────────────────────────────────────

def plot_model_comparison(
    model_names:  list,
    test_accs:    list,
    save_path:    str,
    target_line:  float = 0.70,
    baseline_acc: float = 0.6736,
    title:        str   = "Model Comparison — Test Accuracy",
    figsize:      tuple = (8, 5),
    dpi:          int   = 200,
) -> None:
    """
    Bar chart comparing test accuracies across models.

    Parameters
    ----------
    model_names  : list of str
    test_accs    : list of float (0–1 range)
    save_path    : str
    target_line  : float — horizontal dashed line (e.g. 70% target)
    baseline_acc : float — first bar is always the baseline
    """
    colors = ["#aab4c8"] + ["#4c8cdb"] * (len(model_names) - 2) + ["#2ecc71"]
    colors = colors[: len(model_names)]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        model_names,
        [a * 100 for a in test_accs],
        color=colors, edgecolor="white", width=0.5,
    )
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.axhline(target_line * 100, color="orange", linestyle="--",
               alpha=0.7, label=f"{target_line:.0%} target")
    for bar, acc in zip(bars, test_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{acc:.2%}", ha="center", va="bottom", fontweight="bold",
        )
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {save_path}")
