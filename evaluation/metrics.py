# evaluation/metrics.py
"""
Evaluation utilities shared across all experiments.

Functions
---------
evaluate_sklearn      — accuracy / F1 / classification report for sklearn models
evaluate_wav2vec      — load best checkpoint and run test evaluation
print_final_summary   — formatted summary table
save_summary_csv      — persist results to a CSV file
"""

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from config import EMOTION_NAMES, SAVE_DIR


# ── sklearn / classical models ────────────────────────────────────────────────

def evaluate_sklearn(
    model,
    X_tr, y_tr,
    X_va, y_va,
    X_te, y_te,
    name:      str,
    save_name: str,
    save_dir:  str = SAVE_DIR,
) -> dict:
    """
    Evaluate a fitted sklearn model on all three splits and print a report.

    Returns a dict with keys:
        model, train_acc, val_acc, test_acc, gap,
        test_macro_f1, test_weighted_f1, test_pred
    """
    tr_pred = model.predict(X_tr)
    va_pred = model.predict(X_va)
    te_pred = model.predict(X_te)

    tr_acc = accuracy_score(y_tr, tr_pred)
    va_acc = accuracy_score(y_va, va_pred)
    te_acc = accuracy_score(y_te, te_pred)
    te_f1  = f1_score(y_te, te_pred, average="macro",    zero_division=0)
    te_wf1 = f1_score(y_te, te_pred, average="weighted", zero_division=0)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {name}")
    print(sep)
    print(f"  Train Accuracy : {tr_acc:.2%}")
    print(f"  Val Accuracy   : {va_acc:.2%}")
    print(f"  Test Accuracy  : {te_acc:.2%}")
    print(f"  Train-Val Gap  : {(tr_acc - va_acc):.2%}")
    print(f"\n{classification_report(y_te, te_pred, target_names=EMOTION_NAMES, digits=4, zero_division=0)}")

    return dict(
        model=name,
        train_acc=tr_acc, val_acc=va_acc, test_acc=te_acc,
        gap=tr_acc - va_acc,
        test_macro_f1=te_f1, test_weighted_f1=te_wf1,
        test_pred=te_pred,
    )


# ── Wav2Vec2 checkpoint evaluation ───────────────────────────────────────────

def evaluate_wav2vec_checkpoint(
    model,
    val_loader,
    test_loader,
    run_epoch_fn,
    best_epoch: int,
    ckpt_path:  str,
) -> tuple:
    """
    Load the best validation checkpoint and evaluate on val + test.

    Parameters
    ----------
    model        : Wav2VecEmotionModel (already on device)
    val_loader   : DataLoader
    test_loader  : DataLoader
    run_epoch_fn : callable  — the ``run_epoch`` function from the trainer
    best_epoch   : int
    ckpt_path    : str       — path to the ``.pt`` checkpoint file

    Returns
    -------
    (va_acc, va_f1, yva, pva, te_acc, te_f1, yte, pte)
    """
    from config import device

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    _, va_acc, va_f1, yva, pva = run_epoch_fn(val_loader,  train_mode=False)
    _, te_acc, te_f1, yte, pte = run_epoch_fn(test_loader, train_mode=False)

    sep = "=" * 70
    print(f"\n{sep}")
    print("  Wav2Vec2 Best Checkpoint — Results")
    print(sep)
    print(f"  Best Epoch     : {best_epoch}")
    print(f"  Val Accuracy   : {va_acc:.2%}  |  Val Macro-F1 : {va_f1:.4f}")
    print(f"  Test Accuracy  : {te_acc:.2%}  |  Test Macro-F1: {te_f1:.4f}")

    print("\nClassification Report — Val:")
    print(classification_report(yva, pva, target_names=EMOTION_NAMES, digits=4, zero_division=0))
    print("Classification Report — Test:")
    print(classification_report(yte, pte, target_names=EMOTION_NAMES, digits=4, zero_division=0))

    return va_acc, va_f1, yva, pva, te_acc, te_f1, yte, pte


# ── Final summary helpers ─────────────────────────────────────────────────────

def save_summary_csv(rows: list, filename: str, save_dir: str = SAVE_DIR) -> str:
    """
    Save a list of result dicts to a CSV file.

    Parameters
    ----------
    rows     : list of dicts
    filename : str  (without directory)
    save_dir : str

    Returns
    -------
    Absolute path to the saved CSV.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Saved → {path}")
    return path


def print_final_summary(results: list) -> None:
    """
    Print a formatted summary table.

    Parameters
    ----------
    results : list of dicts
        Each dict must have keys: model, val_acc, test_acc, test_macro_f1.
    """
    sep = "=" * 70
    print(f"\n{sep}")
    print("  FINAL RESULTS SUMMARY")
    print(sep)
    print(f"  {'Model':<35} {'Val Acc':>8} {'Test Acc':>9} {'Macro-F1':>9}")
    print(f"  {'-'*35} {'-'*8} {'-'*9} {'-'*9}")
    for r in results:
        print(
            f"  {r['model']:<35} "
            f"{r['val_acc']:>8.2%} "
            f"{r['test_acc']:>9.2%} "
            f"{r['test_macro_f1']:>9.4f}"
        )
    print(sep)
