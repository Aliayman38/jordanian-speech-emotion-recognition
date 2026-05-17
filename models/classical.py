# models/classical.py
"""
Classical sklearn models for MFCC-based SER.

Includes
--------
* ``build_svm_candidates``   — SVM pipeline grid
* ``build_knn_candidates``   — KNN pipeline grid
* ``build_mlp_candidates``   — MLPClassifier pipeline grid
* ``build_extra_candidates`` — LogReg + ExtraTrees baselines
* ``quick_search``           — validation-based model selection
* ``build_soft_ensemble``    — validation-weighted soft-voting ensemble
* ``MLP``                    — custom PyTorch MLP (used for deep training)
* ``train_pytorch_mlp``      — training loop for the PyTorch MLP
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from config import CLF_BATCH_SIZE, CLF_PATIENCE, LR, MLP_EPOCHS, NUM_CLASSES, SEED, device


# ── Pipeline builder helpers ──────────────────────────────────────────────────

def _make_steps(use_scaler: bool, reducer_type=None, reducer_value=None) -> list:
    """Build the pre-processing steps for a sklearn Pipeline."""
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    if reducer_type == "pca":
        steps.append(("reduce", PCA(n_components=reducer_value, random_state=SEED)))
    elif reducer_type == "kbest":
        steps.append(("reduce", SelectKBest(score_func=f_classif, k=reducer_value)))
    return steps


# ── Candidate grids ───────────────────────────────────────────────────────────

def build_svm_candidates(n_features: int, use_scaler: bool = True) -> list:
    candidates = []
    for reducer_type, reducer_value in [
        (None, None), ("pca", 0.90), ("pca", 0.95),
        ("kbest", 80), ("kbest", 120), ("kbest", 160),
    ]:
        for C, gamma in [
            (0.1, "scale"), (0.5, "scale"), (1, "scale"),
            (2,   "scale"), (5,   "scale"),
            (1, 0.01),      (1, 0.005),     (2, 0.005),
        ]:
            k = min(reducer_value, n_features) if isinstance(reducer_value, int) else reducer_value
            steps = _make_steps(use_scaler, reducer_type, k)
            steps.append(("clf", SVC(
                kernel="rbf", C=C, gamma=gamma,
                class_weight="balanced", probability=True, random_state=SEED,
            )))
            candidates.append({
                "name":  f"SVM | reduce={reducer_type}:{reducer_value} | C={C} | gamma={gamma}",
                "model": Pipeline(steps),
            })
    return candidates[:36]          # keep grid compact


def build_knn_candidates(n_features: int, use_scaler: bool = True) -> list:
    candidates = []
    for reducer_type, reducer_value in [
        ("pca", 0.90), ("pca", 0.95),
        ("kbest", 80), ("kbest", 120), ("kbest", 160),
    ]:
        for n, metric, weights in [
            (3, "manhattan", "distance"), (5, "manhattan", "distance"),
            (7, "manhattan", "distance"), (9, "manhattan", "distance"),
            (5, "cosine",    "distance"), (7, "cosine",    "distance"),
        ]:
            k = min(reducer_value, n_features) if isinstance(reducer_value, int) else reducer_value
            steps = _make_steps(use_scaler, reducer_type, k)
            steps.append(("clf", KNeighborsClassifier(
                n_neighbors=n, metric=metric, weights=weights,
            )))
            candidates.append({
                "name":  f"KNN | reduce={reducer_type}:{reducer_value} | n={n} | metric={metric}",
                "model": Pipeline(steps),
            })
    return candidates[:24]


def build_mlp_candidates(n_features: int, use_scaler: bool = True) -> list:
    candidates = []
    for reducer_type, reducer_value in [
        ("pca", 0.90), ("pca", 0.95), ("kbest", 120), ("kbest", 160),
    ]:
        for hidden, alpha, lr in [
            ((64,),       1e-2, 1e-3),
            ((128,),      1e-2, 1e-3),
            ((256,),      1e-3, 1e-3),
            ((128, 64),   1e-2, 5e-4),
            ((256, 128),  1e-2, 5e-4),
        ]:
            k = min(reducer_value, n_features) if isinstance(reducer_value, int) else reducer_value
            steps = _make_steps(use_scaler, reducer_type, k)
            steps.append(("clf", MLPClassifier(
                hidden_layer_sizes=hidden, alpha=alpha,
                learning_rate_init=lr, activation="relu", solver="adam",
                batch_size=64, max_iter=400,
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=SEED,
            )))
            candidates.append({
                "name":  f"MLP | reduce={reducer_type}:{reducer_value} | hidden={hidden} | alpha={alpha}",
                "model": Pipeline(steps),
            })
    return candidates[:20]


def build_extra_candidates(n_features: int, use_scaler: bool = True) -> list:
    candidates = []
    k = min(160, n_features)
    for C in [0.1, 0.5, 1, 2]:
        steps = _make_steps(use_scaler, "kbest", k)
        steps.append(("clf", LogisticRegression(
            C=C, class_weight="balanced", solver="lbfgs",
            max_iter=3000, random_state=SEED,
        )))
        candidates.append({
            "name":  f"LogReg | SelectKBest={k} | C={C}",
            "model": Pipeline(steps),
        })
    for n_est, leaf in [(600, 2), (800, 4)]:
        candidates.append({
            "name":  f"ExtraTrees | {n_est} trees | leaf={leaf}",
            "model": ExtraTreesClassifier(
                n_estimators=n_est, min_samples_leaf=leaf,
                class_weight="balanced", random_state=SEED, n_jobs=-1,
            ),
        })
    return candidates


# ── Validation-based model selection ─────────────────────────────────────────

def quick_search(
    model_name: str,
    candidates: list,
    X_tr, y_tr,
    X_vl, y_vl,
    overfitting_penalty: float = 0.10,
    gap_threshold:       float = 0.30,
) -> tuple:
    """
    Train every candidate, score by validation accuracy (with a light
    penalty for large train-val gaps), and return the best one.

    Returns
    -------
    (best_model, best_info_dict, history_df)
    """
    print(f"\n{'#' * 70}")
    print(f"  Searching {model_name}  ({len(candidates)} candidates)")
    print(f"{'#' * 70}")

    best_model, best_info = None, None
    rows = []

    for i, item in enumerate(candidates, 1):
        name  = item["name"]
        model = item["model"]
        model.fit(X_tr, y_tr)

        tr_acc = accuracy_score(y_tr, model.predict(X_tr))
        va_acc = accuracy_score(y_vl, model.predict(X_vl))
        gap    = tr_acc - va_acc
        score  = va_acc - overfitting_penalty * max(0, gap - gap_threshold)

        row = dict(candidate=name, train_acc=tr_acc, val_acc=va_acc,
                   gap=gap, score=score)
        rows.append(row)

        print(
            f"  {i:02d}/{len(candidates)} | Val={va_acc:.2%} | "
            f"Train={tr_acc:.2%} | Gap={gap:.2%} | Score={score:.4f} | {name}"
        )

        if best_info is None or score > best_info["score"]:
            best_model = model
            best_info  = row

    print("\n  [BEST]", best_info)
    return best_model, best_info, pd.DataFrame(rows).sort_values("score", ascending=False)


# ── Soft-voting ensemble ──────────────────────────────────────────────────────

def build_soft_ensemble(named_results: dict, X_tr, y_tr):
    """
    Fit a ``VotingClassifier`` (soft) using the best per-model classifiers,
    weighted by their validation accuracy.

    Parameters
    ----------
    named_results : dict
        ``{ name: (fitted_model, val_acc) }``
    X_tr, y_tr : training features / labels

    Returns
    -------
    fitted VotingClassifier
    """
    estimators = []
    weights    = []
    for name, (model, val_acc) in named_results.items():
        estimators.append((name, model))
        weights.append(max(1.0, val_acc * 10))

    vc = VotingClassifier(estimators=estimators, voting="soft", weights=weights)
    vc.fit(X_tr, y_tr)
    return vc


# ── PyTorch MLP (deep variant) ────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Simple fully-connected network for tabular feature vectors.

    Parameters
    ----------
    input_dim : int
    hidden_dims : tuple of int
    num_classes : int
    dropout : float
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dims: tuple = (256, 128, 64),
        num_classes: int   = NUM_CLASSES,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_pytorch_mlp(
    X_tr, y_tr,
    X_vl, y_vl,
    epochs:   int   = MLP_EPOCHS,
    patience: int   = CLF_PATIENCE,
    lr:       float = LR,
    batch:    int   = CLF_BATCH_SIZE,
) -> tuple:
    """
    Train the PyTorch :class:`MLP` with Adam, step-LR decay, and early stopping.

    Returns
    -------
    (model, history_dict, best_epoch)
    """
    Xt = torch.FloatTensor(X_tr)
    yt = torch.LongTensor(y_tr)
    Xv = torch.FloatTensor(X_vl)
    yv = torch.LongTensor(y_vl)

    loader = DataLoader(
        torch.utils.data.TensorDataset(Xt, yt),
        batch_size=batch, shuffle=True, drop_last=True,
    )

    model  = MLP(Xt.shape[1]).to(device)
    opt    = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    sched  = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
    crit   = nn.CrossEntropyLoss()
    hist   = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    best_val, patience_cnt, best_state, best_ep = 0.0, 0, None, 1

    print(f"{'Ep':>5} {'TrLoss':>8} {'TrAcc':>8} {'ValLoss':>8} {'ValAcc':>8}")
    print("-" * 45)

    for ep in range(epochs):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out  = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(yb)
            tc += (out.argmax(1) == yb).sum().item()
            tn += len(yb)
        sched.step()
        tl /= tn;  ta = tc / tn

        model.eval()
        with torch.no_grad():
            vo = model(Xv.to(device))
            vl = crit(vo, yv.to(device)).item()
            va = (vo.argmax(1).cpu() == yv).float().mean().item()

        hist["train_loss"].append(tl); hist["train_acc"].append(ta)
        hist["val_loss"].append(vl);   hist["val_acc"].append(va)

        if va > best_val:
            best_val, patience_cnt = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ep = ep + 1
        else:
            patience_cnt += 1

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"{ep+1:5d} {tl:8.4f} {ta:8.2%} {vl:8.4f} {va:8.2%}")

        if patience_cnt >= patience:
            print(f"  Early stopping at epoch {ep + 1} (best={best_ep})")
            break

    model.load_state_dict(best_state)
    return model, hist, best_ep
