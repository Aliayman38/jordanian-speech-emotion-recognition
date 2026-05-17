# models/fusion_v2.py
"""
Fusion v2 — fine-tuned Wav2Vec2 intermediate features + class probabilities.

This is the fusion pipeline from the first notebook.
It uses features extracted from the best fine-tuned ``Wav2VecEmotionModel``
checkpoint (feat256, pooled hidden states, softmax probs) and searches
over two feature sets:

* ``feat256_probs``  — compact 256-dim features concatenated with probs
* ``pooled_probs``   — richer 2H-dim pooled states concatenated with probs

If MFCC features are available a third set is added automatically.

Usage
-----
    from models.fusion_v2 import build_fusion_sets, search_fusion

    fusion_sets = build_fusion_sets(
        f_tr, p_tr, pool_tr,
        f_va, p_va, pool_va,
        f_te, p_te, pool_te,
        X_tr_mfcc_sc=X_train_sc,    # optional
        X_va_mfcc_sc=X_val_sc,
        X_te_mfcc_sc=X_test_sc,
    )

    best_clf, best_info, best_key, history = search_fusion(
        fusion_sets, y_tr, y_va
    )
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import SEED


# ── Feature set builder ───────────────────────────────────────────────────────

def build_fusion_sets(
    f_tr,   p_tr,   pool_tr,
    f_va,   p_va,   pool_va,
    f_te,   p_te,   pool_te,
    X_tr_mfcc_sc=None,
    X_va_mfcc_sc=None,
    X_te_mfcc_sc=None,
) -> dict:
    """
    Build fusion feature matrices from fine-tuned model outputs.

    Parameters
    ----------
    f_tr / f_va / f_te     : 256-dim intermediate features (N, 256)
    p_tr / p_va / p_te     : softmax probabilities (N, num_classes)
    pool_tr / pool_va / pool_te : 2H-dim pooled hidden states
    X_tr_mfcc_sc, ...      : optional standardised MFCC arrays

    Returns
    -------
    dict mapping feature-set name → (Xtr, Xva, Xte)
    """
    fusion_sets = {
        "feat256_probs": (
            np.concatenate([f_tr, p_tr], axis=1),
            np.concatenate([f_va, p_va], axis=1),
            np.concatenate([f_te, p_te], axis=1),
        ),
        "pooled_probs": (
            np.concatenate([pool_tr, p_tr], axis=1),
            np.concatenate([pool_va, p_va], axis=1),
            np.concatenate([pool_te, p_te], axis=1),
        ),
    }

    if X_tr_mfcc_sc is not None:
        fusion_sets["MFCC_feat256_probs"] = (
            np.concatenate([X_tr_mfcc_sc, f_tr, p_tr], axis=1),
            np.concatenate([X_va_mfcc_sc, f_va, p_va], axis=1),
            np.concatenate([X_te_mfcc_sc, f_te, p_te], axis=1),
        )

    print("[INFO] Fusion v2 feature sets:")
    for k, v in fusion_sets.items():
        print(f"  {k}: train={v[0].shape}")
    return fusion_sets


# ── Candidate pipelines ───────────────────────────────────────────────────────

def _build_candidates(feature_name: str) -> list:
    base = dict(random_state=SEED)
    return [
        (f"{feature_name} | LogReg C=0.5",
         Pipeline([("sc", StandardScaler()),
                   ("pca", PCA(n_components=0.95, **base)),
                   ("clf", LogisticRegression(C=0.5, class_weight="balanced",
                                              max_iter=5000, **base))])),
        (f"{feature_name} | LogReg C=1",
         Pipeline([("sc", StandardScaler()),
                   ("pca", PCA(n_components=0.95, **base)),
                   ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                              max_iter=5000, **base))])),
        (f"{feature_name} | SVM C=1",
         Pipeline([("sc", StandardScaler()),
                   ("pca", PCA(n_components=0.95, **base)),
                   ("clf", SVC(kernel="rbf", C=1, gamma="scale",
                               class_weight="balanced", probability=True, **base))])),
        (f"{feature_name} | SVM C=3",
         Pipeline([("sc", StandardScaler()),
                   ("pca", PCA(n_components=0.95, **base)),
                   ("clf", SVC(kernel="rbf", C=3, gamma="scale",
                               class_weight="balanced", probability=True, **base))])),
        (f"{feature_name} | MLP",
         Pipeline([("sc", StandardScaler()),
                   ("pca", PCA(n_components=0.95, **base)),
                   ("clf", MLPClassifier(hidden_layer_sizes=(256, 128),
                                         alpha=1e-2, learning_rate_init=5e-4,
                                         max_iter=600, early_stopping=True,
                                         validation_fraction=0.15,
                                         n_iter_no_change=25, **base))])),
    ]


# ── Per-feature-set search ────────────────────────────────────────────────────

def _search_one(X_tr, X_va, y_tr, y_va, feature_name: str) -> tuple:
    candidates = _build_candidates(feature_name)
    rows = []
    best_clf, best_info = None, None

    print(f"\n{'=' * 80}")
    print(f"  Fusion v2 Search: {feature_name}")
    print(f"{'=' * 80}")

    for i, (name, clf) in enumerate(candidates, 1):
        clf.fit(X_tr, y_tr)

        tr_pred = clf.predict(X_tr)
        va_pred = clf.predict(X_va)

        tr_acc = accuracy_score(y_tr, tr_pred)
        va_acc = accuracy_score(y_va, va_pred)
        tr_f1  = f1_score(y_tr, tr_pred, average="macro", zero_division=0)
        va_f1  = f1_score(y_va, va_pred, average="macro", zero_division=0)
        gap    = tr_acc - va_acc

        # Selection criterion: val F1 + bonus for val acc, penalty for big gap
        score  = va_f1 + 0.25 * va_acc - 0.05 * max(0, gap - 0.35)

        row = dict(feature_set=feature_name, candidate=name,
                   train_acc=tr_acc, val_acc=va_acc,
                   train_f1=tr_f1,   val_f1=va_f1,
                   gap=gap, score=score)
        rows.append(row)

        print(
            f"  {i:02d}/{len(candidates)} | "
            f"Val={va_acc:.2%} F1={va_f1:.4f} | "
            f"Train={tr_acc:.2%} Gap={gap:.2%} | {name}"
        )

        if best_info is None or score > best_info["score"]:
            best_info = row
            best_clf  = clf

    return best_clf, best_info, pd.DataFrame(rows)


# ── Main entry point ──────────────────────────────────────────────────────────

def search_fusion(
    fusion_sets: dict,
    y_tr,
    y_va,
) -> tuple:
    """
    Iterate over all feature sets, run the candidate search for each,
    and return the globally best model.

    Parameters
    ----------
    fusion_sets : dict
        Output of :func:`build_fusion_sets`.
    y_tr, y_va : array-like
        Integer label arrays.

    Returns
    -------
    (best_clf, best_info, best_feature_key, full_history_df)
    """
    all_histories    = []
    best_clf         = None
    best_info        = None
    best_feature_key = None

    for feat_key, (Xtr, Xva, Xte) in fusion_sets.items():
        clf, info, hist = _search_one(Xtr, Xva, y_tr, y_va, feat_key)
        all_histories.append(hist)
        if best_info is None or info["score"] > best_info["score"]:
            best_info        = info
            best_clf         = clf
            best_feature_key = feat_key

    full_history = (
        pd.concat(all_histories, ignore_index=True)
          .sort_values("score", ascending=False)
    )

    print(f"\n✓ Best feature set  : {best_feature_key}")
    print(f"  Best candidate    : {best_info['candidate']}")
    print(f"  Best val acc      : {best_info['val_acc']:.2%}")
    print(f"  Best val macro-F1 : {best_info['val_f1']:.4f}")

    return best_clf, best_info, best_feature_key, full_history
