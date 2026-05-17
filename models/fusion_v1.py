# models/fusion_v1.py
"""
Fusion v1 — MFCC features + frozen Wav2Vec2 embeddings.

This is the classical fusion pipeline from the second notebook.
It concatenates MFCC feature vectors with mean+std-pooled Wav2Vec2
embeddings (from a frozen ``facebook/wav2vec2-base`` backbone), then
runs a compact validation-based sklearn classifier search.

Optionally blends class probabilities from the classical ensemble and
the Wav2Vec2 classifier.

Usage
-----
    from models.fusion_v1 import build_fusion_features, run_fusion_search

    fusion_data = build_fusion_features(
        X_train_mfcc_sc, X_val_mfcc_sc, X_test_mfcc_sc,
        w2v_data,                        # dict from extract_pretrained_embeddings
    )
    best_model, best_info, history = run_fusion_search(
        *fusion_data["MFCC_W2V"], y_train, y_val, "MFCC + Wav2Vec2"
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

def build_fusion_features(
    X_tr_mfcc_sc, X_va_mfcc_sc, X_te_mfcc_sc,
    w2v: dict,
) -> dict:
    """
    Construct all fusion feature sets.

    Parameters
    ----------
    X_tr_mfcc_sc, X_va_mfcc_sc, X_te_mfcc_sc : np.ndarray
        Standardised MFCC feature arrays (fitted on train only).
    w2v : dict
        Output of ``extract_pretrained_embeddings`` — keys include
        ``X_train``, ``X_val``, ``X_test`` (raw embeddings).

    Returns
    -------
    dict mapping feature-set name → (Xtr, Xva, Xte)
    """
    # Standardise Wav2Vec2 embeddings on train only
    from sklearn.preprocessing import StandardScaler
    w2v_scaler  = StandardScaler()
    X_tr_w2v_sc = w2v_scaler.fit_transform(w2v["X_train"])
    X_va_w2v_sc = w2v_scaler.transform(w2v["X_val"])
    X_te_w2v_sc = w2v_scaler.transform(w2v["X_test"])

    fusion_sets = {
        # Wav2Vec2 embeddings only
        "Wav2Vec2_only": (X_tr_w2v_sc, X_va_w2v_sc, X_te_w2v_sc),
        # MFCC + Wav2Vec2
        "MFCC_W2V": (
            np.concatenate([X_tr_mfcc_sc, X_tr_w2v_sc], axis=1),
            np.concatenate([X_va_mfcc_sc, X_va_w2v_sc], axis=1),
            np.concatenate([X_te_mfcc_sc, X_te_w2v_sc], axis=1),
        ),
    }
    print("[INFO] Fusion v1 feature sets:")
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
                                              max_iter=4000, **base))])),
        (f"{feature_name} | LogReg C=1",
         Pipeline([("sc", StandardScaler()),
                   ("pca", PCA(n_components=0.95, **base)),
                   ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                              max_iter=4000, **base))])),
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
                                         max_iter=500, early_stopping=True,
                                         validation_fraction=0.15,
                                         n_iter_no_change=20, **base))])),
    ]


# ── Validation-based search ───────────────────────────────────────────────────

def run_fusion_search(
    X_tr, X_va, X_te,
    y_tr, y_va,
    feature_name: str,
) -> tuple:
    """
    Train all candidates and pick the best by validation score.

    Returns
    -------
    (best_clf, best_info_dict, history_df)
    """
    candidates = _build_candidates(feature_name)
    rows = []
    best_clf, best_info = None, None

    print(f"\n{'=' * 80}")
    print(f"  Fusion v1 Search: {feature_name}")
    print(f"{'=' * 80}")

    for i, (name, clf) in enumerate(candidates, 1):
        clf.fit(X_tr, y_tr)
        tr_acc = accuracy_score(y_tr, clf.predict(X_tr))
        va_acc = accuracy_score(y_va, clf.predict(X_va))
        tr_f1  = f1_score(y_tr, clf.predict(X_tr), average="macro", zero_division=0)
        va_f1  = f1_score(y_va, clf.predict(X_va), average="macro", zero_division=0)
        gap    = tr_acc - va_acc
        score  = va_acc - 0.10 * max(0, gap - 0.30)

        row = dict(feature_set=feature_name, candidate=name,
                   train_acc=tr_acc, val_acc=va_acc,
                   train_f1=tr_f1,   val_f1=va_f1,
                   gap=gap, score=score)
        rows.append(row)

        print(
            f"  {i:02d}/{len(candidates)} | Val={va_acc:.2%} | "
            f"Train={tr_acc:.2%} | Gap={gap:.2%} | {name}"
        )

        if best_info is None or score > best_info["score"]:
            best_info = row
            best_clf  = clf

    return best_clf, best_info, pd.DataFrame(rows).sort_values("score", ascending=False)


# ── Optional probability-level ensemble ───────────────────────────────────────

def probability_blend_search(
    p_val_w2v,   p_test_w2v,   y_val,   y_test,
    p_val_clf,   p_test_clf,
) -> tuple:
    """
    Grid-search over blending weights ``w`` ∈ [0, 1] (step 0.05) where
    the blended prediction is ``w * Wav2Vec2 + (1-w) * classical``.
    Model selection is strictly on validation — test is never touched.

    Returns
    -------
    (best_info_dict, best_test_predictions)
    """
    best_score, best_info, best_pred = -1.0, None, None

    for w in np.arange(0.0, 1.05, 0.05):
        p_va_mix = w * p_val_w2v  + (1 - w) * p_val_clf
        p_te_mix = w * p_test_w2v + (1 - w) * p_test_clf

        va_pred  = np.argmax(p_va_mix, axis=1)
        te_pred  = np.argmax(p_te_mix, axis=1)

        va_acc = accuracy_score(y_val,  va_pred)
        va_f1  = f1_score(y_val,  va_pred, average="macro", zero_division=0)
        te_acc = accuracy_score(y_test, te_pred)
        score  = va_f1 + 0.25 * va_acc

        if score > best_score:
            best_score = score
            best_info  = dict(weight_w2v=w, weight_clf=round(1 - w, 2),
                              val_acc=va_acc, val_f1=va_f1, test_acc=te_acc)
            best_pred  = te_pred

    return best_info, best_pred
