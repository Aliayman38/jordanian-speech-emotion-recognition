"""
Speech Emotion Recognition — Final Classical ML Pipeline (Tuned LOSO)
Jordanian Arabic Dialect | MFCC Handcrafted Features

Tuning protocol:
- Hyperparameters were selected using speaker-based cluster split:
  Train speakers = 23
  Validation speakers = 5
  Test speakers = 5

Final evaluation:
- Leave-One-Speaker-Out (LOSO) over all speakers
- StandardScaler is fitted inside each fold only
- No data leakage

Run:
python models/classical.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
FEATURES_CSV = ROOT / "outputs" / "features" / "mfcc_features_loso.csv"
RESULTS_DIR = ROOT / "outputs" / "results" / "classical_mfcc_loso_tuned"

RANDOM_STATE = 42
META_COLS = {"label", "speaker_id", "gender", "rel_path"}
EMOTION_NAMES = ["Happy", "Sad", "Angry", "Neutral"]


TUNED_PARAMS = {
    "svm": {
        "C": 5,
        "gamma": "scale",
    },
    "knn": {
        "metric": "manhattan",
        "n_neighbors": 5,
        "weights": "distance",
    },
    "mlp": {
        "hidden_layer_sizes": (256,),
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
    },
}


def get_logger() -> logging.Logger:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_DIR / "run.log", encoding="utf-8"),
        ],
    )

    return logging.getLogger("SER_TUNED_LOSO")


def load_data(log: logging.Logger):
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Feature file not found: {FEATURES_CSV}")

    df = pd.read_csv(FEATURES_CSV).dropna()
    feature_cols = [c for c in df.columns if c not in META_COLS]

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)
    speakers = df["speaker_id"].to_numpy()

    log.info(f"Dataset  : {X.shape[0]:,} samples x {X.shape[1]} features")
    log.info(f"Speakers : {np.unique(speakers).tolist()}")
    log.info(f"Classes  : { {i: int((y == i).sum()) for i in range(4)} }")

    return X, y, speakers


def build_models():
    return {
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=TUNED_PARAMS["svm"]["C"],
                gamma=TUNED_PARAMS["svm"]["gamma"],
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),

        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=TUNED_PARAMS["knn"]["n_neighbors"],
                weights=TUNED_PARAMS["knn"]["weights"],
                metric=TUNED_PARAMS["knn"]["metric"],
                n_jobs=-1,
            )),
        ]),

        "mlp": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=TUNED_PARAMS["mlp"]["hidden_layer_sizes"],
                alpha=TUNED_PARAMS["mlp"]["alpha"],
                learning_rate_init=TUNED_PARAMS["mlp"]["learning_rate_init"],
                activation="relu",
                solver="adam",
                batch_size=64,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def run_loso(name, model, X, y, speakers, log):
    unique_speakers = np.unique(speakers)

    fold_rows = []
    all_true = []
    all_pred = []

    log.info("\n" + "─" * 70)
    log.info(f"{name.upper()} | Tuned LOSO | {len(unique_speakers)} folds")
    log.info("─" * 70)

    for fold, test_speaker in enumerate(unique_speakers, start=1):
        train_mask = speakers != test_speaker
        test_mask = speakers == test_speaker

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - start

        acc = accuracy_score(y_test, y_pred)

        fold_rows.append({
            "fold": fold,
            "test_speaker": int(test_speaker),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "accuracy": round(float(acc), 4),
            "time_sec": round(elapsed, 2),
        })

        all_true.extend(y_test)
        all_pred.extend(y_pred)

        log.info(
            f"Fold {fold:02d}/{len(unique_speakers)} | "
            f"Speaker {test_speaker:<3} | "
            f"n_train={len(y_train):>4} | "
            f"n_test={len(y_test):>3} | "
            f"Acc={acc:.4f} | "
            f"{elapsed:.1f}s"
        )

    fold_accs = [r["accuracy"] for r in fold_rows]

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    overall_acc = accuracy_score(all_true, all_pred)

    report = classification_report(
        all_true,
        all_pred,
        target_names=EMOTION_NAMES,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(all_true, all_pred)

    log.info(f"\n{name.upper()} RESULTS")
    log.info(f"Mean LOSO Accuracy    : {mean_acc:.4f} +/- {std_acc:.4f}")
    log.info(f"Overall Sample Accuracy: {overall_acc:.4f}")

    for line in report.splitlines():
        log.info("  " + line)

    return {
        "model": name,
        "tuned_params": TUNED_PARAMS[name],
        "mean_loso_accuracy": round(mean_acc, 4),
        "std_loso_accuracy": round(std_acc, 4),
        "overall_sample_accuracy": round(float(overall_acc), 4),
        "best_fold": max(fold_accs),
        "worst_fold": min(fold_accs),
        "n_folds": len(fold_rows),
        "fold_results": fold_rows,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def save_model_results(result, log):
    model_dir = RESULTS_DIR / result["model"]
    model_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(result["fold_results"]).to_csv(
        model_dir / "fold_results.csv",
        index=False,
    )

    pd.DataFrame(
        result["confusion_matrix"],
        index=[f"True_{e}" for e in EMOTION_NAMES],
        columns=[f"Pred_{e}" for e in EMOTION_NAMES],
    ).to_csv(model_dir / "confusion_matrix.csv")

    with open(model_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(result["classification_report"])

    json_payload = {
        k: v for k, v in result.items()
        if k not in {"classification_report", "confusion_matrix"}
    }

    with open(model_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    log.info(f"Saved {result['model'].upper()} results -> {model_dir}")


def save_summary(results, log):
    rows = []

    for r in results:
        rows.append({
            "model": r["model"].upper(),
            "mean_loso_accuracy": r["mean_loso_accuracy"],
            "std_loso_accuracy": r["std_loso_accuracy"],
            "overall_sample_accuracy": r["overall_sample_accuracy"],
            "best_fold": r["best_fold"],
            "worst_fold": r["worst_fold"],
            "n_folds": r["n_folds"],
            "tuned_params": str(r["tuned_params"]),
        })

    summary = pd.DataFrame(rows).sort_values(
        "mean_loso_accuracy",
        ascending=False,
    )

    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    log.info("\n" + "=" * 80)
    log.info("FINAL TUNED LOSO SUMMARY")
    log.info("=" * 80)
    log.info(summary)
    log.info(f"Summary saved -> {RESULTS_DIR / 'summary.csv'}")


def main():
    log = get_logger()

    log.info("=" * 80)
    log.info("SER Classical ML Pipeline | Tuned LOSO Final Evaluation")
    log.info("=" * 80)
    log.info("Tuning source: speaker-based cluster split")
    log.info("Cluster tuning split: Train=23 speakers, Val=5 speakers, Test=5 speakers")
    log.info(f"Tuned parameters: {TUNED_PARAMS}")

    X, y, speakers = load_data(log)
    models = build_models()

    results = []
    start_all = time.time()

    for name, model in models.items():
        start = time.time()
        result = run_loso(name, model, X, y, speakers, log)
        save_model_results(result, log)
        results.append(result)
        log.info(f"{name.upper()} finished in {time.time() - start:.1f}s")

    save_summary(results, log)

    log.info(f"\nDone. Total time: {time.time() - start_all:.1f}s")
    log.info(f"Results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()