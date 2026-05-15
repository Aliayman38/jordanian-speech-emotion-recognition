from pathlib import Path
import sys, json, logging, time, warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from experiments.clusterer import get_stratified_speakers

FEATURES_CSV = ROOT / "outputs" / "features" / "mfcc_features_loso.csv"
METADATA_CSV = ROOT / "data" / "metadata.csv"
RESULTS_DIR = ROOT / "outputs" / "results" / "classical_cluster_tuning"

RANDOM_STATE = 42
META_COLS = {"rel_path", "label", "speaker_id", "gender"}
EMOTION_NAMES = ["Happy", "Sad", "Angry", "Neutral"]


def get_logger():
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
    return logging.getLogger("SER_CLUSTER_TUNING")


def load_features(log):
    df = pd.read_csv(FEATURES_CSV).dropna()
    feature_cols = [c for c in df.columns if c not in META_COLS]

    train_spks, val_spks, test_spks = get_stratified_speakers(METADATA_CSV)

    train_df = df[df["speaker_id"].isin(train_spks)]
    val_df = df[df["speaker_id"].isin(val_spks)]
    test_df = df[df["speaker_id"].isin(test_spks)]

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=int)

    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val = val_df["label"].to_numpy(dtype=int)

    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=int)

    log.info(f"Features: {len(feature_cols)}")
    log.info(f"Train speakers: {train_spks}")
    log.info(f"Val speakers  : {val_spks}")
    log.info(f"Test speakers : {test_spks}")
    log.info(f"Train samples : {len(y_train)}")
    log.info(f"Val samples   : {len(y_val)}")
    log.info(f"Test samples  : {len(y_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_model(name, params):
    if name == "svm":
        clf = SVC(
            kernel="rbf",
            C=params["C"],
            gamma=params["gamma"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    elif name == "knn":
        clf = KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            weights=params["weights"],
            metric=params["metric"],
            n_jobs=-1,
        )

    elif name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            alpha=params["alpha"],
            learning_rate_init=params["learning_rate_init"],
            activation="relu",
            solver="adam",
            batch_size=64,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(f"Unknown model: {name}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def get_grids():
    return {
        "svm": {
            "C": [1, 5, 10, 20, 50, 100],
            "gamma": ["scale", 0.01, 0.005, 0.001],
        },
        "knn": {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "mlp": {
            "hidden_layer_sizes": [(128,), (256,), (256, 128), (128, 64)],
            "alpha": [1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-3, 5e-4],
        },
    }


def tune_model(name, grid, X_train, y_train, X_val, y_val, log):
    best_acc = -1
    best_model = None
    best_params = None
    history = []

    combos = list(ParameterGrid(grid))

    log.info("=" * 70)
    log.info(f"Tuning {name.upper()} | {len(combos)} combinations")
    log.info("=" * 70)

    for i, params in enumerate(combos, start=1):
        start = time.time()

        model = make_model(name, params)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        elapsed = time.time() - start

        row = {
            "model": name,
            "combo": i,
            "val_acc": round(float(val_acc), 4),
            "time_sec": round(elapsed, 2),
            **params,
        }
        history.append(row)

        log.info(
            f"{name.upper()} [{i:02d}/{len(combos)}] "
            f"Val={val_acc:.4f} | Params={params} | {elapsed:.1f}s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_params = params

    log.info(f"BEST {name.upper()} | Val={best_acc:.4f} | Params={best_params}")
    return best_model, best_params, best_acc, history


def evaluate_test(name, model, best_params, best_val, X_test, y_test, log):
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    report = classification_report(
        y_test,
        test_pred,
        target_names=EMOTION_NAMES,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, test_pred)

    log.info("-" * 70)
    log.info(f"{name.upper()} FINAL")
    log.info(f"Best Validation Accuracy: {best_val:.4f}")
    log.info(f"Test Accuracy           : {test_acc:.4f}")
    log.info(f"Best Params             : {best_params}")

    return {
        "model": name,
        "best_val_acc": round(float(best_val), 4),
        "test_acc": round(float(test_acc), 4),
        "best_params": best_params,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def save_results(results, histories, log):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for res in results:
        model_dir = RESULTS_DIR / res["model"]
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

        with open(model_dir / "classification_report.txt", "w", encoding="utf-8") as f:
            f.write(res["classification_report"])

        pd.DataFrame(res["confusion_matrix"]).to_csv(
            model_dir / "confusion_matrix.csv",
            index=False,
        )

        summary_rows.append({
            "model": res["model"].upper(),
            "best_val_acc": res["best_val_acc"],
            "test_acc": res["test_acc"],
            "best_params": str(res["best_params"]),
        })

    for model_name, hist in histories.items():
        pd.DataFrame(hist).sort_values(
            "val_acc", ascending=False
        ).to_csv(RESULTS_DIR / f"{model_name}_tuning_history.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values("best_val_acc", ascending=False)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    log.info("\nFINAL SUMMARY")
    log.info(summary)
    log.info(f"Saved to: {RESULTS_DIR}")


def main():
    log = get_logger()
    log.info("SER Classical ML | Speaker-Based Cluster Split Tuning")

    X_train, X_val, X_test, y_train, y_val, y_test = load_features(log)

    results = []
    histories = {}

    for name, grid in get_grids().items():
        best_model, best_params, best_val, history = tune_model(
            name,
            grid,
            X_train,
            y_train,
            X_val,
            y_val,
            log,
        )

        histories[name] = history

        result = evaluate_test(
            name,
            best_model,
            best_params,
            best_val,
            X_test,
            y_test,
            log,
        )

        results.append(result)

    save_results(results, histories, log)


if __name__ == "__main__":
    main()
