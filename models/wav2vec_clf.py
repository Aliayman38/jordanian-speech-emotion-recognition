"""
wav2vec_clf_v2_gap_stop.py — Gap-based early stopping to prevent overfitting.

Stops training when:
1. Val accuracy stops improving for N epochs (patience)
2. OR train-val gap exceeds max_allowed_gap (default 0.15 = 15%)
3. OR val accuracy decreases while train accuracy increases (divergence)

Logs per-epoch train/val accuracy for plotting.
"""

from pathlib import Path
import argparse, sys, json, time, copy, shutil, logging, warnings
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import ParameterGrid, StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from experiments.clusterer import get_stratified_speakers

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_WAV2VEC_DIR = ROOT / "data" / "features" / "wav2vec"
DEFAULT_METADATA    = ROOT / "data" / "metadata.csv"

RESULTS_DIR    = ROOT / "outputs" / "results" / "wav2vec_clf_v2_gap"
CHECKPOINT_DIR = ROOT / "checkpoints" / "wav2vec_clf_v2_gap"

RANDOM_STATE  = 42
DEFAULT_SEEDS = [42, 123, 7, 2025, 999]

EMOTION_NAMES = ["Happy", "Sad", "Angry", "Neutral"]
NUM_CLASSES   = 4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ── Logging ──────────────────────────────────────────────────────────────────

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
    return logging.getLogger("SER_WAV2VEC_CLF_V2_GAP")


# ── Gradient Reversal Layer ──────────────────────────────────────────────────

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_init=1.0):
        super().__init__()
        self.lambda_ = lambda_init

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


# ── Spectral Normalization wrapper ────────────────────────────────────────────

def spectral_norm_linear(in_dim, out_dim, bias=True):
    layer = nn.Linear(in_dim, out_dim, bias=bias)
    return nn.utils.spectral_norm(layer)


# ── Path resolution ──────────────────────────────────────────────────────────

def _build_emb_index(feature_dir: Path, log) -> dict:
    index, collisions = {}, 0
    for p in feature_dir.rglob("*.npy"):
        parts = p.parts
        if len(parts) < 3:
            continue
        key = (parts[-3].lower(), parts[-2].lower(), p.stem.lower())
        if key in index:
            collisions += 1
        index[key] = p
    if not index:
        raise FileNotFoundError(f"No .npy files found under {feature_dir}")
    log.info(f"Indexed {len(index)} cached embeddings from {feature_dir}")
    if collisions:
        log.warning(f"{collisions} duplicate keys in index (latest path kept)")
    return index


def _row_key(rel_path: str) -> tuple:
    rel = str(rel_path).replace("\\", "/")
    parts = Path(rel).parts
    if len(parts) < 3:
        raise ValueError(f"rel_path '{rel_path}' has fewer than 3 segments")
    return (parts[-3].lower(), parts[-2].lower(), Path(parts[-1]).stem.lower())


def _detect_emb_dim(index: dict) -> int:
    sample_path = next(iter(index.values()))
    arr = np.load(sample_path)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D embedding, got shape {arr.shape} at {sample_path}")
    return int(arr.shape[0])


# ── Data loading with speaker IDs ────────────────────────────────────────────

def _load_split_with_speakers(df: pd.DataFrame, index: dict, name: str, log):
    feats, labels, speakers, missing = [], [], [], []
    for _, row in df.iterrows():
        try:
            key = _row_key(row["rel_path"])
        except ValueError as e:
            log.warning(f"[{name}] {e}")
            missing.append(str(row["rel_path"]))
            continue
        p = index.get(key)
        if p is None:
            missing.append(str(row["rel_path"]))
            continue
        feats.append(np.load(p))
        labels.append(int(row["label"]))
        speakers.append(str(row.get("speaker_id", "unknown")))

    if missing:
        log.warning(f"[{name}] {len(missing)}/{len(df)} embeddings missing.")
        if len(missing) == len(df):
            raise FileNotFoundError(f"All embeddings missing for split '{name}'")

    return (np.stack(feats).astype(np.float32), np.array(labels, dtype=np.int64), np.array(speakers))


def load_features_with_speakers(feature_dir: Path, metadata_csv: Path, log):
    meta = pd.read_csv(metadata_csv)
    index = _build_emb_index(feature_dir, log)
    emb_dim = _detect_emb_dim(index)
    log.info(f"Embedding dim: {emb_dim}")

    train_spks, val_spks, test_spks = get_stratified_speakers(metadata_csv)

    X_train, y_train, spk_train = _load_split_with_speakers(meta[meta["speaker_id"].isin(train_spks)], index, "train", log)
    X_val,   y_val,   spk_val   = _load_split_with_speakers(meta[meta["speaker_id"].isin(val_spks)],   index, "val",   log)
    X_test,  y_test,  spk_test  = _load_split_with_speakers(meta[meta["speaker_id"].isin(test_spks)],  index, "test",  log)

    all_speakers = sorted(set(spk_train) | set(spk_val) | set(spk_test))
    spk_to_idx = {s: i for i, s in enumerate(all_speakers)}
    num_speakers = len(all_speakers)

    spk_train_idx = np.array([spk_to_idx[s] for s in spk_train], dtype=np.int64)
    spk_val_idx   = np.array([spk_to_idx[s] for s in spk_val],   dtype=np.int64)
    spk_test_idx  = np.array([spk_to_idx[s] for s in spk_test],  dtype=np.int64)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    log.info(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)} | Speakers: {num_speakers}")
    return (X_train, X_val, X_test, y_train, y_val, y_test, spk_train_idx, spk_val_idx, spk_test_idx, emb_dim, num_speakers)


# ── Model ────────────────────────────────────────────────────────────────────

class SpeakerInvariantMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple = (64,), num_classes: int = NUM_CLASSES,
                 num_speakers: int = 33, dropout: float = 0.7, use_spectral_norm: bool = True, grl_lambda: float = 1.0):
        super().__init__()
        encoder_layers, prev = [], in_dim
        for h in hidden_dims:
            if use_spectral_norm:
                encoder_layers.append(spectral_norm_linear(prev, h))
            else:
                encoder_layers.append(nn.Linear(prev, h))
            encoder_layers += [nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*encoder_layers)
        self.emotion_head = nn.Linear(prev, num_classes)
        self.grl = GradientReversalLayer(lambda_init=grl_lambda)
        self.speaker_head = nn.Sequential(nn.Linear(prev, max(prev // 2, 32)), nn.ReLU(), nn.Dropout(dropout), nn.Linear(max(prev // 2, 32), num_speakers))

    def forward(self, x, return_speaker=False, alpha=None):
        if alpha is not None:
            self.grl.set_lambda(alpha)
        z = self.encoder(x)
        emotion_logits = self.emotion_head(z)
        if return_speaker or self.training:
            return emotion_logits, self.speaker_head(self.grl(z))
        return emotion_logits


# ── Mixup ────────────────────────────────────────────────────────────────────

def mixup_batch(xb: torch.Tensor, yb: torch.Tensor, alpha: float):
    if alpha <= 0:
        return xb, yb, yb, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(xb.size(0), device=xb.device)
    return lam * xb + (1.0 - lam) * xb[idx], yb, yb[idx], lam


# ── LR Scheduler ───────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# ── DataLoader helper ────────────────────────────────────────────────────────

def _make_loader(X, y, batch_size, shuffle, spk=None):
    tensors = [torch.from_numpy(X), torch.from_numpy(y)]
    if spk is not None:
        tensors.append(torch.from_numpy(spk))
    ds = torch.utils.data.TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=(DEVICE == "cuda"))


# ── Training with GAP-BASED EARLY STOPPING ─────────────────────────────────

def train_one_adversarial(
    params, X_train, y_train, spk_train,
    X_val, y_val, spk_val,
    emb_dim: int, num_speakers: int,
    seed: int = RANDOM_STATE,
    mixup_alpha: float = 0.3,
    adversarial_weight: float = 0.5,
    grl_schedule: str = "linear",
    max_gap: float = 0.15,  # STOP if train_acc - val_acc > 15%
    gap_patience: int = 5,   # STOP if gap grows for 5 consecutive epochs
):
    """
    Train with gap-based early stopping.
    Returns: model, best_val_f1, best_val_acc, last_epoch, epoch_history
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SpeakerInvariantMLP(
        in_dim=emb_dim,
        hidden_dims=params["hidden_dims"],
        dropout=params["dropout"],
        num_speakers=num_speakers,
        use_spectral_norm=params.get("spectral_norm", True),
        grl_lambda=params.get("grl_lambda", 1.0),
    ).to(DEVICE)

    cls_w = compute_class_weight(class_weight="balanced", classes=np.arange(NUM_CLASSES), y=y_train)
    cls_w = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)

    emotion_loss_fn = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=params.get("label_smoothing", 0.2))
    speaker_loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], betas=(0.9, 0.999))
    sched = WarmupCosineScheduler(opt, params.get("warmup_epochs", 5), params["epochs"], params["lr"], params.get("min_lr", 1e-6))

    train_loader = _make_loader(X_train, y_train, params["batch_size"], shuffle=True, spk=spk_train)
    val_loader   = _make_loader(X_val,   y_val,   params["batch_size"], shuffle=False, spk=spk_val)
    train_eval_loader = _make_loader(X_train, y_train, params["batch_size"] * 2, shuffle=False, spk=spk_train)

    # Tracking
    best_f1, best_acc, best_state = -1.0, -1.0, None
    bad_f1, bad_gap = 0, 0
    last_epoch = 0
    epoch_history = []  # For plotting

    for epoch in range(1, params["epochs"] + 1):
        last_epoch = epoch
        current_lr = sched.step()

        # GRL schedule
        if grl_schedule == "linear":
            model.grl.set_lambda(min(params.get("grl_lambda", 1.0) * (epoch / params["epochs"]), 2.0))
        elif grl_schedule == "step":
            model.grl.set_lambda(0.1 if epoch < params["epochs"] // 3 else (1.0 if epoch < 2 * params["epochs"] // 3 else 2.0))

        # Training
        model.train()
        for xb, yb, spkb in train_loader:
            xb, yb, spkb = xb.to(DEVICE), yb.to(DEVICE), spkb.to(DEVICE)
            opt.zero_grad()
            if mixup_alpha > 0:
                mixed_x, ya, yb_b, lam = mixup_batch(xb, yb, mixup_alpha)
                el, sl = model(mixed_x, return_speaker=True)
                loss_emo = lam * emotion_loss_fn(el, ya) + (1.0 - lam) * emotion_loss_fn(el, yb_b)
                loss_spk = speaker_loss_fn(sl, spkb)
            else:
                el, sl = model(xb, return_speaker=True)
                loss_emo = emotion_loss_fn(el, yb)
                loss_spk = speaker_loss_fn(sl, spkb)
            (loss_emo + adversarial_weight * loss_spk).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.get("grad_clip", 1.0))
            opt.step()

        # Evaluation: Train
        model.eval()
        train_preds, train_y = [], []
        with torch.no_grad():
            for xb, yb, _ in train_eval_loader:
                train_preds.append(model(xb.to(DEVICE), return_speaker=False).argmax(dim=1).cpu().numpy())
                train_y.append(yb.numpy())
        train_acc = accuracy_score(np.concatenate(train_y), np.concatenate(train_preds))

        # Evaluation: Val
        val_preds, val_y = [], []
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                val_preds.append(model(xb.to(DEVICE), return_speaker=False).argmax(dim=1).cpu().numpy())
                val_y.append(yb.numpy())
        val_preds_all = np.concatenate(val_preds)
        val_y_all = np.concatenate(val_y)
        val_acc = accuracy_score(val_y_all, val_preds_all)
        val_f1 = f1_score(val_y_all, val_preds_all, average="macro", zero_division=0)

        gap = train_acc - val_acc

        # Log every epoch for plotting
        epoch_history.append({
            "epoch": epoch,
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
            "val_f1": round(val_f1, 4),
            "gap": round(gap, 4),
            "lr": round(current_lr, 6),
        })

        # ── EARLY STOPPING LOGIC ──
        stop_reason = None

        # 1. Standard: val F1 not improving
        if val_f1 > best_f1:
            best_f1, best_acc = val_f1, val_acc
            best_state = copy.deepcopy(model.state_dict())
            bad_f1 = 0
            bad_gap = 0
        else:
            bad_f1 += 1

        # 2. Gap-based: if gap exceeds max_gap, increment counter
        if gap > max_gap:
            bad_gap += 1
        else:
            bad_gap = 0

        # 3. Divergence: val decreasing while train increasing
        if len(epoch_history) >= 2:
            prev = epoch_history[-2]
            if train_acc > prev["train_acc"] and val_acc < prev["val_acc"]:
                bad_gap += 1  # Extra penalty for divergence

        # Stop conditions
        if bad_f1 >= params["patience"]:
            stop_reason = f"val F1 plateau ({bad_f1} epochs)"
        elif bad_gap >= gap_patience:
            stop_reason = f"generalization gap exceeded {max_gap} for {gap_patience} epochs (gap={gap:.3f})"

        if stop_reason:
            # logging.info(f"  Early stop at epoch {epoch}: {stop_reason}")
            break

    model.load_state_dict(best_state)
    return model, best_f1, best_acc, last_epoch, epoch_history


# ── Best hyperparameters (hardcoded) ───────────────────────────────────────

def get_grid_v2():
    return {
        "hidden_dims": [(64,)], "dropout": [0.7], "lr": [0.0005],
        "weight_decay": [0.01], "batch_size": [32], "epochs": [150],
        "patience": [25], "label_smoothing": [0.2], "spectral_norm": [True],
        "grl_lambda": [1.0], "warmup_epochs": [5], "grad_clip": [1.0], "min_lr": [1e-6],
    }


# ── Tuning (1 combo only) ──────────────────────────────────────────────────

def tune_adversarial(X_train, y_train, spk_train, X_val, y_val, spk_val, emb_dim, num_speakers,
                     mixup_alpha, adversarial_weight, grl_schedule, log):
    grid = get_grid_v2()
    combos = list(ParameterGrid(grid))
    best = {"f1": -1.0, "acc": -1.0, "model": None, "params": None, "history": []}

    log.info("=" * 70)
    log.info(f"USING BEST HYPERPARAMETERS (1 combo) | seed={RANDOM_STATE} | emb_dim={emb_dim}")
    log.info("=" * 70)

    for i, params in enumerate(combos, start=1):
        start = time.time()
        model, val_f1, val_acc, used_epochs, epoch_history = train_one_adversarial(
            params, X_train, y_train, spk_train, X_val, y_val, spk_val,
            emb_dim=emb_dim, num_speakers=num_speakers, seed=RANDOM_STATE,
            mixup_alpha=mixup_alpha, adversarial_weight=adversarial_weight, grl_schedule=grl_schedule,
        )
        elapsed = time.time() - start

        log.info(f"[{i}/{len(combos)}] valF1={val_f1:.4f} valAcc={val_acc:.4f} epochs={used_epochs} | {elapsed:.1f}s")
        log.info(f"  Epoch history saved with {len(epoch_history)} points")

        best.update(f1=val_f1, acc=val_acc, model=model, params=params, history=epoch_history)

    return best, []


# ── Ensemble ─────────────────────────────────────────────────────────────────

def train_ensemble_adversarial(params, X_train, y_train, spk_train, X_val, y_val, spk_val,
                               emb_dim, num_speakers, seeds, mixup_alpha, adversarial_weight, grl_schedule, log):
    log.info("=" * 70)
    log.info(f"Ensemble | {len(seeds)} seeds: {seeds}")
    log.info("=" * 70)

    models, per_seed, all_histories = [], [], []
    for seed in seeds:
        start = time.time()
        model, val_f1, val_acc, used_epochs, epoch_history = train_one_adversarial(
            params, X_train, y_train, spk_train, X_val, y_val, spk_val,
            emb_dim=emb_dim, num_speakers=num_speakers, seed=seed,
            mixup_alpha=mixup_alpha, adversarial_weight=adversarial_weight, grl_schedule=grl_schedule,
        )
        elapsed = time.time() - start
        log.info(f"[seed {seed:>4}] valF1={val_f1:.4f} valAcc={val_acc:.4f} epochs={used_epochs} | {elapsed:.1f}s")
        models.append(model)
        per_seed.append({"seed": seed, "val_f1": round(float(val_f1), 4), "val_acc": round(float(val_acc), 4), "epochs": used_epochs})
        all_histories.append(epoch_history)

    return models, per_seed, all_histories


# ── Evaluation ──────────────────────────────────────────────────────────────

def _ensemble_probs(models, X):
    dummy_y = np.zeros(len(X), dtype=np.int64)
    loader = _make_loader(X, dummy_y, batch_size=256, shuffle=False)
    probs_sum = None
    for model in models:
        model.eval()
        chunks = []
        with torch.no_grad():
            for xb, _ in loader:
                chunks.append(torch.softmax(model(xb.to(DEVICE), return_speaker=False), dim=1).cpu().numpy())
        probs = np.concatenate(chunks, axis=0)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(models)


def _eval_ens(models, X, y):
    probs = _ensemble_probs(models, X)
    preds = probs.argmax(axis=1)
    return float(accuracy_score(y, preds)), float(f1_score(y, preds, average="macro", zero_division=0)), preds, probs


def evaluate_all(models, best_params, per_seed, X_train, y_train, X_val, y_val, X_test, y_test, log):
    train_acc, train_f1, _, _ = _eval_ens(models, X_train, y_train)
    val_acc, val_f1, _, _ = _eval_ens(models, X_val, y_val)
    test_acc, test_f1, test_p, _ = _eval_ens(models, X_test, y_test)

    report = classification_report(y_test, test_p, target_names=EMOTION_NAMES, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, test_p)

    log.info("-" * 70)
    log.info(f"FINAL ({len(models)}-seed ensemble)")
    log.info(f"{'Split':<8} {'Accuracy':>10} {'MacroF1':>10}")
    log.info(f"{'Train':<8} {train_acc:>10.4f} {train_f1:>10.4f}")
    log.info(f"{'Val':<8}   {val_acc:>10.4f} {val_f1:>10.4f}")
    log.info(f"{'Test':<8}  {test_acc:>10.4f} {test_f1:>10.4f}")
    log.info(f"Gap (train→test): +{train_acc - test_acc:.4f} acc | +{train_f1 - test_f1:.4f} F1")
    log.info(f"Best Params: {best_params}")
    for s in per_seed:
        log.info(f"  seed={s['seed']:>4} | valF1={s['val_f1']:.4f} | valAcc={s['val_acc']:.4f} | epochs={s['epochs']}")
    log.info("\nTest-set classification report:\n" + report)

    return {
        "model": f"wav2vec_adv_ens{len(models)}",
        "train_acc": round(train_acc, 4), "train_f1": round(train_f1, 4),
        "val_acc": round(val_acc, 4), "val_f1": round(val_f1, 4),
        "test_acc": round(test_acc, 4), "test_f1": round(test_f1, 4),
        "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_params.items()},
        "per_seed_val": per_seed,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def save_results(result, history, models, all_histories, log):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(result["classification_report"])

    pd.DataFrame(result["confusion_matrix"]).to_csv(RESULTS_DIR / "confusion_matrix.csv", index=False)

    # Save per-epoch histories for plotting
    for i, hist in enumerate(all_histories):
        pd.DataFrame(hist).to_csv(RESULTS_DIR / f"epoch_history_seed{DEFAULT_SEEDS[i]}.csv", index=False)

    for entry, model in zip(result["per_seed_val"], models):
        torch.save(model.state_dict(), CHECKPOINT_DIR / f"wav2vec_adv_clf_seed{entry['seed']:04d}.pt")

    log.info(f"Saved results → {RESULTS_DIR}")
    log.info(f"Saved epoch histories for {len(all_histories)} seeds → {RESULTS_DIR}")


# ── Plotting helper ─────────────────────────────────────────────────────────

def plot_training_curves(all_histories, seeds, save_dir: Path):
    """Plot train/val accuracy curves for all seeds."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: All seeds overlaid
    ax = axes[0]
    for hist, seed in zip(all_histories, seeds):
        epochs = [e["epoch"] for e in hist]
        train_accs = [e["train_acc"] for e in hist]
        val_accs = [e["val_acc"] for e in hist]
        ax.plot(epochs, train_accs, '--', alpha=0.5, label=f'Train seed {seed}')
        ax.plot(epochs, val_accs, '-', alpha=0.7, label=f'Val seed {seed}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Train vs Val Accuracy (All Seeds)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Plot 2: Average curves
    ax = axes[1]
    max_len = max(len(h) for h in all_histories)
    avg_train = np.zeros(max_len)
    avg_val = np.zeros(max_len)
    counts = np.zeros(max_len)
    for hist in all_histories:
        for i, e in enumerate(hist):
            avg_train[i] += e["train_acc"]
            avg_val[i] += e["val_acc"]
            counts[i] += 1
    avg_train /= counts
    avg_val /= counts
    epochs = list(range(1, max_len + 1))
    ax.plot(epochs, avg_train, 'b-o', label='Avg Train', markersize=3)
    ax.plot(epochs, avg_val, 'r-s', label='Avg Val', markersize=3)
    ax.fill_between(epochs, avg_train, avg_val, alpha=0.2, color='red', label='Gap')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Average Train vs Val Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_path = save_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_dir", type=Path, default=DEFAULT_WAV2VEC_DIR)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--mixup_alpha", type=float, default=0.3)
    p.add_argument("--adversarial_weight", type=float, default=0.5)
    p.add_argument("--grl_schedule", type=str, default="linear", choices=["constant", "linear", "step"])
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--max_gap", type=float, default=0.15, help="Stop if train-val gap exceeds this")
    p.add_argument("--gap_patience", type=int, default=5, help="Epochs of gap growth before stopping")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fresh:
        for d in (RESULTS_DIR, CHECKPOINT_DIR):
            if d.exists(): shutil.rmtree(d)

    log = get_logger()
    log.info("SER WAV2VEC V2 | Gap-Based Early Stopping")
    log.info(f"Device: {DEVICE} | Feature dir: {args.feature_dir}")
    log.info(f"Max gap: {args.max_gap} | Gap patience: {args.gap_patience}")

    (X_train, X_val, X_test, y_train, y_val, y_test,
     spk_train, spk_val, spk_test, emb_dim, num_speakers) = load_features_with_speakers(args.feature_dir, args.metadata, log)

    best, _ = tune_adversarial(X_train, y_train, spk_train, X_val, y_val, spk_val,
                               emb_dim, num_speakers, args.mixup_alpha, args.adversarial_weight, args.grl_schedule, log)

    seeds = DEFAULT_SEEDS[:args.n_seeds]
    models, per_seed, all_histories = train_ensemble_adversarial(
        best["params"], X_train, y_train, spk_train, X_val, y_val, spk_val,
        emb_dim, num_speakers, seeds, args.mixup_alpha, args.adversarial_weight, args.grl_schedule, log)

    result = evaluate_all(models, best["params"], per_seed, X_train, y_train, X_val, y_val, X_test, y_test, log)
    save_results(result, [], models, all_histories, log)

    # Plot curves
    plot_training_curves(all_histories, seeds, RESULTS_DIR)
    log.info(f"Done. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
