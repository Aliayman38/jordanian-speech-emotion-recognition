"""
wav2vec_clf_fixed.py — Fixed version with smart early stopping and per-epoch logging.

Changes from broken version:
1. Uses original working clusterer splits (reverted from strict 70/15/15)
2. Gap-based stopping ONLY triggers after minimum epochs (no premature stops)
3. Logs train/val accuracy every epoch for plotting
4. Saves best model by val F1 (not gap)
"""

from pathlib import Path
import argparse, sys, json, time, copy, shutil, logging, warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from experiments.clusterer import get_stratified_speakers

RESULTS_DIR = ROOT / "outputs" / "results" / "wav2vec_clf_fixed"
CHECKPOINT_DIR = ROOT / "checkpoints" / "wav2vec_clf_fixed"

RANDOM_STATE = 42
DEFAULT_SEEDS = [42, 123, 7, 2025, 999]
EMOTION_NAMES = ["Happy", "Sad", "Angry", "Neutral"]
NUM_CLASSES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    return logging.getLogger("SER_FIXED")


# ── GRL ──────────────────────────────────────────────────────────────────────

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


def spectral_norm_linear(in_dim, out_dim, bias=True):
    layer = nn.Linear(in_dim, out_dim, bias=bias)
    return nn.utils.spectral_norm(layer)


# ── Data loading ─────────────────────────────────────────────────────────────

def _build_emb_index(feature_dir: Path, log) -> dict:
    index = {}
    for p in feature_dir.rglob("*.npy"):
        parts = p.parts
        if len(parts) < 3:
            continue
        key = (parts[-3].lower(), parts[-2].lower(), p.stem.lower())
        index[key] = p
    if not index:
        raise FileNotFoundError(f"No .npy files under {feature_dir}")
    log.info(f"Indexed {len(index)} embeddings from {feature_dir}")
    return index

def _row_key(rel_path: str) -> tuple:
    rel = str(rel_path).replace("\\", "/")
    parts = Path(rel).parts
    return (parts[-3].lower(), parts[-2].lower(), Path(parts[-1]).stem.lower())

def _detect_emb_dim(index: dict) -> int:
    arr = np.load(next(iter(index.values())))
    return int(arr.shape[0])

def load_features(feature_dir: Path, metadata_csv: Path, log):
    meta = pd.read_csv(metadata_csv)
    index = _build_emb_index(feature_dir, log)
    emb_dim = _detect_emb_dim(index)

    train_spks, val_spks, test_spks = get_stratified_speakers(metadata_csv)

    def load_split(df, name):
        feats, labels, speakers = [], [], []
        for _, row in df.iterrows():
            try:
                key = _row_key(row["rel_path"])
            except:
                continue
            p = index.get(key)
            if p is None:
                continue
            feats.append(np.load(p))
            labels.append(int(row["label"]))
            speakers.append(str(row.get("speaker_id", "unknown")))
        return np.stack(feats).astype(np.float32), np.array(labels, dtype=np.int64), np.array(speakers)

    X_train, y_train, spk_train = load_split(meta[meta["speaker_id"].isin(train_spks)], "train")
    X_val, y_val, spk_val = load_split(meta[meta["speaker_id"].isin(val_spks)], "val")
    X_test, y_test, spk_test = load_split(meta[meta["speaker_id"].isin(test_spks)], "test")

    all_spk = sorted(set(spk_train) | set(spk_val) | set(spk_test))
    spk_to_idx = {s: i for i, s in enumerate(all_spk)}
    num_speakers = len(all_spk)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    log.info(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)} | Speakers: {num_speakers}")
    return (X_train, X_val, X_test, y_train, y_val, y_test,
            np.array([spk_to_idx[s] for s in spk_train], dtype=np.int64),
            np.array([spk_to_idx[s] for s in spk_val], dtype=np.int64),
            np.array([spk_to_idx[s] for s in spk_test], dtype=np.int64),
            emb_dim, num_speakers)


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
        self.speaker_head = nn.Sequential(
            nn.Linear(prev, max(prev // 2, 32)), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(max(prev // 2, 32), num_speakers))

    def forward(self, x, return_speaker=False, alpha=None):
        if alpha is not None:
            self.grl.set_lambda(alpha)
        z = self.encoder(x)
        emotion_logits = self.emotion_head(z)
        if return_speaker or self.training:
            return emotion_logits, self.speaker_head(self.grl(z))
        return emotion_logits


# ── Mixup ────────────────────────────────────────────────────────────────────

def mixup_batch(xb, yb, alpha):
    if alpha <= 0:
        return xb, yb, yb, 1.0
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(xb.size(0), device=xb.device)
    return lam * xb + (1.0 - lam) * xb[idx], yb, yb[idx], lam


# ── Training with SMART early stopping ───────────────────────────────────────

def _make_loader(X, y, batch_size, shuffle, spk=None):
    tensors = [torch.from_numpy(X), torch.from_numpy(y)]
    if spk is not None:
        tensors.append(torch.from_numpy(spk))
    ds = torch.utils.data.TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=(DEVICE == "cuda"))


def train_one(params, X_train, y_train, spk_train, X_val, y_val, spk_val, emb_dim, num_speakers,
              seed=RANDOM_STATE, mixup_alpha=0.3, adversarial_weight=0.5, grl_schedule="linear",
              min_epochs=20, max_gap=0.25):
    """
    Smart early stopping:
    - Never stop before min_epochs (prevents premature stopping)
    - Stop if val F1 plateaus (standard)
    - Stop if gap exceeds max_gap AND val F1 is decreasing
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SpeakerInvariantMLP(in_dim=emb_dim, hidden_dims=params["hidden_dims"], dropout=params["dropout"],
                                num_speakers=num_speakers, use_spectral_norm=params.get("spectral_norm", True),
                                grl_lambda=params.get("grl_lambda", 1.0)).to(DEVICE)

    cls_w = torch.tensor(compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train),
                         dtype=torch.float32, device=DEVICE)
    emotion_loss_fn = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=params.get("label_smoothing", 0.2))
    speaker_loss_fn = nn.CrossEntropyLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"], betas=(0.9, 0.999))

    train_loader = _make_loader(X_train, y_train, params["batch_size"], shuffle=True, spk=spk_train)
    val_loader = _make_loader(X_val, y_val, params["batch_size"], shuffle=False, spk=spk_val)
    train_eval_loader = _make_loader(X_train, y_train, params["batch_size"] * 2, shuffle=False, spk=spk_train)

    best_f1, best_acc, best_state = -1.0, -1.0, None
    bad_f1, last_epoch = 0, 0
    epoch_history = []

    for epoch in range(1, params["epochs"] + 1):
        last_epoch = epoch

        # LR warmup then cosine
        warmup = params.get("warmup_epochs", 5)
        if epoch <= warmup:
            lr = params["lr"] * (epoch / warmup)
        else:
            progress = (epoch - warmup) / (params["epochs"] - warmup)
            min_lr = params.get("min_lr", 1e-6)
            lr = min_lr + (params["lr"] - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in opt.param_groups:
            pg['lr'] = lr

        # GRL schedule
        if grl_schedule == "linear":
            model.grl.set_lambda(min(params.get("grl_lambda", 1.0) * (epoch / params["epochs"]), 2.0))
        elif grl_schedule == "step":
            model.grl.set_lambda(0.1 if epoch < params["epochs"] // 3 else (1.0 if epoch < 2 * params["epochs"] // 3 else 2.0))

        # Train
        model.train()
        for xb, yb, spkb in train_loader:
            xb, yb, spkb = xb.to(DEVICE), yb.to(DEVICE), spkb.to(DEVICE)
            opt.zero_grad()
            if mixup_alpha > 0:
                mx, ya, yb_b, lam = mixup_batch(xb, yb, mixup_alpha)
                el, sl = model(mx, return_speaker=True)
                loss_emo = lam * emotion_loss_fn(el, ya) + (1.0 - lam) * emotion_loss_fn(el, yb_b)
            else:
                el, sl = model(xb, return_speaker=True)
                loss_emo = emotion_loss_fn(el, yb)
            (loss_emo + adversarial_weight * speaker_loss_fn(sl, spkb)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.get("grad_clip", 1.0))
            opt.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_preds = [model(xb.to(DEVICE), return_speaker=False).argmax(dim=1).cpu().numpy()
                          for xb, _, _ in train_eval_loader]
            train_y = [yb.numpy() for _, yb, _ in train_eval_loader]
            train_acc = accuracy_score(np.concatenate(train_y), np.concatenate(train_preds))

            val_preds = [model(xb.to(DEVICE), return_speaker=False).argmax(dim=1).cpu().numpy()
                        for xb, _, _ in val_loader]
            val_y = [yb.numpy() for _, yb, _ in val_loader]
            val_acc = accuracy_score(np.concatenate(val_y), np.concatenate(val_preds))
            val_f1 = f1_score(np.concatenate(val_y), np.concatenate(val_preds), average="macro", zero_division=0)

        gap = train_acc - val_acc
        epoch_history.append({"epoch": epoch, "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4),
                              "val_f1": round(val_f1, 4), "gap": round(gap, 4)})

        # Early stopping logic
        if val_f1 > best_f1:
            best_f1, best_acc = val_f1, val_acc
            best_state = copy.deepcopy(model.state_dict())
            bad_f1 = 0
        else:
            bad_f1 += 1

        # Only consider gap-based stop after min_epochs AND if val is actually dropping
        stop_gap = False
        if epoch > min_epochs and gap > max_gap:
            # Check if val has been dropping for last 3 epochs
            if len(epoch_history) >= 4:
                recent_vals = [e["val_f1"] for e in epoch_history[-4:]]
                if recent_vals[-1] <= recent_vals[0]:  # Flat or dropping
                    stop_gap = True

        if bad_f1 >= params["patience"]:
            break
        if stop_gap:
            break

    model.load_state_dict(best_state)
    return model, best_f1, best_acc, last_epoch, epoch_history


# ── Best params (hardcoded) ────────────────────────────────────────────────

def get_grid():
    return {
        "hidden_dims": [(64,)], "dropout": [0.7], "lr": [0.0005],
        "weight_decay": [0.01], "batch_size": [32], "epochs": [150],
        "patience": [25], "label_smoothing": [0.2], "spectral_norm": [True],
        "grl_lambda": [1.0], "warmup_epochs": [5], "grad_clip": [1.0], "min_lr": [1e-6],
    }


def tune(X_train, y_train, spk_train, X_val, y_val, spk_val, emb_dim, num_speakers,
         mixup_alpha, adversarial_weight, grl_schedule, log):
    grid = get_grid()
    combos = list(ParameterGrid(grid))
    best = {"f1": -1.0, "acc": -1.0, "model": None, "params": None, "history": []}

    log.info("=" * 70)
    log.info(f"BEST PARAMS (1 combo) | seed={RANDOM_STATE} | emb_dim={emb_dim}")
    log.info("=" * 70)

    for i, params in enumerate(combos, start=1):
        start = time.time()
        model, val_f1, val_acc, used_epochs, hist = train_one(
            params, X_train, y_train, spk_train, X_val, y_val, spk_val,
            emb_dim, num_speakers, seed=RANDOM_STATE,
            mixup_alpha=mixup_alpha, adversarial_weight=adversarial_weight, grl_schedule=grl_schedule)
        elapsed = time.time() - start
        log.info(f"[{i}/{len(combos)}] valF1={val_f1:.4f} valAcc={val_acc:.4f} epochs={used_epochs} | {elapsed:.1f}s")
        best.update(f1=val_f1, acc=val_acc, model=model, params=params, history=hist)

    return best, []


def train_ensemble(params, X_train, y_train, spk_train, X_val, y_val, spk_val,
                   emb_dim, num_speakers, seeds, mixup_alpha, adversarial_weight, grl_schedule, log):
    log.info("=" * 70)
    log.info(f"Ensemble | {len(seeds)} seeds: {seeds}")
    log.info("=" * 70)

    models, per_seed, all_hist = [], [], []
    for seed in seeds:
        start = time.time()
        model, val_f1, val_acc, used_epochs, hist = train_one(
            params, X_train, y_train, spk_train, X_val, y_val, spk_val,
            emb_dim, num_speakers, seed=seed,
            mixup_alpha=mixup_alpha, adversarial_weight=adversarial_weight, grl_schedule=grl_schedule)
        elapsed = time.time() - start
        log.info(f"[seed {seed:>4}] valF1={val_f1:.4f} valAcc={val_acc:.4f} epochs={used_epochs} | {elapsed:.1f}s")
        models.append(model)
        per_seed.append({"seed": seed, "val_f1": round(float(val_f1), 4), "val_acc": round(float(val_acc), 4), "epochs": used_epochs})
        all_hist.append(hist)

    return models, per_seed, all_hist


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
        "model": f"wav2vec_fixed_ens{len(models)}",
        "train_acc": round(train_acc, 4), "train_f1": round(train_f1, 4),
        "val_acc": round(val_acc, 4), "val_f1": round(val_f1, 4),
        "test_acc": round(test_acc, 4), "test_f1": round(test_f1, 4),
        "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_params.items()},
        "per_seed_val": per_seed,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def save_results(result, history, models, all_hist, log):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(result["classification_report"])
    pd.DataFrame(result["confusion_matrix"]).to_csv(RESULTS_DIR / "confusion_matrix.csv", index=False)

    for i, hist in enumerate(all_hist):
        pd.DataFrame(hist).to_csv(RESULTS_DIR / f"epoch_history_seed{DEFAULT_SEEDS[i]}.csv", index=False)

    for entry, model in zip(result["per_seed_val"], models):
        torch.save(model.state_dict(), CHECKPOINT_DIR / f"wav2vec_fixed_seed{entry['seed']:04d}.pt")

    log.info(f"Saved results → {RESULTS_DIR}")


def plot_curves(all_hist, seeds, save_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for hist, seed in zip(all_hist, seeds):
        epochs = [e["epoch"] for e in hist]
        train_accs = [e["train_acc"] for e in hist]
        val_accs = [e["val_acc"] for e in hist]
        ax.plot(epochs, train_accs, '--', alpha=0.5, label=f'Train {seed}')
        ax.plot(epochs, val_accs, '-', alpha=0.7, label=f'Val {seed}')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('All Seeds'); ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

    ax = axes[1]
    max_len = max(len(h) for h in all_hist)
    avg_train = np.zeros(max_len)
    avg_val = np.zeros(max_len)
    counts = np.zeros(max_len)
    for hist in all_hist:
        for i, e in enumerate(hist):
            avg_train[i] += e["train_acc"]; avg_val[i] += e["val_acc"]; counts[i] += 1
    avg_train /= counts; avg_val /= counts
    epochs = list(range(1, max_len + 1))
    ax.plot(epochs, avg_train, 'b-o', label='Avg Train', markersize=3)
    ax.plot(epochs, avg_val, 'r-s', label='Avg Val', markersize=3)
    ax.fill_between(epochs, avg_train, avg_val, alpha=0.2, color='red', label='Gap')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_title('Average'); ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_path = save_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved curves to {save_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_dir", type=Path, default=ROOT / "data" / "features" / "wav2vec")
    p.add_argument("--metadata", type=Path, default=ROOT / "data" / "metadata.csv")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--mixup_alpha", type=float, default=0.3)
    p.add_argument("--adversarial_weight", type=float, default=0.5)
    p.add_argument("--grl_schedule", type=str, default="linear", choices=["constant", "linear", "step"])
    p.add_argument("--n_seeds", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    if args.fresh:
        for d in (RESULTS_DIR, CHECKPOINT_DIR):
            if d.exists(): shutil.rmtree(d)

    log = get_logger()
    log.info("SER WAV2VEC Fixed | Smart Early Stopping + Per-Epoch Logging")
    log.info(f"Device: {DEVICE} | Feature dir: {args.feature_dir}")

    (X_train, X_val, X_test, y_train, y_val, y_test,
     spk_train, spk_val, spk_test, emb_dim, num_speakers) = load_features(args.feature_dir, args.metadata, log)

    best, _ = tune(X_train, y_train, spk_train, X_val, y_val, spk_val,
                   emb_dim, num_speakers, args.mixup_alpha, args.adversarial_weight, args.grl_schedule, log)

    seeds = DEFAULT_SEEDS[:args.n_seeds]
    models, per_seed, all_hist = train_ensemble(
        best["params"], X_train, y_train, spk_train, X_val, y_val, spk_val,
        emb_dim, num_speakers, seeds, args.mixup_alpha, args.adversarial_weight, args.grl_schedule, log)

    result = evaluate_all(models, best["params"], per_seed, X_train, y_train, X_val, y_val, X_test, y_test, log)
    save_results(result, [], models, all_hist, log)
    plot_curves(all_hist, seeds, RESULTS_DIR)
    log.info(f"Done. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
