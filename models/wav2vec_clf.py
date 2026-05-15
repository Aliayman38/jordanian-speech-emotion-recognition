"""
wav2vec_clf.py — MLP classifier on top of wav2vec2-large-xlsr-53 embeddings

Architecture:
    1024-dim embedding
        └─ Linear(1024→512) + BatchNorm + ReLU + Dropout(0.3)
        └─ Linear(512→256)  + BatchNorm + ReLU + Dropout(0.3)
        └─ Linear(256→4)    → logits

Two checkpoint files are maintained:
  ┌─ best_wav2vec_clf.pt   ── best model weights only  (used for inference)
  └─ resume_wav2vec_clf.pt ── full training state       (used to resume)

Press Ctrl+C at any time — the resume checkpoint is saved every epoch
automatically, so the next run continues from exactly where you stopped.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EMOTIONS = ["angry", "happy", "neutral", "sad"]


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data
    feature_dir:   str   = "data/features/wav2vec"
    val_split:     float = 0.15
    test_split:    float = 0.15
    random_seed:   int   = 42

    # Model
    input_dim:   int   = 1024
    hidden_dims: list  = field(default_factory=lambda: [512, 256])
    num_classes: int   = 4
    dropout:     float = 0.3

    # Training
    epochs:        int   = 100
    batch_size:    int   = 64
    learning_rate: float = 1e-3
    weight_decay:  float = 1e-4
    patience:      int   = 15   # early-stopping: epochs without val improvement
    lr_patience:   int   = 7    # ReduceLROnPlateau patience

    # Checkpoints
    #   best_path   → best model weights only    (inference)
    #   resume_path → full training state        (resume)
    best_path:   str = "outputs/best_wav2vec_clf.pt"
    resume_path: str = "outputs/resume_wav2vec_clf.pt"

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────

class EmotionDataset(Dataset):
    """
    Loads cached .npy feature files produced by wav2vec.py.

    Expected structure:
        feature_dir/
            angry/   *.npy
            happy/   *.npy
            neutral/ *.npy
            sad/     *.npy
    """

    def __init__(self, feature_dir: str, indices: np.ndarray, label_encoder: LabelEncoder):
        all_files, all_labels = self._load_index(feature_dir, label_encoder)
        self.files  = [all_files[i]  for i in indices]
        self.labels = [all_labels[i] for i in indices]

    @staticmethod
    def _load_index(feature_dir: str, label_encoder: LabelEncoder):
        feature_dir = Path(feature_dir)
        files, labels = [], []
        for npy_path in sorted(feature_dir.rglob("*.npy")):
            emotion = npy_path.parent.name
            if emotion not in label_encoder.classes_:
                log.warning("Skipping unknown emotion folder: %s", emotion)
                continue
            files.append(npy_path)
            labels.append(label_encoder.transform([emotion])[0])
        return files, labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feat  = np.load(self.files[idx]).astype(np.float32)
        label = self.labels[idx]
        return torch.tensor(feat), torch.tensor(label, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────

class Wav2VecClassifier(nn.Module):
    """3-layer MLP: BatchNorm + ReLU + Dropout after each hidden layer."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        dims = [cfg.input_dim] + cfg.hidden_dims
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [
                nn.Linear(in_d, out_d),
                nn.BatchNorm1d(out_d),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
            ]
        layers.append(nn.Linear(dims[-1], cfg.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Training helpers ──────────────────────────────────────────────────────────

def _make_weighted_sampler(labels: list, num_classes: int) -> WeightedRandomSampler:
    counts  = np.bincount(labels, minlength=num_classes).astype(float)
    class_w = 1.0 / np.where(counts > 0, counts, 1.0)
    sample_w = torch.tensor([class_w[l] for l in labels])
    return WeightedRandomSampler(sample_w, len(sample_w), replacement=True)


def _class_weights_tensor(labels: list, num_classes: int, device: str) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    w = 1.0 / np.where(counts > 0, counts, 1.0)
    w /= w.sum()
    return torch.tensor(w, dtype=torch.float32).to(device)


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Resumable training pipeline.

    Stopping and resuming
    ---------------------
    Press Ctrl+C at any time. The current epoch's state is written to
    resume_path before the process exits. On the next run, pass the same cfg
    and training continues from the next epoch automatically.

    Two files on disk
    -----------------
    best_path   — only the best model weights + metadata (small, for inference)
    resume_path — full training state (model + optimizer + scheduler + history)
                  overwritten every epoch; safe to Ctrl+C between epochs

    Usage
    -----
    trainer = Trainer(cfg)
    results  = trainer.run()

    # next session — resumes automatically
    trainer = Trainer(cfg)
    results  = trainer.run()
    """

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)

        Path(cfg.best_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.resume_path).parent.mkdir(parents=True, exist_ok=True)

        # ── label encoder ─────────────────────────────────────────────────────
        self.le = LabelEncoder()
        self.le.fit(EMOTIONS)

        # ── build dataset index & splits ──────────────────────────────────────
        all_files, all_labels = EmotionDataset._load_index(cfg.feature_dir, self.le)
        if not all_files:
            raise FileNotFoundError(
                f"No .npy files found under '{cfg.feature_dir}'. "
                "Run wav2vec.py first to extract features."
            )
        log.info("Total samples: %d", len(all_files))
        self._log_distribution(all_labels)

        idx = np.arange(len(all_files))
        idx_trainval, idx_test = train_test_split(
            idx, test_size=cfg.test_split, stratify=all_labels, random_state=cfg.random_seed
        )
        val_ratio       = cfg.val_split / (1.0 - cfg.test_split)
        labels_trainval = [all_labels[i] for i in idx_trainval]
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=val_ratio, stratify=labels_trainval, random_state=cfg.random_seed
        )
        log.info("Split → train: %d | val: %d | test: %d", len(idx_train), len(idx_val), len(idx_test))

        # ── datasets & loaders ────────────────────────────────────────────────
        train_ds     = EmotionDataset(cfg.feature_dir, idx_train, self.le)
        val_ds       = EmotionDataset(cfg.feature_dir, idx_val,   self.le)
        self.test_ds = EmotionDataset(cfg.feature_dir, idx_test,  self.le)

        sampler = _make_weighted_sampler(train_ds.labels, cfg.num_classes)
        self.train_loader = DataLoader(train_ds,     batch_size=cfg.batch_size, sampler=sampler,  num_workers=4, pin_memory=True)
        self.val_loader   = DataLoader(val_ds,       batch_size=cfg.batch_size, shuffle=False,    num_workers=4, pin_memory=True)
        self.test_loader  = DataLoader(self.test_ds, batch_size=cfg.batch_size, shuffle=False,    num_workers=4, pin_memory=True)

        # ── model, loss, optimizer, scheduler ─────────────────────────────────
        self.model = Wav2VecClassifier(cfg).to(cfg.device)
        log.info("Model parameters: %s", f"{sum(p.numel() for p in self.model.parameters()):,}")

        class_w        = _class_weights_tensor(train_ds.labels, cfg.num_classes, cfg.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_w)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=cfg.lr_patience, verbose=True
        )

        # ── training state (may be overwritten by _try_resume) ────────────────
        self.start_epoch:  int        = 1
        self.best_val_acc: float      = 0.0
        self.no_improve:   int        = 0
        self.history:      list[dict] = []

        # ── auto-resume ───────────────────────────────────────────────────────
        self._try_resume()

    # ── resume ────────────────────────────────────────────────────────────────

    def _try_resume(self):
        """Load full training state from resume_path if it exists."""
        resume_path = Path(self.cfg.resume_path)
        if not resume_path.exists():
            log.info("No resume checkpoint found — starting fresh.")
            return

        log.info("▶  Resume checkpoint found: %s", resume_path)
        ckpt = torch.load(resume_path, map_location=self.cfg.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])

        self.start_epoch  = ckpt["epoch"] + 1
        self.best_val_acc = ckpt["best_val_acc"]
        self.no_improve   = ckpt["no_improve"]
        self.history      = ckpt.get("history", [])

        log.info(
            "Resumed → next epoch: %d | best val acc: %.2f%% | "
            "early-stop counter: %d/%d",
            self.start_epoch,
            self.best_val_acc * 100,
            self.no_improve,
            self.cfg.patience,
        )

    # ── checkpoint helpers ────────────────────────────────────────────────────

    def _save_resume(self, epoch: int):
        """Overwrite resume checkpoint with current full training state."""
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_val_acc":    self.best_val_acc,
                "no_improve":      self.no_improve,
                "history":         self.history,
                "config":          self.cfg,
                "label_encoder_classes": self.le.classes_.tolist(),
            },
            self.cfg.resume_path,
        )

    def _save_best(self, epoch: int, val_acc: float):
        """Save only the best model weights (small file, for inference)."""
        torch.save(
            {
                "epoch":       epoch,
                "model_state": self.model.state_dict(),
                "val_acc":     val_acc,
                "config":      self.cfg,
                "label_encoder_classes": self.le.classes_.tolist(),
            },
            self.cfg.best_path,
        )

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        cfg   = self.cfg
        epoch = self.start_epoch - 1   # in case Ctrl+C before first epoch

        if self.start_epoch > cfg.epochs:
            log.info("All %d epochs already completed. Running test evaluation.", cfg.epochs)
            return self._test_evaluation()

        log.info(
            "Training epochs %d → %d  (device: %s)",
            self.start_epoch, cfg.epochs, cfg.device,
        )

        try:
            for epoch in range(self.start_epoch, cfg.epochs + 1):

                train_loss, train_acc = self._train_epoch()
                val_loss,   val_acc   = self._eval_epoch(self.val_loader)
                self.scheduler.step(val_acc)

                log.info(
                    "Epoch %3d/%d | train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%%",
                    epoch, cfg.epochs,
                    train_loss, train_acc * 100,
                    val_loss,   val_acc   * 100,
                )

                # record history
                self.history.append({
                    "epoch":      epoch,
                    "train_loss": round(train_loss, 6),
                    "train_acc":  round(train_acc,  6),
                    "val_loss":   round(val_loss,   6),
                    "val_acc":    round(val_acc,    6),
                })

                # best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.no_improve   = 0
                    self._save_best(epoch, val_acc)
                    log.info("  ✓ New best — saved to %s  (val acc %.2f%%)",
                             cfg.best_path, val_acc * 100)
                else:
                    self.no_improve += 1
                    log.info("  No improvement (%d/%d)", self.no_improve, cfg.patience)

                # resume state — written every epoch so Ctrl+C is always safe
                self._save_resume(epoch)

                # early stopping
                if self.no_improve >= cfg.patience:
                    log.info(
                        "Early stopping at epoch %d (no improvement for %d epochs).",
                        epoch, cfg.patience,
                    )
                    break

        except KeyboardInterrupt:
            log.info(
                "\n\n"
                "  ⚠  Training paused by user (Ctrl+C) after epoch %d.\n"
                "\n"
                "     Resume checkpoint : %s\n"
                "     Best model so far : %s  (val acc %.2f%%)\n"
                "\n"
                "     Re-run the same command to continue from epoch %d.\n"
                "     Add --fresh to start over from scratch.\n",
                epoch,
                cfg.resume_path,
                cfg.best_path,
                self.best_val_acc * 100,
                epoch + 1,
            )
            # _save_resume was called at end of last completed epoch;
            # if Ctrl+C hit mid-epoch the previous epoch's state is still intact.
            return {
                "interrupted":   True,
                "stopped_epoch": epoch,
                "best_val_acc":  self.best_val_acc,
                "resume_path":   cfg.resume_path,
            }

        return self._test_evaluation()

    # ── test evaluation ───────────────────────────────────────────────────────

    def _test_evaluation(self) -> dict:
        cfg = self.cfg
        if Path(cfg.best_path).exists():
            log.info("Loading best model from %s …", cfg.best_path)
            ckpt = torch.load(cfg.best_path, map_location=cfg.device)
            self.model.load_state_dict(ckpt["model_state"])
        else:
            log.warning("best_path not found — evaluating with current weights.")

        _, test_acc, preds, targets = self._eval_epoch(self.test_loader, return_preds=True)
        report = classification_report(targets, preds, target_names=self.le.classes_, digits=4)
        cm     = confusion_matrix(targets, preds)

        log.info("\n── Test Results ─────────────────────────────")
        log.info("Test accuracy : %.2f%%", test_acc * 100)
        log.info("\nClassification report:\n%s", report)
        log.info("Confusion matrix (rows=true, cols=pred):\n%s", cm)

        return {
            "interrupted":      False,
            "best_val_acc":     self.best_val_acc,
            "test_acc":         test_acc,
            "test_report":      report,
            "confusion_matrix": cm,
            "history":          self.history,
        }

    # ── epoch helpers ─────────────────────────────────────────────────────────

    def _train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for feats, labels in self.train_loader:
            feats, labels = feats.to(self.cfg.device), labels.to(self.cfg.device)
            self.optimizer.zero_grad()
            logits = self.model(feats)
            loss   = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)
        return total_loss / total, correct / total

    def _eval_epoch(self, loader: DataLoader, return_preds: bool = False):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for feats, labels in loader:
                feats, labels = feats.to(self.cfg.device), labels.to(self.cfg.device)
                logits = self.model(feats)
                loss   = self.criterion(logits, labels)
                preds  = logits.argmax(1)
                total_loss += loss.item() * len(labels)
                correct    += (preds == labels).sum().item()
                total      += len(labels)
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(labels.cpu().tolist())
        if return_preds:
            return total_loss / total, correct / total, all_preds, all_targets
        return total_loss / total, correct / total

    def _log_distribution(self, labels: list):
        counts = np.bincount(labels, minlength=self.cfg.num_classes)
        dist   = {self.le.inverse_transform([i])[0]: int(c) for i, c in enumerate(counts)}
        log.info("Class distribution: %s", dist)


# ── Inference helpers ─────────────────────────────────────────────────────────

def load_model(checkpoint_path: str) -> tuple:
    """Load best-model checkpoint for inference."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt["config"]
    le   = LabelEncoder()
    le.classes_ = np.array(ckpt["label_encoder_classes"])
    model = Wav2VecClassifier(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, le, cfg


def predict(model, features: np.ndarray, label_encoder: LabelEncoder, device: str = "cpu") -> tuple:
    """
    Predict emotion from a single 1024-dim wav2vec feature vector.
    Returns (emotion_string, softmax_probabilities_array).
    """
    model = model.to(device)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).squeeze().cpu().numpy()
    emotion = label_encoder.inverse_transform([probs.argmax()])[0]
    return emotion, probs


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Train (or resume) the wav2vec emotion classifier.\n"
            "Press Ctrl+C at any time to pause — re-run the same command to continue."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--feature_dir",  default="data/features/wav2vec")
    parser.add_argument("--best_path",    default="outputs/best_wav2vec_clf.pt",
                        help="Save location for the best model (inference use)")
    parser.add_argument("--resume_path",  default="outputs/resume_wav2vec_clf.pt",
                        help="Save/load location for full training state (resume use)")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--patience",     type=int,   default=15)
    parser.add_argument("--val_split",    type=float, default=0.15)
    parser.add_argument("--test_split",   type=float, default=0.15)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--fresh",        action="store_true",
                        help="Delete resume checkpoint and start training from scratch")
    args = parser.parse_args()

    cfg = TrainConfig(
        feature_dir   = args.feature_dir,
        best_path     = args.best_path,
        resume_path   = args.resume_path,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        learning_rate = args.lr,
        dropout       = args.dropout,
        patience      = args.patience,
        val_split     = args.val_split,
        test_split    = args.test_split,
        random_seed   = args.seed,
    )

    if args.fresh and Path(cfg.resume_path).exists():
        Path(cfg.resume_path).unlink()
        log.info("--fresh: deleted resume checkpoint, starting from scratch.")

    trainer = Trainer(cfg)
    results = trainer.run()

    if results.get("interrupted"):
        print(f"\n  Paused at epoch {results['stopped_epoch']}. Re-run to continue.\n")
    else:
        print(f"\n{'═' * 50}")
        print(f"  Best val accuracy : {results['best_val_acc'] * 100:.2f}%")
        print(f"  Test accuracy     : {results['test_acc'] * 100:.2f}%")
        print(f"{'═' * 50}\n")




'''
# Start training
python3 models/wav2vec_clf.py --feature_dir data/features/wav2vec

# Press Ctrl+C whenever you want → you'll see:
#   ⚠  Training paused after epoch 34.
#   Re-run the same command to continue from epoch 35.

# Resume — just run the exact same command again
python3 models/wav2vec_clf.py --feature_dir data/features/wav2vec

# Start completely from scratch (ignore existing resume)
python3 models/wav2vec_clf.py --feature_dir data/features/wav2vec --fresh
'''