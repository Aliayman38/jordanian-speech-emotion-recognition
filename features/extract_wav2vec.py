# features/extract_wav2vec.py
"""
Wav2Vec2 embedding extraction — two modes:

1. ``extract_pretrained_embeddings``
   Uses a frozen, off-the-shelf Wav2Vec2 (e.g. ``facebook/wav2vec2-base``).
   Returns mean+std pooled hidden states (1536-dim for the base model).
   Used in the classical/CNN pipeline (Notebook 2, Cell 8).

2. ``extract_finetuned_features``
   Uses an already fine-tuned ``Wav2VecEmotionModel`` checkpoint.
   Returns three arrays per split: softmax probs, 256-dim intermediate
   features, and the full pooled hidden states.
   Used in the fusion pipeline (Notebook 1, Section 11).

Both functions cache results to ``outputs/features/`` to avoid re-running
the expensive forward passes.
"""

import os

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from config import BATCH_SIZE, SAMPLE_RATE, SAVE_DIR, device
from features.wav2vec import SERAudioDataset, collate_fn


# ── 1. Pretrained embeddings (frozen backbone) ────────────────────────────────

def _load_audio_fixed(path: str, max_duration: float = 5.0) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    max_len = int(SAMPLE_RATE * max_duration)
    y = y[:max_len] if len(y) > max_len else np.pad(y, (0, max_len - len(y)))
    return y.astype(np.float32)


@torch.no_grad()
def extract_pretrained_embeddings(
    train_df,
    val_df,
    test_df,
    model_name: str = "facebook/wav2vec2-base",
    cache_path: str | None = None,
) -> dict:
    """
    Extract mean+std pooled hidden states from a **frozen** pretrained
    Wav2Vec2 model.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
    model_name : str
        HuggingFace model identifier.
    cache_path : str or None
        If provided and the file exists, load from cache instead of running
        the model.  If *None*, defaults to
        ``outputs/features/wav2vec2_embeddings.npz``.

    Returns
    -------
    dict with keys:
        ``X_train``, ``y_train``, ``X_val``, ``y_val``, ``X_test``, ``y_test``
    """
    if cache_path is None:
        os.makedirs(os.path.join(SAVE_DIR, "features"), exist_ok=True)
        cache_path = os.path.join(SAVE_DIR, "features", "wav2vec2_embeddings.npz")

    if os.path.exists(cache_path):
        print(f"[INFO] Loading cached embeddings from {cache_path}")
        d = np.load(cache_path)
        return {k: d[k] for k in d.files}

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    wav2vec   = Wav2Vec2Model.from_pretrained(model_name).to(device)
    wav2vec.eval()

    def _extract(df, desc):
        embeddings, labels = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            y = _load_audio_fixed(row["file_path"])
            inputs = processor(
                y, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
            )
            out = wav2vec(
                inputs.input_values.to(device)
            ).last_hidden_state.squeeze(0)         # [T, H]
            emb = torch.cat(
                [out.mean(dim=0), out.std(dim=0)], dim=0
            ).cpu().numpy()
            embeddings.append(emb)
            labels.append(row["label"])
        return np.array(embeddings, dtype=np.float32), np.array(labels, dtype=np.int64)

    X_tr, y_tr = _extract(train_df, "Train Wav2Vec2")
    X_va, y_va = _extract(val_df,   "Val Wav2Vec2")
    X_te, y_te = _extract(test_df,  "Test Wav2Vec2")

    np.savez(
        cache_path,
        X_train=X_tr, y_train=y_tr,
        X_val=X_va,   y_val=y_va,
        X_test=X_te,  y_test=y_te,
    )
    print(f"[DONE] Saved embeddings → {cache_path}")

    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val":   X_va, "y_val":   y_va,
        "X_test":  X_te, "y_test":  y_te,
    }


# ── 2. Fine-tuned features (from Wav2VecEmotionModel) ────────────────────────

@torch.no_grad()
def extract_finetuned_features(
    df,
    name: str,
    model,
    feature_extractor,
    batch_size: int = BATCH_SIZE,
) -> tuple:
    """
    Extract three feature arrays from a **fine-tuned** ``Wav2VecEmotionModel``.

    The model's ``forward`` is called with ``return_features=True``, which
    returns ``(logits, feat256, pooled)``.

    Parameters
    ----------
    df : pd.DataFrame
    name : str
        Human-readable label for the tqdm progress bar.
    model : Wav2VecEmotionModel
        Must already have the best checkpoint loaded and be in eval mode.
    feature_extractor : AutoFeatureExtractor

    Returns
    -------
    (y, probs, feat256, pooled) : four np.ndarray arrays
        * y       — integer labels
        * probs   — softmax class probabilities (N, num_classes)
        * feat256 — 256-dim pre-logit features  (N, 256)
        * pooled  — mean+std pooled hidden states (N, 2H)
    """
    ds     = SERAudioDataset(df, augment=False)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, feature_extractor),
        num_workers=0,
    )

    all_labels, all_probs, all_feat, all_pooled = [], [], [], []

    for batch in tqdm(loader, desc=f"Extracting {name}"):
        iv = batch["input_values"].to(device)
        am = batch["attention_mask"]
        if am is not None:
            am = am.to(device)

        logits, feat256, pooled = model(iv, am, return_features=True)
        probs = torch.softmax(logits, dim=1)

        all_labels.append(batch["labels"].numpy())
        all_probs.append(probs.cpu().numpy())
        all_feat.append(feat256.cpu().numpy())
        all_pooled.append(pooled.cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_probs),
        np.concatenate(all_feat),
        np.concatenate(all_pooled),
    )
