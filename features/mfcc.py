# features/mfcc.py
"""
MFCC feature extraction, raw audio loading, and train-only augmentation.

Public API
----------
    load_audio(path)               → clean waveform (center-crop, val/test)
    load_audio_random_crop(path)   → waveform with random crop (train)
    augment_audio(y)               → augmented waveform (train-only)
    extract_mfcc(path)             → 1-D feature vector (~440 dims)
    extract_features_for_split(df) → (X, y) arrays for a DataFrame split
"""

import random

import librosa
import numpy as np
from tqdm import tqdm

from config import (
    AUG_GAIN_PROB, AUG_NOISE_PROB, AUG_SHIFT_PROB,
    HOP_LENGTH, MAX_LEN, N_FFT, N_MFCC, SAMPLE_RATE,
)


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_audio(
    file_path: str,
    max_len: int   = MAX_LEN,
    center_crop: bool = True,
) -> np.ndarray:
    """
    Load a WAV file, normalize, and pad/crop to *max_len* samples.
    Used for **validation and test** (deterministic).
    """
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    y = y.astype(np.float32)
    y -= y.mean()
    peak = np.abs(y).max()
    if peak > 0:
        y /= peak + 1e-6

    if len(y) > max_len:
        start = (len(y) - max_len) // 2 if center_crop else 0
        y = y[start : start + max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)), mode="constant")
    return y


def load_audio_random_crop(
    file_path: str,
    max_len: int = MAX_LEN,
) -> np.ndarray:
    """
    Load a WAV file, normalize, and apply a **random** crop.
    Used during **training** to add positional variety.
    """
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    y = y.astype(np.float32)
    y -= y.mean()
    peak = np.abs(y).max()
    if peak > 0:
        y /= peak + 1e-6

    if len(y) > max_len:
        start = random.randint(0, len(y) - max_len)
        y = y[start : start + max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)), mode="constant")
    return y


# ── Augmentation (train-only) ─────────────────────────────────────────────────

def augment_audio(y: np.ndarray) -> np.ndarray:
    """
    Apply light, emotion-safe augmentations to a **training** waveform.

    Augmentations applied stochastically:

    * **Gaussian noise** (level 0.001–0.006) — simulates microphone noise.
    * **Volume gain** (0.85–1.15) — simulates recording distance.
    * **Time shift** (circular roll ±8 %) — simulates utterance position.

    NOT applied: pitch shift (alters emotional cues), speed perturbation
    (changes prosody), reverb (too destructive for a small dataset).

    .. warning::
        Call this function **only on training samples**.
        Validation and test waveforms must always be loaded clean.
    """
    y = y.copy()

    if random.random() < AUG_NOISE_PROB:
        level = random.uniform(0.001, 0.006)
        y += level * np.random.randn(len(y)).astype(np.float32)

    if random.random() < AUG_GAIN_PROB:
        y *= random.uniform(0.85, 1.15)

    if random.random() < AUG_SHIFT_PROB:
        max_shift = int(0.08 * len(y))
        shift = random.randint(-max_shift, max_shift)
        y = np.roll(y, shift)

    return np.clip(y, -1.0, 1.0).astype(np.float32)


# ── MFCC feature extraction ───────────────────────────────────────────────────

def extract_mfcc(file_path: str, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Extract a rich 1-D feature vector from one WAV file (~440 dims).

    Components
    ----------
    * MFCC mean/std/min/max  (n_mfcc × 4)
    * MFCC delta + delta-delta mean/std  (n_mfcc × 4)
    * Chroma STFT mean  (12)
    * Spectral centroid, ZCR, RMS, bandwidth, rolloff  (5 × 2)
    * Spectral contrast mean/std  (7 × 2)
    * Mel-spectrogram mean/std  (n_mfcc × 2)
    * Pitch F0 statistics  (4)
    """
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    feats = []

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for stat in [np.mean, np.std, np.min, np.max]:
        feats.extend(stat(mfcc, axis=1))

    # Delta & delta-delta
    for order in [1, 2]:
        d = librosa.feature.delta(mfcc, order=order)
        feats.extend(np.mean(d, axis=1))
        feats.extend(np.std(d, axis=1))

    # Chroma
    feats.extend(np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1))

    # Spectral features
    for fn in [
        librosa.feature.spectral_centroid,
        librosa.feature.zero_crossing_rate,
        librosa.feature.rms,
        librosa.feature.spectral_bandwidth,
        librosa.feature.spectral_rolloff,
    ]:
        v = (
            fn(y=y)
            if fn in (librosa.feature.zero_crossing_rate, librosa.feature.rms)
            else fn(y=y, sr=sr)
        )
        feats.extend([np.mean(v), np.std(v)])

    # Spectral contrast
    sc = librosa.feature.spectral_contrast(y=y, sr=sr)
    feats.extend(np.mean(sc, axis=1))
    feats.extend(np.std(sc, axis=1))

    # Mel-spectrogram
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mfcc), ref=np.max
    )
    feats.extend(np.mean(mel, axis=1))
    feats.extend(np.std(mel, axis=1))

    # Pitch F0
    try:
        f0 = librosa.pyin(y, fmin=50, fmax=500)[0]
        f0 = f0[~np.isnan(f0)]
        feats.extend(
            [np.mean(f0), np.std(f0), np.min(f0), np.max(f0)]
            if len(f0) > 0
            else [0.0, 0.0, 0.0, 0.0]
        )
    except Exception:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(feats, dtype=np.float32)


def extract_features_for_split(df, desc: str = "") -> tuple:
    """
    Extract MFCC features for every row in *df*.

    Returns
    -------
    (X, y) : (np.ndarray, np.ndarray)
        Feature matrix and integer label vector.
    """
    import pandas as pd

    X, y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc or "MFCC"):
        try:
            X.append(extract_mfcc(row["file_path"]))
            y.append(row["label"])
        except Exception as e:
            print(f"  [skip] {row['file_path']}: {e}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
