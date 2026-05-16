"""
features/wav2vec.py — Extract rich statistical features from wav2vec2 frame outputs.

Uses facebook/wav2vec2-large-xlsr-53 model weights with Wav2Vec2FeatureExtractor
(since base XLSR-53 has no tokenizer bundled — it's a representation model only).

Computes per-dimension statistics across time:
  mean, std, max, min, q25, q75 (and optionally skew, kurt, median, range)

Output: (D * n_stats,) vector per audio file, where D=1024 for wav2vec2-large

Usage:
    python features/wav2vec.py \
        --input_dir Dataset \
        --output_dir data/features/wav2vec_stats \
        --stats mean std max min q25 q75
"""

from pathlib import Path
import argparse, sys, json, warnings, os
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Feature computation ───────────────────────────────────────────────────────

def compute_stats(frames: np.ndarray, stats: List[str]) -> np.ndarray:
    """
    frames: (T, D) array of frame-level features
    stats: list of statistic names to compute
    Returns: concatenated vector of all statistics
    """
    results = []
    for stat in stats:
        if stat == "mean":
            results.append(frames.mean(axis=0))
        elif stat == "std":
            results.append(frames.std(axis=0))
        elif stat == "max":
            results.append(frames.max(axis=0))
        elif stat == "min":
            results.append(frames.min(axis=0))
        elif stat == "q25":
            results.append(np.percentile(frames, 25, axis=0))
        elif stat == "q75":
            results.append(np.percentile(frames, 75, axis=0))
        elif stat == "skew":
            from scipy.stats import skew
            results.append(skew(frames, axis=0, bias=False))
        elif stat == "kurt":
            from scipy.stats import kurtosis
            results.append(kurtosis(frames, axis=0, bias=False))
        elif stat == "median":
            results.append(np.median(frames, axis=0))
        elif stat == "range":
            results.append(frames.max(axis=0) - frames.min(axis=0))
        else:
            raise ValueError(f"Unknown statistic: {stat}")

    return np.concatenate(results).astype(np.float32)


# ── Wav2Vec extraction ───────────────────────────────────────────────────────

def extract_wav2vec_frames(audio_path: Path, feature_extractor, model, device) -> np.ndarray:
    """Extract frame-level wav2vec features (T, D) without pooling."""
    import librosa

    wav, sr = librosa.load(str(audio_path), sr=16000)

    # Use feature_extractor (not processor) since XLSR-53 has no tokenizer
    inputs = feature_extractor(wav, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        hidden = outputs.last_hidden_state  # (batch, T, D)

    return hidden.squeeze(0).cpu().numpy()  # (T, D)


# ── Main extraction pipeline ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True,
                       help="Root directory with speaker/emotion/wav structure")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Where to save .npy feature files")
    parser.add_argument("--model", type=str, default="facebook/wav2vec2-large-xlsr-53",
                       help="Wav2vec model to use (default: facebook/wav2vec2-large-xlsr-53)")
    parser.add_argument("--stats", nargs="+", 
                       default=["mean", "std", "max", "min", "q25", "q75"],
                       help="Statistics to compute per dimension")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--token", type=str, default=None,
                       help="HuggingFace access token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    # Resolve token
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    print(f"Loading model: {args.model}")
    if token:
        print("Using provided HuggingFace token for authentication")

    # Use FeatureExtractor (not Processor) — XLSR-53 base has no tokenizer
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model, token=token)
    model = Wav2Vec2Model.from_pretrained(args.model, token=token).to(args.device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all wav files
    wav_files = list(args.input_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")

    extracted = 0
    errors = []

    for wav_path in tqdm(wav_files, desc="Extracting"):
        try:
            frames = extract_wav2vec_frames(wav_path, feature_extractor, model, args.device)
            features = compute_stats(frames, args.stats)

            # Build output path mirroring input structure
            rel_path = wav_path.relative_to(args.input_dir)
            out_path = args.output_dir / rel_path.with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            np.save(out_path, features)
            extracted += 1

        except Exception as e:
            errors.append((str(wav_path), str(e)))
            continue

    print(f"\nExtraction complete: {extracted}/{len(wav_files)} files")
    if errors:
        print(f"Errors: {len(errors)}")
        for path, err in errors[:5]:
            print(f"  {path}: {err}")

    if extracted > 0:
        sample = next(args.output_dir.rglob("*.npy"))
        feat = np.load(sample)
        print(f"\nFeature shape per file: {feat.shape}")
        print(f"Statistics used: {args.stats}")
        print(f"Wav2vec dim: {len(feat) // len(args.stats)}")


if __name__ == "__main__":
    main()
