"""
extract_wav2vec_aligned.py — Extract wav2vec features and save aligned npy files.

Handles Windows-style backslash paths in metadata.csv by normalizing to forward slashes.

Outputs:
    features.npy  — (N, D) float32
    labels.npy    — (N,) int64
    speakers.npy  — (N,) int64

Usage:
    python extract_wav2vec_aligned.py \
        --input_dir Dataset \
        --output_dir data/features/wav2vec_aligned \
        --metadata data/metadata.csv \
        --stats mean std max min q25 q75
"""

from pathlib import Path
import argparse, sys, os, warnings
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")


def compute_stats(frames: np.ndarray, stats: List[str]) -> np.ndarray:
    """Compute statistics per dimension across time."""
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
        elif stat == "median":
            results.append(np.median(frames, axis=0))
        elif stat == "range":
            results.append(frames.max(axis=0) - frames.min(axis=0))
        else:
            raise ValueError(f"Unknown stat: {stat}")
    return np.concatenate(results).astype(np.float32)


def extract_wav2vec_frames(audio_path: Path, feature_extractor, model, device) -> np.ndarray:
    """Extract frame-level wav2vec features (T, D)."""
    import librosa
    wav, sr = librosa.load(str(audio_path), sr=16000)
    inputs = feature_extractor(wav, sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        hidden = model(input_values).last_hidden_state
    return hidden.squeeze(0).cpu().numpy()


def normalize_path(path_str: str) -> str:
    """Normalize backslashes to forward slashes and strip leading/trailing slashes."""
    return path_str.replace("\\", "/").strip("/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True, help="Root with speaker/emotion/wav structure")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where to save .npy files")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata.csv")
    parser.add_argument("--model", type=str, default="facebook/wav2vec2-large-xlsr-53")
    parser.add_argument("--stats", nargs="+", default=["mean", "std", "max", "min", "q25", "q75"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    print(f"Loading model: {args.model}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model, token=token)
    model = Wav2Vec2Model.from_pretrained(args.model, token=token).to(args.device)
    model.eval()

    # Load metadata
    meta = pd.read_csv(args.metadata)
    print(f"Metadata: {len(meta)} entries")

    # Find all wav files and build normalized mapping
    wav_files = list(args.input_dir.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files")

    wav_map = {}
    for wav_path in wav_files:
        # Store with normalized relative path as key
        rel = normalize_path(str(wav_path.relative_to(args.input_dir)))
        wav_map[rel] = wav_path
        # Also store without extension for flexible matching
        wav_map[rel.replace(".wav", "")] = wav_path

    print(f"Built mapping with {len(wav_map)} entries")

    # Extract features in metadata order
    features_list, labels_list, speakers_list = [], [], []
    errors = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Extracting"):
        rel_path = normalize_path(str(row["rel_path"]))

        # Try multiple matching strategies
        wav_path = None
        candidates = [
            rel_path,                                    # Exact normalized path
            rel_path.replace(".npy", ".wav"),            # .npy -> .wav
            rel_path + ".wav",                           # Add .wav extension
            rel_path.replace(".wav", ""),                # Without extension
        ]

        for cand in candidates:
            if cand in wav_map:
                wav_path = wav_map[cand]
                break

        # Fallback: case-insensitive search
        if wav_path is None:
            cand_lower = rel_path.lower()
            for k, v in wav_map.items():
                if k.lower() == cand_lower or k.lower() == cand_lower.replace(".npy", ".wav"):
                    wav_path = v
                    break

        if wav_path is None:
            errors.append(f"NOT FOUND: {rel_path}")
            continue

        try:
            frames = extract_wav2vec_frames(wav_path, feature_extractor, model, args.device)
            feat = compute_stats(frames, args.stats)
            features_list.append(feat)
            labels_list.append(int(row["label"]))
            speakers_list.append(str(row["speaker_id"]))
        except Exception as e:
            errors.append(f"ERROR {rel_path}: {e}")
            continue

    print(f"\nExtracted: {len(features_list)}/{len(meta)} files")
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:10]:
            print(f"  {e}")

    if len(features_list) == 0:
        raise RuntimeError("No features extracted! Check path matching above.")

    # Convert to arrays
    features = np.stack(features_list).astype(np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    unique_speakers = sorted(set(speakers_list))
    spk_to_idx = {s: i for i, s in enumerate(unique_speakers)}
    speakers = np.array([spk_to_idx[s] for s in speakers_list], dtype=np.int64)

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "features.npy", features)
    np.save(args.output_dir / "labels.npy", labels)
    np.save(args.output_dir / "speakers.npy", speakers)

    with open(args.output_dir / "speaker_mapping.txt", "w") as f:
        for spk, idx in spk_to_idx.items():
            f.write(f"{idx}: {spk}\n")

    print(f"\nSaved to {args.output_dir}:")
    print(f"  features.npy: {features.shape}")
    print(f"  labels.npy:   {labels.shape} | classes: {np.unique(labels)}")
    print(f"  speakers.npy: {speakers.shape} | unique speakers: {len(unique_speakers)}")


if __name__ == "__main__":
    main()
