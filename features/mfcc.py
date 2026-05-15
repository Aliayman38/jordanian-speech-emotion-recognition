from pathlib import Path
import sys

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from data.dataset import JordanianSERDataset


DATASET_DIR = PROJECT_ROOT / "Dataset"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "features" / "mfcc_features_loso.csv"

SAMPLE_RATE = 16000
N_MFCC = 40


def to_numpy_mono(waveform):
    """
    Convert torch waveform from dataset.py to 1D numpy array.
    Expected waveform shape: (channels, samples)
    """
    waveform = waveform.detach().cpu().numpy()

    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=0)

    return waveform.astype(np.float32)


def summarize(feature):
    """
    Convert time-based feature matrix into fixed-size vector.
    """
    mean = np.mean(feature, axis=1)
    std = np.std(feature, axis=1)
    return np.concatenate([mean, std])


def extract_mfcc_features(y):
    """
    Extract handcrafted SER features.
    Dataset loading / padding is already handled by dataset.py.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features = np.concatenate([
        summarize(mfcc),
        summarize(delta),
        summarize(delta2),
        summarize(chroma),
        summarize(spectral_centroid),
        summarize(spectral_bandwidth),
        summarize(spectral_rolloff),
        summarize(zcr),
        summarize(rms),
    ])

    return features


def build_feature_names():
    names = []

    groups = [
        ("mfcc", N_MFCC),
        ("delta_mfcc", N_MFCC),
        ("delta2_mfcc", N_MFCC),
        ("chroma", 12),
        ("spectral_centroid", 1),
        ("spectral_bandwidth", 1),
        ("spectral_rolloff", 1),
        ("zcr", 1),
        ("rms", 1),
    ]

    for group_name, count in groups:
        for stat in ["mean", "std"]:
            for i in range(count):
                names.append(f"{group_name}_{stat}_{i}")

    return names


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = JordanianSERDataset(data_dir=DATASET_DIR)
    feature_names = build_feature_names()

    rows = []

    print("[INFO] Extracting MFCC features using JordanianSERDataset...")
    print(f"[INFO] Dataset size: {len(dataset)}")

    for idx in tqdm(range(len(dataset))):
        try:
            waveform, label, speaker_id, gender = dataset[idx]

            y = to_numpy_mono(waveform)
            features = extract_mfcc_features(y)

            row = {
                "rel_path": dataset.file_rel_paths[idx],
                "label": int(label.item()),
                "speaker_id": int(speaker_id),
                "gender": int(gender),
            }

            for name, value in zip(feature_names, features):
                row[name] = float(value)

            rows.append(row)

        except Exception as e:
            print(f"[ERROR] Failed at index {idx}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print("[SUCCESS] MFCC feature extraction completed.")
    print(f"[INFO] Saved to: {OUTPUT_PATH}")
    print(f"[INFO] Output shape: {df.shape}")
    print(f"[INFO] Speakers count: {df['speaker_id'].nunique()}")


if __name__ == "__main__":
    main()