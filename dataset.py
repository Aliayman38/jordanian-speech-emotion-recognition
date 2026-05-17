# dataset.py
"""
Dataset scanning and speaker-independent split.

Usage
-----
    from dataset import build_metadata, speaker_independent_split

    df = build_metadata(DATASET_DIR)
    train_df, val_df, test_df = speaker_independent_split(df)
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import EMOTION_ALIASES, EMOTION2ID, SEED


# ── Metadata builder ──────────────────────────────────────────────────────────

def build_metadata(dataset_dir: str) -> pd.DataFrame:
    """
    Scan ``<dataset_dir>/gender/Speaker_XX/Emotion/*.wav`` and return one
    row per WAV file.

    Columns
    -------
    file_path, rel_path, speaker_id, gender, emotion, label
    """
    records = []
    root = Path(dataset_dir)

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    print(f"[INFO] Scanning: {root}")

    for gender_dir in sorted(root.iterdir()):
        if not gender_dir.is_dir():
            continue
        gender = gender_dir.name.lower()
        if gender not in {"male", "female"}:
            print(f"  [skip] Unknown gender folder: {gender_dir.name}")
            continue

        for speaker_dir in sorted(gender_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = speaker_dir.name

            for emotion_dir in sorted(speaker_dir.iterdir()):
                if not emotion_dir.is_dir():
                    continue
                key = emotion_dir.name.lower().strip()
                if key not in EMOTION_ALIASES:
                    print(f"  [skip] Unknown emotion folder: {emotion_dir.name}")
                    continue

                emotion = EMOTION_ALIASES[key]
                label   = EMOTION2ID[emotion]

                for wav in sorted(emotion_dir.glob("*.wav")):
                    records.append({
                        "file_path":  str(wav),
                        "rel_path":   str(wav.relative_to(root)),
                        "speaker_id": speaker_id,
                        "gender":     gender,
                        "emotion":    emotion,
                        "label":      label,
                    })

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(
            "No WAV files found — check that DATASET_DIR points to the folder "
            "containing 'male/' and 'female/' sub-directories."
        )

    print(f"\nTotal files : {len(df)}")
    print(f"Speakers    : {df['speaker_id'].nunique()}")
    print(f"Gender dist :\n{df['gender'].value_counts().to_string()}")
    print(f"Emotion dist:\n{df['emotion'].value_counts().to_string()}")
    return df


# ── Speaker-independent split ─────────────────────────────────────────────────

def speaker_independent_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    seed: int          = SEED,
) -> tuple:
    """
    Split at the **speaker** level (70 / 15 / 15), gender-stratified.
    No speaker ever appears in more than one partition.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    speakers  = df[["speaker_id", "gender"]].drop_duplicates()
    spk_list  = speakers["speaker_id"].tolist()
    gend_list = speakers["gender"].tolist()

    train_spk, rem_spk = train_test_split(
        spk_list, test_size=(1 - train_ratio),
        stratify=gend_list, random_state=seed,
    )

    rem_gender = speakers.set_index("speaker_id").loc[rem_spk, "gender"].tolist()
    val_spk, test_spk = train_test_split(
        rem_spk, test_size=0.5,
        stratify=rem_gender, random_state=seed,
    )

    train_df = df[df["speaker_id"].isin(train_spk)].reset_index(drop=True)
    val_df   = df[df["speaker_id"].isin(val_spk)].reset_index(drop=True)
    test_df  = df[df["speaker_id"].isin(test_spk)].reset_index(drop=True)

    # Zero-overlap assertions
    assert not set(train_spk) & set(val_spk),  "Train/Val speaker overlap!"
    assert not set(train_spk) & set(test_spk), "Train/Test speaker overlap!"
    assert not set(val_spk)   & set(test_spk), "Val/Test speaker overlap!"

    sep = "=" * 58
    print(f"\n{sep}")
    print("  Speaker-Independent Split  (70 / 15 / 15)")
    print(sep)
    print(f"  Train : {len(train_spk):3d} speakers | {len(train_df):4d} files")
    print(f"  Val   : {len(val_spk):3d} speakers | {len(val_df):4d} files")
    print(f"  Test  : {len(test_spk):3d} speakers | {len(test_df):4d} files")
    print("  Overlap → 0  ✓")
    print(sep)

    return train_df, val_df, test_df


def verify_no_overlap(train_df, val_df, test_df) -> None:
    """Assert and print speaker overlap stats across all three splits."""
    tr = set(train_df["speaker_id"].astype(str))
    va = set(val_df["speaker_id"].astype(str))
    te = set(test_df["speaker_id"].astype(str))
    print(f"  Train-Val overlap  : {tr & va}")
    print(f"  Train-Test overlap : {tr & te}")
    print(f"  Val-Test overlap   : {va & te}")
    assert not (tr & va), "Leakage: train/val overlap"
    assert not (tr & te), "Leakage: train/test overlap"
    assert not (va & te), "Leakage: val/test overlap"
    print("  ✓ No overlap")
