# features/wav2vec.py
"""
PyTorch Dataset and collation for Wav2Vec2 fine-tuning.

SERAudioDataset
    * ``augment=True``  → training  (random crop + augmentation)
    * ``augment=False`` → val / test (center crop, clean)

collate_fn
    Pads a batch of raw waveforms using the HuggingFace feature extractor
    and stacks labels into a single tensor.

Usage
-----
    from transformers import AutoFeatureExtractor
    from features.wav2vec import SERAudioDataset, collate_fn

    feature_extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)
    train_ds = SERAudioDataset(train_df, augment=True)
    loader   = DataLoader(train_ds, batch_size=4, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, feature_extractor))
"""

import torch
import pandas as pd
from torch.utils.data import Dataset

from config import MAX_LEN, SAMPLE_RATE
from features.mfcc import augment_audio, load_audio, load_audio_random_crop


class SERAudioDataset(Dataset):
    """
    Waveform dataset for Wav2Vec2 fine-tuning.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``file_path`` and ``label``.
    augment : bool
        If *True* apply random crop + augmentation (training mode).
        If *False* use center crop with no augmentation (eval mode).
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False) -> None:
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        if self.augment:
            audio = load_audio_random_crop(row["file_path"])
            audio = augment_audio(audio)          # train-only
        else:
            audio = load_audio(row["file_path"])  # clean, deterministic

        return {"audio": audio, "label": int(row["label"])}


def collate_fn(batch: list, feature_extractor) -> dict:
    """
    Custom collation: pad waveforms with the HuggingFace feature extractor
    and return a dict suitable for the Wav2Vec2 model forward pass.

    Parameters
    ----------
    batch : list of dicts
        Each dict has keys ``audio`` (np.ndarray) and ``label`` (int).
    feature_extractor : AutoFeatureExtractor
        Loaded from ``AutoFeatureExtractor.from_pretrained(BASE_MODEL)``.

    Returns
    -------
    dict with keys ``input_values``, ``attention_mask``, ``labels``.
    """
    audios = [b["audio"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

    inputs = feature_extractor(
        audios,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    return {
        "input_values":   inputs["input_values"],
        "attention_mask": inputs.get("attention_mask", None),
        "labels":         labels,
    }
