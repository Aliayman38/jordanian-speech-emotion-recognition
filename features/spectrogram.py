# features/spectrogram.py
"""
Mel-spectrogram dataset for CNN-based SER.

The dataset loads raw audio on the fly, converts it to a dB-scaled
Mel-spectrogram, and pads / truncates the time axis to a fixed length
so that all tensors in a batch have the same shape.

Usage
-----
    from features.spectrogram import MelDataset
    from torch.utils.data import DataLoader

    ds = MelDataset(train_df)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
    mel, label = next(iter(loader))
    # mel.shape → (B, 1, N_MELS, MAX_FRAMES)
"""

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

from config import HOP_LENGTH, MAX_FRAMES, N_FFT, N_MELS, SAMPLE_RATE


class MelDataset(Dataset):
    """
    PyTorch Dataset that returns ``(mel_spectrogram, label)`` pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``file_path`` and ``label``.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row["file_path"])

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        # Peak normalise
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        # Mel-spectrogram → (1, N_MELS, T)
        mel = self.db_transform(self.mel_transform(waveform))

        # Pad / truncate to fixed MAX_FRAMES
        T = mel.shape[-1]
        if T < MAX_FRAMES:
            mel = torch.nn.functional.pad(mel, (0, MAX_FRAMES - T))
        else:
            mel = mel[:, :, :MAX_FRAMES]

        return mel, int(row["label"])
