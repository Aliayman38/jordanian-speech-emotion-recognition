# features/spectrogram.py
"""
Mel-spectrogram dataset for CNN-based SER.
"""

import librosa
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

from config import HOP_LENGTH, MAX_FRAMES, N_FFT, N_MELS, SAMPLE_RATE


class MelDataset(Dataset):
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

        # librosa avoids the torchcodec dependency
        y, _ = librosa.load(row["file_path"], sr=SAMPLE_RATE, mono=True)
        waveform = torch.from_numpy(y).unsqueeze(0)  # (1, T)

        # Peak normalise
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        # Mel-spectrogram -> (1, N_MELS, T)
        mel = self.db_transform(self.mel_transform(waveform))

        # Pad / truncate to fixed MAX_FRAMES
        T = mel.shape[-1]
        if T < MAX_FRAMES:
            mel = torch.nn.functional.pad(mel, (0, MAX_FRAMES - T))
        else:
            mel = mel[:, :, :MAX_FRAMES]

        return mel, int(row["label"])
