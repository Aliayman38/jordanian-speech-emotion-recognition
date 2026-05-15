import torch
import torchaudio.transforms as T

class CNNSpectrogramExtractor:
    def __init__(self, sr=16000, n_mels=128, n_fft=1024, hop_length=512):
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __call__(self, waveform):
        # 1. Generate Mel-Spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # 2. Convert to DB scale
        spec_db = self.amplitude_to_db(mel_spec)
        
        # 3. Z-score Normalization (CRITICAL FIX)
        # This forces every spectrogram to have a mean of 0 and std of 1
        mean = spec_db.mean()
        std = spec_db.std()
        spec_normalized = (spec_db - mean) / (std + 1e-6) # 1e-6 prevents division by zero
        
        return spec_normalized
