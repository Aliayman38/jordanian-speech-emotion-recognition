import torch
import random

class AudioAugmenter:
    def __init__(self, noise_factor=0.005, shift_limit=0.1, p=0.5):
        """
        Emotion-safe audio augmentation pipeline.
        
        Args:
            noise_factor (float): Amplitude of the added white noise.
            shift_limit (float): Max fraction of total audio length to shift horizontally.
            p (float): Probability of applying each augmentation.
        """
        self.noise_factor = noise_factor
        self.shift_limit = shift_limit
        self.p = p

    def add_white_noise(self, waveform):
        """Adds subtle Gaussian noise to the audio."""
        noise = torch.randn_like(waveform)
        augmented = waveform + self.noise_factor * noise
        # Clamp values to valid audio range [-1.0, 1.0]
        return torch.clamp(augmented, min=-1.0, max=1.0)

    def time_shift(self, waveform):
        """Shifts the audio horizontally (left or right) and rolls the overflow."""
        # waveform shape: (channels, time)
        shift_amt = int(random.uniform(-self.shift_limit, self.shift_limit) * waveform.shape[1])
        return torch.roll(waveform, shifts=shift_amt, dims=1)

    def __call__(self, waveform):
        """
        Applies a random combination of safe augmentations dynamically.
        """
        augmented = waveform.clone()
        
        if random.random() < self.p:
            augmented = self.add_white_noise(augmented)
            
        if random.random() < self.p:
            augmented = self.time_shift(augmented)
            
        return augmented