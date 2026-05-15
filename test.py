import torch
from data.dataset import JordanianSERDataset
from data.augment import AudioAugmenter
from features.spectrogram import CNNSpectrogramExtractor

# 1. Initialize modules
augmenter = AudioAugmenter(p=1.0) # Force it for testing
extractor = CNNSpectrogramExtractor()

# 2. Load Dataset using your metadata.csv
# Update the path to where your audio files are stored
dataset = JordanianSERDataset(data_dir="./Dataset", transform=augmenter)

# 3. Pull one sample
waveform, label, speaker_id, gender = dataset[0]

# 4. Transform to Spectrogram (Your CNN Input)
spectrogram = extractor(waveform)

print(f"Successfully processed Speaker {speaker_id}")
print(f"Waveform Shape: {waveform.shape}") # Should be [1, 64000]
print(f"Spectrogram Shape (CNN Input): {spectrogram.shape}") # Should be [1, 128, 126]

from models.cnn_spec import EmotionCNN

# Test the model with the spectrogram we just created
model = EmotionCNN(num_classes=4)
# spectrogram is [1, 128, 126]. CNN expects [Batch, Channels, Height, Width]
# So we add a batch dimension to make it [1, 1, 128, 126]
output = model(spectrogram.unsqueeze(0))

print(f"CNN Model Output Shape: {output.shape}") 
# Expected: torch.Size([1, 4]) -> (One prediction for each of the 4 emotions)