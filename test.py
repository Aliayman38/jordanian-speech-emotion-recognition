import torch
import os
from data.dataset import JordanianSERDataset
from features.spectrogram import CNNSpectrogramExtractor

def test_dataset_structure():
    # 1. Define the data path
    # Ensure the directory is named 'Dataset' (Case Sensitive on Linux)
    data_path = "./Dataset"

    print(f"--- Environment Check ---")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Absolute Dataset Path: {os.path.abspath(data_path)}")

    if not os.path.exists(data_path):
        print(f"[ERROR] Directory NOT found at: {os.path.abspath(data_path)}")
        return

    # 2. Initialize Dataset
    try:
        print(f"\n[INFO] Loading Jordanian Dialect SER Dataset...")
        dataset = JordanianSERDataset(data_dir=data_path, transform=None)

        print(f"--- Statistics ---")
        print(f"Total Audio Files Found: {len(dataset)}")

        if len(dataset) == 0:
            print("[ALERT] Zero samples found! Check if .wav files exist in subdirectories.")
            return

        # 3. Test first sample extraction
        waveform, label, speaker_id, gender = dataset[0]

        print(f"--- First Sample Data ---")
        print(f"Speaker ID: {speaker_id}")
        print(f"Gender: {'Male (0)' if gender == 0 else 'Female (1)'}")
        print(f"Emotion Label: {label}")
        print(f"Waveform Shape: {waveform.shape}")

        # 4. Test Feature Extraction
        extractor = CNNSpectrogramExtractor()
        spectrogram = extractor(waveform)

        print(f"--- Feature Extraction ---")
        print(f"Spectrogram Shape (CNN Input): {spectrogram.shape}")

        print("\n[SUCCESS] Dataset pipeline is fully operational!")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline failed: {e}")

if __name__ == "main":
    test_dataset_structure()