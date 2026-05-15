import torch
import os
import time
from pathlib import Path
from data.dataset import JordanianSERDataset
from data.augment import AudioAugmenter

# Configuration: Update this path to your local dataset folder
DATASET_ROOT = "Dataset"

def run_pipeline_validation():
    print("--- Starting Pipeline Validation ---")
    
    # 1. Directory Integrity Check
    data_path = Path(DATASET_ROOT).resolve()
    if not data_path.exists():
        print(f"[ERROR] Directory not found at: {data_path}")
        print("[ACTION] Please ensure the dataset is downloaded and path is correct.")
        return

    # 2. Augmentation Module Initialization
    print("\n[STEP 1] Initializing AudioAugmenter...")
    try:
        # Set p=1.0 to force augmentation for testing purposes
        augmenter = AudioAugmenter(noise_factor=0.005, shift_limit=0.1, p=1.0)
        print("[SUCCESS] Augmenter initialized with forced probability (p=1.0).")
    except Exception as e:
        print(f"[FAILED] Augmenter initialization failed: {e}")
        return

    # 3. Dataset & Metadata Caching Test
    print("\n[STEP 2] Initializing JordanianSERDataset...")
    start_time = time.time()
    try:
        dataset = JordanianSERDataset(data_dir=data_path, transform=augmenter)
        load_duration = time.time() - start_time
        
        print(f"[SUCCESS] Dataset loaded {len(dataset)} samples.")
        print(f"[PERFORMANCE] Initialization time: {load_duration:.4f} seconds.")
        
        # Verify metadata file creation
        metadata_file = data_path / "metadata.csv"
        if metadata_file.exists():
            print(f"[VERIFIED] Metadata cache found at: {metadata_file}")
        else:
            print("[WARNING] Metadata file was not created. Check write permissions.")
            
    except Exception as e:
        print(f"[FAILED] Dataset initialization failed: {e}")
        return

    # 4. Data Retrieval & Transformation Test
    print("\n[STEP 3] Testing Data Retrieval (Item 0)...")
    try:
        # Fetching the first sample
        waveform, label, speaker_id, gender = dataset[0]
        
        print(f"  - Waveform Tensor Shape: {waveform.shape}")
        print(f"  - Label Index: {label.item()} (Tensor Type: {label.dtype})")
        print(f"  - Speaker ID: {speaker_id}")
        print(f"  - Gender ID: {gender} (0:Male, 1:Female)")
        
        # Validation of tensor dimensions (Expected: [Channels, Time])
        if waveform.ndim == 2:
            print("[VERIFIED] Waveform dimensions are correct.")
        else:
            print(f"[WARNING] Unexpected waveform dimensions: {waveform.shape}")

    except Exception as e:
        print(f"[FAILED] Data retrieval failed: {e}")
        return

    print("\n" + "="*40)
    print("FINAL STATUS: PIPELINE READY FOR PRODUCTION")
    print("="*40)

if __name__ == "__main__":
    run_pipeline_validation()