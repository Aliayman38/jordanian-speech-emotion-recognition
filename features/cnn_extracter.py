import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

DATASET_PATH = os.path.join(parent_dir, "Dataset")
CNN_WEIGHTS_PATH = os.path.join(parent_dir, "checkpoints", "best_cnn_model.pth")
OUTPUT_DIR = os.path.join(parent_dir, "outputs", "features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import your Dataset and CNN Model structures
from data.dataset import JordanianSERDataset
from features.spectrogram import CNNSpectrogramExtractor

# IMPORTANT: You must import your exact CNN class name from your models folder
from models.cnn_spec import EmotionCNN  # <--- REPLACE 'YourCNNClassName' WITH YOUR ACTUAL CLASS NAME

# ==========================================
# 2. FEATURE EXTRACTION PIPELINE
# ==========================================
def extract_cnn_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Extracting CNN Features using {device}...")

    # 1. Load the Dataset
    # Assuming your CNN uses spectrograms, make sure your dataset class handles the correct transform
    spec_transform = CNNSpectrogramExtractor()
    dataset = JordanianSERDataset(data_dir=DATASET_PATH, transform=spec_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 2. Load the Pre-trained CNN Model
    print(f"[INFO] Loading trained CNN weights from {CNN_WEIGHTS_PATH}")
    model = EmotionCNN(num_classes=4) # Initialize your model
    
    try:
        model.load_state_dict(torch.load(CNN_WEIGHTS_PATH, map_location=device))
    except FileNotFoundError:
        print(f"[ERROR] CNN weights not found at {CNN_WEIGHTS_PATH}. Did you train the CNN first?")
        sys.exit(1)

    # 3. Modify the Model to act as a Feature Extractor
    # NOTE: Replace 'fc' with the actual name of your final classification layer in your CNN class.
    # Often it is named 'fc', 'classifier', or 'out'. This replaces it with an Identity layer 
    # so the model outputs the raw feature vector instead of 4 classes.
    model.fc = nn.Identity() 
    
    model.to(device)
    model.eval()

    all_features = []
    
    print(f"[INFO] Processing {len(dataset)} samples to extract deep features...")
    
    # 4. Extract Features
    with torch.no_grad():
        for inputs, labels, speaker_ids, genders in tqdm(dataloader):
            inputs = inputs.to(device)
            
            # Forward pass: since we removed the last layer, this outputs the feature vector
            features = model(inputs)
            
            # Move to CPU and convert to numpy array
            all_features.append(features.cpu().numpy())
            
    # Concatenate all batches
    final_features = np.vstack(all_features)
    
    # 5. Save the Features
    save_path = os.path.join(OUTPUT_DIR, "cnn_features.npy")
    np.save(save_path, final_features)
    
    print(f"\n[SUCCESS] CNN Features Extracted!")
    print(f"[INFO] Final Feature Matrix Shape: {final_features.shape}")
    print(f"[INFO] Saved securely to: {save_path}")

if __name__ == "__main__":
    extract_cnn_features()