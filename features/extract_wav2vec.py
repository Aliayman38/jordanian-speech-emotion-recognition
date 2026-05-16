import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data.dataset import JordanianSERDataset

def clean_speaker_id(spk_id):
    try:
        return int(str(spk_id).split('_')[-1])
    except ValueError:
        return hash(spk_id)

def extract_wav2vec_features():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Hardware Accelerator: {device}")

    # 1. Load Arabic Wav2Vec2 Model
    model_name = "facebook/wav2vec2-large-xlsr-53"
    print(f"[INFO] Loading Model: {model_name}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device)
    model.eval()

    DATASET_PATH = os.path.join(parent_dir, "Dataset") 
    dataset = JordanianSERDataset(data_dir=DATASET_PATH, transform=None)
    
    all_features, all_labels, all_speakers = [], [], []

    print(f"[INFO] Extracting features with MEAN POOLING for {len(dataset)} samples...")
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            waveform, label, speaker_id, _ = dataset[i]
            
            waveform = waveform.squeeze().numpy()
            
            inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
            
            outputs = model(input_values)
            
        
            hidden_states = outputs.last_hidden_state
            feature_vector = torch.mean(hidden_states, dim=1).squeeze().cpu().numpy()
            
            all_features.append(feature_vector)
            all_labels.append(label)
            all_speakers.append(clean_speaker_id(speaker_id))

    # 3. Save Files
    OUTPUT_DIR = os.path.join(parent_dir, "outputs", "features")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    np.save(os.path.join(OUTPUT_DIR, "wav2vec_features2.npy"), np.array(all_features))
    np.save(os.path.join(OUTPUT_DIR, "labels2.npy"), np.array(all_labels))
    np.save(os.path.join(OUTPUT_DIR, "speakers2.npy"), np.array(all_speakers))

    print("\n[SUCCESS] Extraction Complete!")
    print(f"[INFO] CORRECTED Wav2Vec2 Shape: {np.array(all_features).shape} (Must be N, 1024)")
    print(f"[INFO] Saved securely to: {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_wav2vec_features()