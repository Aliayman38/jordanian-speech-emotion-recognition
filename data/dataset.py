import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path

class JordanianSERDataset(Dataset):
    def __init__(self, data_dir, transform=None, metadata_file="metadata.csv"):
        self.data_dir = Path(data_dir).resolve()
        self.transform = transform
        self.metadata_path = self.data_dir / metadata_file
        self.emotions = {'Happy': 0, 'Sad': 1, 'Angry': 2, 'Neutral': 3}
        
        # Consistent variable naming
        self.file_rel_paths = []
        self.labels = []
        self.speaker_ids = []
        self.genders = []

        if self.metadata_path.exists():
            print(f"[INFO] Loading relative metadata from {self.metadata_path}...")
            self._load_metadata()
        else:
            print("[INFO] Metadata not found. Parsing directory structure...")
            self._parse_dataset()
            self._save_metadata()

    def _parse_dataset(self):
        for gender_str in ['Male', 'Female']:
            gender_path = self.data_dir / gender_str
            if not gender_path.exists(): continue
            
            gender_label = 0 if gender_str == 'Male' else 1
            
            for speaker_folder in os.listdir(gender_path):
                speaker_path = gender_path / speaker_folder
                if not speaker_path.is_dir(): continue
                
                try:
                    speaker_id = int(speaker_folder.split('_')[1])
                except (IndexError, ValueError): continue
                
                for emotion_folder in os.listdir(speaker_path):
                    emotion_path = speaker_path / emotion_folder
                    if emotion_folder not in self.emotions: continue
                    
                    label = self.emotions[emotion_folder]
                    
                    for audio_file in os.listdir(emotion_path):
                        if audio_file.endswith('.wav') and not audio_file.startswith('.'):
                            full_path = emotion_path / audio_file
                            # Correct logic for relative pathing
                            rel_path = full_path.relative_to(self.data_dir)
                            self.file_rel_paths.append(str(rel_path))
                            self.labels.append(label)
                            self.speaker_ids.append(speaker_id)
                            self.genders.append(gender_label)

    def _save_metadata(self):
        df = pd.DataFrame({
            'rel_path': self.file_rel_paths,
            'label': self.labels,
            'speaker_id': self.speaker_ids,
            'gender': self.genders
        })
        df.to_csv(self.metadata_path, index=False)
        print(f"[SUCCESS] Metadata saved to {self.metadata_path}")

    def _load_metadata(self):
        # Loading must use the same variable name as defined in __init__
        df = pd.read_csv(self.metadata_path)
        self.file_rel_paths = df['rel_path'].tolist()
        self.labels = df['label'].tolist()
        self.speaker_ids = df['speaker_id'].tolist()
        self.genders = df['gender'].tolist()

    def __len__(self):
        return len(self.file_rel_paths)

    def __getitem__(self, idx):
        # Using the standardized variable name
        rel_path = self.file_rel_paths[idx]
        full_path = self.data_dir / rel_path
        
        waveform, _ = torchaudio.load(str(full_path))
        
        # Standardization to 4 seconds (64000 samples)
        target_length = 64000 
        current_length = waveform.shape[1]
        
        if current_length > target_length:
            waveform = waveform[:, :target_length]
        elif current_length < target_length:
            padding = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        if self.transform:
            waveform = self.transform(waveform)
            
        return (
            waveform, 
            torch.tensor(self.labels[idx], dtype=torch.long), 
            self.speaker_ids[idx], 
            self.genders[idx]
        )