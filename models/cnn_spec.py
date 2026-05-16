import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data.augment import AudioAugmenter
from data.dataset import JordanianSERDataset
from features.spectrogram import CNNSpectrogramExtractor

from evaluation.plots import plot_confusion_matrix
from evaluation.metrics import save_classification_report

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(EmotionCNN, self).__init__()
        
        # Spatial Feature Extractor (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        
        # Temporal Dynamics (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=1024, 
            hidden_size=64, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0.5
        )
        
        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(128, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x) 
        
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2).contiguous() 
        x = x.view(batch_size, time, channels * freq) 
        
        lstm_out, _ = self.lstm(x) 
        x = lstm_out[:, -1, :] 

        # Dynamic Reshaping Safety
        if x.shape[1] != 128:
            dynamic_pool = nn.Linear(x.shape[1], 128).to(x.device)
            x = dynamic_pool(x)

        x = self.fc(x)
        return x


def run_loso_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Starting Professional LOSO on {device}")

    dataset = JordanianSERDataset(data_dir=os.path.join(parent_dir, "Dataset"), transform=None)
    augmenter = AudioAugmenter(p=0.8) 
    extractor = CNNSpectrogramExtractor()
    
    logo = LeaveOneGroupOut()
    groups = dataset.speaker_ids
    labels = dataset.labels
    
    all_fold_results = []
    y_true_all = []
    y_pred_all = []

    for fold, (train_idx, val_idx) in enumerate(logo.split(dataset, labels, groups)):
        spk_id = groups[val_idx[0]]
        print(f"\n>>> Fold {fold+1}/33 | Testing on Speaker: {spk_id}")

        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        
        # Training Phase
        dataset.transform = augmenter 
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        
        # Validation Phase
        dataset.transform = None 
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

        model = EmotionCNN(num_classes=4).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        best_val_acc = 0.0
        best_fold_preds, best_fold_trues = [], []
        
        for epoch in range(15):
            model.train()
            dataset.transform = augmenter
            for waveforms, targets, _, _ in train_loader:
                targets = targets.to(device)
                specs = extractor(waveforms).to(device)
                optimizer.zero_grad()
                outputs = model(specs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            model.eval()
            dataset.transform = None
            correct, total = 0, 0
            temp_preds, temp_trues = [], []
            with torch.no_grad():
                for waveforms, targets, _, _ in val_loader:
                    targets = targets.to(device)
                    specs = extractor(waveforms).to(device)
                    outputs = model(specs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    temp_preds.extend(predicted.cpu().numpy())
                    temp_trues.extend(targets.cpu().numpy())

            val_acc = 100 * correct / total
            scheduler.step(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_fold_preds = temp_preds
                best_fold_trues = temp_trues

        print(f"Fold {fold+1} Finished. Best Acc: {best_val_acc:.2f}%")
        all_fold_results.append(best_val_acc)
        y_pred_all.extend(best_fold_preds)
        y_true_all.extend(best_fold_trues)

    save_results(y_true_all, y_pred_all, all_fold_results)

def save_results(y_true, y_pred, accs):
    import numpy as np
    
    print("\n[SYSTEM] Generating final evaluation reports...")
    
    # 1. Generate and save Plots
    plot_confusion_matrix(y_true, y_pred)
    
    # 2. Generate and save Metrics Report
    save_classification_report(y_true, y_pred)
    
    print("\n" + "="*40)
    print(f"FINAL MODEL AVG ACCURACY: {np.mean(accs):.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_loso_experiment()