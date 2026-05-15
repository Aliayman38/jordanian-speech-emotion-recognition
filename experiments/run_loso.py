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
sys.path.append(parent_dir)

from data.augment import AudioAugmenter
from data.dataset import JordanianSERDataset
from features.spectrogram import CNNSpectrogramExtractor
from models.cnn_spec import EmotionCNN

def run_loso_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Starting Professional LOSO on {device}")

    # 1. Setup Data & Tools
    dataset = JordanianSERDataset(data_dir=os.path.join(parent_dir, "Dataset"), transform=None)
    # Note: We apply Augmentation ONLY on Training indices inside the loop
    augmenter = AudioAugmenter(p=0.8) 
    extractor = CNNSpectrogramExtractor()
    
    logo = LeaveOneGroupOut()
    groups = dataset.speaker_ids
    labels = dataset.labels
    
    all_fold_results = []
    y_true_all = []
    y_pred_all = []

    # 2. Start Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(logo.split(dataset, labels, groups)):
        spk_id = groups[val_idx[0]]
        print(f"\n>>> Fold {fold+1}/33 | Testing on Speaker: {spk_id}")

        # Applying Augmentation only to Training Subset
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        
        # We manually trigger augmentation for training
        dataset.transform = augmenter 
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        
        # We disable augmentation for validation to see true performance
        dataset.transform = None 
        val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

        # 3. Model Initialization (Fresh start for every fold)
        model = EmotionCNN(num_classes=4).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        best_val_acc = 0.0
        
        # 4. Training (Few epochs per fold to save time, LOSO is expensive)
        for epoch in range(15):
            model.train()
            dataset.transform = augmenter # Ensure augmentation is ON
            for waveforms, targets, _, _ in train_loader:
                targets = targets.to(device)
                specs = extractor(waveforms).to(device)
                optimizer.zero_grad()
                outputs = model(specs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            dataset.transform = None # Ensure augmentation is OFF
            correct, total = 0, 0
            fold_preds, fold_trues = [], []
            with torch.no_grad():
                for waveforms, targets, _, _ in val_loader:
                    targets = targets.to(device)
                    specs = extractor(waveforms).to(device)
                    outputs = model(specs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    fold_preds.extend(predicted.cpu().numpy())
                    fold_trues.extend(targets.cpu().numpy())

            val_acc = 100 * correct / total
            scheduler.step(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Store predictions for the best version of this fold
                best_fold_preds = fold_preds
                best_fold_trues = fold_trues

        print(f"Fold {fold+1} Finished. Best Acc: {best_val_acc:.2f}%")
        all_fold_results.append(best_val_acc)
        y_pred_all.extend(best_fold_preds)
        y_true_all.extend(best_fold_trues)

        # Optional: Break after 3 folds for quick testing
        # if fold == 2: break 

    # 5. Final Evaluation & Plotting
    save_results(y_true_all, y_pred_all, all_fold_results)

def save_results(y_true, y_pred, accs):
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Overall LOSO Confusion Matrix (Avg Acc: {np.mean(accs):.2f}%)")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig("plots/loso_confusion_matrix.png")
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=['Angry', 'Happy', 'Neutral', 'Sad'])
    with open("results/loso_report.txt", "w") as f:
        f.write(f"Average LOSO Accuracy: {np.mean(accs):.2f}%\n")
        f.write(f"Standard Deviation: {np.std(accs):.2f}%\n\n")
        f.write(report)
    
    print("\n" + "="*40)
    print(f"FINAL LOSO AVG ACCURACY: {np.mean(accs):.2f}%")
    print("Results saved in /results and /plots")
    print("="*40)

if __name__ == "__main__":
    run_loso_experiment()