import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
MFCC_CSV_PATH = r"C:\Users\Ali83\jordanian-speech-emotion-recognition\outputs\features\mfcc_features_loso.csv"
WAV2VEC_NPY_PATH = r"C:\Users\Ali83\jordanian-speech-emotion-recognition\outputs\features\wav2vec_features.npy"
LABELS_PATH = r"C:\Users\Ali83\jordanian-speech-emotion-recognition\outputs\features\labels.npy"
SPEAKERS_PATH = r"C:\Users\Ali83\jordanian-speech-emotion-recognition\outputs\features\speakers.npy"

OUTPUT_DIR = r"C:\Users\Ali83\jordanian-speech-emotion-recognition\outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ==========================================
# 2. FEATURE FUSION
# ==========================================
def load_and_fuse_features():
    print("[SYSTEM] Loading and Fusing MFCC & Wav2Vec2 Features...")
    
    # Load MFCC
    df_mfcc = pd.read_csv(MFCC_CSV_PATH)
    mfcc_features = df_mfcc.drop(columns=['rel_path', 'label', 'speaker_id', 'gender']).values
    
    # Load Wav2Vec2, Labels, and Speakers
    wav2vec_features = np.load(WAV2VEC_NPY_PATH)
    labels = np.load(LABELS_PATH)
    speakers = np.load(SPEAKERS_PATH)
    
    if mfcc_features.shape[0] != wav2vec_features.shape[0]:
        raise ValueError("[ERROR] Dimension mismatch between MFCC and Wav2Vec2 features!")
        
    fused_features = np.concatenate((mfcc_features, wav2vec_features), axis=1)
    print(f"[SUCCESS] Fused Shape (Samples, Features): {fused_features.shape}")
    
    return fused_features, labels, speakers

# ==========================================
# 3. NEURAL NETWORK ARCHITECTURE
# ==========================================
class CombinedEmotionNet(nn.Module):
    def __init__(self, input_dim=1282, num_classes=4):
        super(CombinedEmotionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 4. LOSO TRAINING PIPELINE
# ==========================================
def run_loso_combined_model():
    X, y, groups = load_and_fuse_features()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Hardware Accelerator: {device}")
    print("[SYSTEM] Starting Deep Learning LOSO Validation...")
    print("[INFO] Applying manual early stopping at epoch 50 for optimal generalization.\n")
    
    logo = LeaveOneGroupOut()
    
    y_true_all = []
    y_pred_all = []
    fold_accuracies = []
    
    # Hyperparameters
    MAX_EPOCHS = 50 # Manual early stopping applied here
    BATCH_SIZE = 16
    
    for fold, (train_val_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        test_speaker = groups[test_idx[0]]
        
        # 1. Split data for the current fold
        X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # 2. Create a small Validation set from the training data for early stopping tracking
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
        )
        
        # 3. Scale Features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        # 4. Prepare DataLoaders
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=BATCH_SIZE, shuffle=False)
        
        # 5. Initialize Model, Optimizer, and Scheduler newly for EACH fold
        model = CombinedEmotionNet(input_dim=X.shape[1], num_classes=4).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
        
        best_val_acc = 0.0
        fold_model_path = os.path.join(CHECKPOINTS_DIR, f"best_model_fold_{fold+1}.pth")
        
        # --- EPOCH LOOP ---
        for epoch in range(MAX_EPOCHS):
            # Train
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            # Validate
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = (val_correct / val_total) * 100
            scheduler.step()
            
            # Save best model for this fold
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), fold_model_path)
                
        # --- TEST ON LEFT-OUT SPEAKER ---
        model.load_state_dict(torch.load(fold_model_path, weights_only=True))
        model.eval()
        
        test_correct, test_total = 0, 0
        fold_y_true, fold_y_pred = [], []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                fold_y_true.extend(labels.cpu().numpy())
                fold_y_pred.extend(predicted.cpu().numpy())
                
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
        test_acc = (test_correct / test_total) * 100
        fold_accuracies.append(test_acc)
        
        y_true_all.extend(fold_y_true)
        y_pred_all.extend(fold_y_pred)
        
        print(f"Fold {fold+1:02d} | Left-Out Speaker: {test_speaker:<3} | Test Accuracy: {test_acc:.2f}%")

    # ==========================================
    # 5. FINAL REPORTING
    # ==========================================
    print("\n[SYSTEM] LOSO Validation Complete. Generating Final Reports...")
    classes = ['Angry', 'Happy', 'Neutral', 'Sad']
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title("Combined Model (MFCC + Wav2Vec2) LOSO - Confusion Matrix")
    plt.ylabel('Actual Emotion')
    plt.xlabel('Predicted Emotion')
    plt.savefig(os.path.join(FIGURES_DIR, "combined_loso_cm.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Classification Report
    final_acc = np.mean(fold_accuracies)
    report = classification_report(y_true_all, y_pred_all, target_names=classes)
    
    with open(os.path.join(LOGS_DIR, "combined_loso_report.txt"), "w") as f:
        f.write("="*50 + "\n")
        f.write("  COMBINED MODEL (MFCC + WAV2VEC2) LOSO REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Average LOSO Accuracy: {final_acc:.2f}%\n\n")
        f.write("Detailed Metrics:\n")
        f.write("-" * 50 + "\n")
        f.write(report)
        
    print("="*50)
    print(f"FINAL COMBINED MODEL LOSO ACCURACY: {final_acc:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_loso_combined_model()