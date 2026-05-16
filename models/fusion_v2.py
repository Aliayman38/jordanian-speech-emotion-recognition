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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from experiments.clusterer import get_stratified_speakers

METADATA_CSV_PATH = os.path.join(parent_dir, "data", "metadata.csv")
MFCC_CSV_PATH = os.path.join(parent_dir, "outputs", "features", "mfcc_features_loso.csv")
WAV2VEC_NPY_PATH = os.path.join(parent_dir, "outputs", "features", "wav2vec_features.npy")
CNN_NPY_PATH = os.path.join(parent_dir, "outputs", "features", "cnn_features.npy")
LABELS_PATH = os.path.join(parent_dir, "outputs", "features", "labels.npy")
SPEAKERS_PATH = os.path.join(parent_dir, "outputs", "features", "speakers.npy")

OUTPUT_DIR = os.path.join(parent_dir, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# ==========================================
# 2. DATA LOADING & STRATIFIED SPLIT
# ==========================================
def clean_speaker_id(spk_id):
    try:
        return int(str(spk_id).split('_')[-1])
    except ValueError:
        return hash(spk_id)

def load_and_split_triple_data():
    print("[SYSTEM] Loading 3 Modalities and applying 70/15/15 Clusterer Split...")
    
    df_mfcc = pd.read_csv(MFCC_CSV_PATH)
    mfcc_features = df_mfcc.drop(columns=['rel_path', 'label', 'speaker_id', 'gender']).values
    wav2vec_features = np.load(WAV2VEC_NPY_PATH)
    cnn_features = np.load(CNN_NPY_PATH)
    labels = np.load(LABELS_PATH)
    speakers = np.load(SPEAKERS_PATH)
    
    # Expected dimensions: 258 + 1024 + 128 = 1410
    fused_features = np.concatenate((mfcc_features, wav2vec_features, cnn_features), axis=1)
    
    train_spks_raw, val_spks_raw, test_spks_raw = get_stratified_speakers(METADATA_CSV_PATH)
    
    train_spks = [clean_speaker_id(s) for s in train_spks_raw]
    val_spks = [clean_speaker_id(s) for s in val_spks_raw]
    test_spks = [clean_speaker_id(s) for s in test_spks_raw]
    
    train_idx = np.isin(speakers, train_spks)
    val_idx = np.isin(speakers, val_spks)
    test_idx = np.isin(speakers, test_spks)
    
    X_train, y_train = fused_features[train_idx], labels[train_idx]
    X_val, y_val = fused_features[val_idx], labels[val_idx]
    X_test, y_test = fused_features[test_idx], labels[test_idx]
    
    print(f"[INFO] Modalities Merged. Total Features: {X_train.shape[1]}")
    print(f"[INFO] Train Samples: {X_train.shape[0]} | Val Samples: {X_val.shape[0]} | Test Samples: {X_test.shape[0]}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ==========================================
# 3. OPTIMIZED NEURAL NETWORK ARCHITECTURE
# ==========================================
class OptimizedFusionNet(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(OptimizedFusionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# ==========================================
# 4. TRAINING & EVALUATION PIPELINE
# ==========================================
def train_triple_fusion():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split_triple_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[SYSTEM] Training Triple Fusion Model on {device}...")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    BATCH_SIZE = 16
    MAX_EPOCHS = 40
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=BATCH_SIZE, shuffle=False)
    
    model = OptimizedFusionNet(input_dim=X_train.shape[1], num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    
    best_val_acc = 0.0
    model_path = os.path.join(CHECKPOINTS_DIR, "best_fusion_v3.pth")
    
    t_losses, v_losses, t_accs, v_accs = [], [], [], []
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        run_loss, run_corr, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            run_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            run_corr += (pred == labels).sum().item()
            
        t_losses.append(run_loss / len(train_loader))
        t_accs.append((run_corr / total) * 100)
        
        model.eval()
        v_loss, v_corr, v_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                v_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                v_total += labels.size(0)
                v_corr += (pred == labels).sum().item()
                
        v_acc = (v_corr / v_total) * 100
        v_losses.append(v_loss / len(val_loader))
        v_accs.append(v_acc)
        scheduler.step()
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), model_path)
            
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{MAX_EPOCHS}] | Train Acc: {t_accs[-1]:.2f}% | Val Acc: {v_accs[-1]:.2f}%")
            
    print("\n[SYSTEM] Evaluating Best Triple Fusion Model on Held-Out Test Set...")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(pred.cpu().numpy())
            
    generate_reports(y_true, y_pred, t_losses, v_losses, t_accs, v_accs, "v3_triple_fusion")

def generate_reports(y_true, y_pred, tl, vl, ta, va, name):
    classes = ['Angry', 'Happy', 'Neutral', 'Sad']
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"Triple Fusion ({name}) - Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(FIGURES_DIR, f"{name}_cm.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    epochs = range(1, len(ta) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, tl, 'b-', label='Train')
    plt.plot(epochs, vl, 'r-', label='Val')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, ta, 'b-', label='Train')
    plt.plot(epochs, va, 'r-', label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, f"{name}_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    acc = accuracy_score(y_true, y_pred) * 100
    report = classification_report(y_true, y_pred, target_names=classes)
    with open(os.path.join(LOGS_DIR, f"{name}_report.txt"), "w") as f:
        f.write(f"TEST ACCURACY: {acc:.2f}%\n\n{report}")
        
    print("="*50)
    print(f"FINAL TEST ACCURACY: {acc:.2f}%")
    print("="*50)

if __name__ == "__main__":
    train_triple_fusion()