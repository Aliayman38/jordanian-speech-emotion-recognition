import os
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from evaluation.plots import plot_confusion_matrix
from evaluation.metrics import save_classification_report

class Wav2VecSVMClassifier:
    def __init__(self, kernel='rbf', C=1.0):
        self.scaler = StandardScaler()
        self.svm = SVC(kernel=kernel, C=C, class_weight='balanced', probability=True, random_state=42)

    def train_and_evaluate_loso(self, X_features, y_labels, groups_speakers):
        logo = LeaveOneGroupOut()
        
        y_true_all = []
        y_pred_all = []
        accuracies = []

        print("[SYSTEM] Starting Wav2Vec2 + SVM (v1) with LOSO Validation...")

        for fold, (train_idx, val_idx) in enumerate(logo.split(X_features, y_labels, groups_speakers)):
            X_train, X_val = X_features[train_idx], X_features[val_idx]
            y_train, y_val = y_labels[train_idx], y_labels[val_idx]
            spk_id = groups_speakers[val_idx[0]]

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            self.svm.fit(X_train_scaled, y_train)

            y_pred = self.svm.predict(X_val_scaled)
            
            y_true_all.extend(y_val)
            y_pred_all.extend(y_pred)
            
            fold_acc = np.mean(y_pred == y_val) * 100
            accuracies.append(fold_acc)
            
            print(f"Fold {fold+1:02d} (Speaker {spk_id}) | Accuracy: {fold_acc:.2f}%")

        print("\n[SYSTEM] Generating Final Wav2Vec2+SVM Reports...")
        
        plot_confusion_matrix(
            y_true_all, 
            y_pred_all, 
            save_path=os.path.join(parent_dir, "outputs", "figures", "wav2vec_svm_confusion_matrix.png")
        )
        
        save_classification_report(
            y_true_all, 
            y_pred_all, 
            save_path=os.path.join(parent_dir, "outputs", "logs", "wav2vec_svm_report.txt")
        )
        
        print("\n" + "="*50)
        print(f"FINAL WAV2VEC+SVM AVG ACCURACY: {np.mean(accuracies):.2f}%")
        print("="*50)

def run_wav2vec_svm_experiment():
    print("[INFO] Loading Wav2Vec2 Features...")
    
    try:
        # feature_path = os.path.join(parent_dir, "outputs", "features", "wav2vec_features.npy")
        # labels_path = os.path.join(parent_dir, "outputs", "features", "labels.npy")
        # speakers_path = os.path.join(parent_dir, "outputs", "features", "speakers.npy")
        # 
        # X_wav2vec = np.load(feature_path)
        # labels = np.load(labels_path)
        # speakers = np.load(speakers_path)
        
        print("[!] Running with simulated Wav2Vec2 dummy data for structural testing.")
        
        num_samples = 500
        feature_dim = 768 
        X_wav2vec = np.random.rand(num_samples, feature_dim)
        labels = np.random.randint(0, 4, num_samples)
        speakers = np.random.randint(1, 10, num_samples)

        classifier = Wav2VecSVMClassifier(kernel='rbf', C=10)
        classifier.train_and_evaluate_loso(X_wav2vec, labels, speakers)

    except FileNotFoundError as e:
        print(f"[ERROR] Feature files not found! {e}")

if __name__ == "__main__":
    run_wav2vec_svm_experiment()
