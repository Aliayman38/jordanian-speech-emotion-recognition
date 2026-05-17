# experiments/train_classical.py
"""
End-to-end training script — Notebook 2 pipeline.

Steps
-----
1.  Load and split dataset (speaker-independent, 70/15/15)
2.  Extract MFCC features (with disk cache)
3.  Train and search classical models: SVM, KNN, MLP, ExtraTrees
4.  Build soft-voting ensemble from best per-model classifiers
5.  Train CNN on Mel-spectrograms
6.  Extract frozen Wav2Vec2 embeddings (with disk cache)
7.  Fusion v1: MFCC + Wav2Vec2 embeddings classifier search
8.  Save all confusion matrices, training curves, and summary CSVs

Run
---
    python experiments/train_classical.py

Outputs written to ``outputs/``
"""

import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, CLF_BATCH_SIZE, CLF_PATIENCE, CNN_EPOCHS,
    DATASET_DIR, LR, NUM_CLASSES, SAVE_DIR, SEED, device, set_seed,
)
from dataset import build_metadata, speaker_independent_split, verify_no_overlap
from features.mfcc import extract_features_for_split
from features.spectrogram import MelDataset
from features.extract_wav2vec import extract_pretrained_embeddings
from models.classical import (
    build_svm_candidates, build_knn_candidates,
    build_mlp_candidates, build_extra_candidates,
    quick_search, build_soft_ensemble,
    MLP as TorchMLP, train_pytorch_mlp,
)
from models.cnn_spec import EmotionCNN, run_epoch as cnn_run_epoch
from models.fusion_v1 import build_fusion_features, run_fusion_search
from evaluation.metrics import evaluate_sklearn, save_summary_csv, print_final_summary
from evaluation.plots import (
    plot_confusion_matrix, plot_training_curves, plot_model_comparison,
)


# ── Setup ─────────────────────────────────────────────────────────────────────

set_seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "features"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "checkpoints"), exist_ok=True)

print(f"[INFO] Device    : {device}")
print(f"[INFO] Output    : {SAVE_DIR}")


# ── 1. Dataset ────────────────────────────────────────────────────────────────

metadata = build_metadata(DATASET_DIR)
train_df, val_df, test_df = speaker_independent_split(metadata)
verify_no_overlap(train_df, val_df, test_df)

y_train = train_df["label"].values
y_val   = val_df["label"].values
y_test  = test_df["label"].values


# ── 2. MFCC features (cached) ─────────────────────────────────────────────────

MFCC_CACHE = os.path.join(SAVE_DIR, "features", "mfcc_features.npz")

if os.path.exists(MFCC_CACHE):
    print("[INFO] Loading cached MFCC features...")
    d = np.load(MFCC_CACHE)
    X_train_mfcc, X_val_mfcc, X_test_mfcc = d["X_train"], d["X_val"], d["X_test"]
else:
    X_train_mfcc, _ = extract_features_for_split(train_df, "Train MFCC")
    X_val_mfcc,   _ = extract_features_for_split(val_df,   "Val MFCC")
    X_test_mfcc,  _ = extract_features_for_split(test_df,  "Test MFCC")
    np.savez(MFCC_CACHE,
             X_train=X_train_mfcc, X_val=X_val_mfcc, X_test=X_test_mfcc)
    print(f"  Saved MFCC cache → {MFCC_CACHE}")

# Scale on train only (no leakage)
mfcc_scaler  = StandardScaler()
X_train_sc   = mfcc_scaler.fit_transform(X_train_mfcc)
X_val_sc     = mfcc_scaler.transform(X_val_mfcc)
X_test_sc    = mfcc_scaler.transform(X_test_mfcc)
N_FEATURES   = X_train_sc.shape[1]
print(f"  MFCC feature dim: {N_FEATURES}")


# ── 3. Classical models ───────────────────────────────────────────────────────

results  = {}
summaries = []

# 3a) SVM
svm_candidates = build_svm_candidates(N_FEATURES, use_scaler=False)
svm_model, svm_info, svm_hist = quick_search(
    "SVM", svm_candidates, X_train_sc, y_train, X_val_sc, y_val
)
svm_res = evaluate_sklearn(
    svm_model, X_train_sc, y_train, X_val_sc, y_val, X_test_sc, y_test,
    "Best SVM (MFCC)", "svm_best",
)
plot_confusion_matrix(
    y_test, svm_res["test_pred"], "SVM — Test Confusion Matrix",
    save_path=os.path.join(SAVE_DIR, "cm_svm.png"),
)
pickle.dump(svm_model, open(os.path.join(SAVE_DIR, "checkpoints", "svm_model.pkl"), "wb"))
svm_hist.to_csv(os.path.join(SAVE_DIR, "svm_search.csv"), index=False)
results["svm"] = (svm_model, svm_info["val_acc"])
summaries.append(dict(model="SVM", **{k: svm_res[k] for k in
                       ["val_acc", "test_acc", "test_macro_f1"]}))

# 3b) KNN
knn_candidates = build_knn_candidates(N_FEATURES, use_scaler=False)
knn_model, knn_info, knn_hist = quick_search(
    "KNN", knn_candidates, X_train_sc, y_train, X_val_sc, y_val
)
knn_res = evaluate_sklearn(
    knn_model, X_train_sc, y_train, X_val_sc, y_val, X_test_sc, y_test,
    "Best KNN (MFCC)", "knn_best",
)
plot_confusion_matrix(
    y_test, knn_res["test_pred"], "KNN — Test Confusion Matrix",
    save_path=os.path.join(SAVE_DIR, "cm_knn.png"),
)
pickle.dump(knn_model, open(os.path.join(SAVE_DIR, "checkpoints", "knn_model.pkl"), "wb"))
results["knn"] = (knn_model, knn_info["val_acc"])
summaries.append(dict(model="KNN", **{k: knn_res[k] for k in
                       ["val_acc", "test_acc", "test_macro_f1"]}))

# 3c) MLP (sklearn)
mlp_candidates = build_mlp_candidates(N_FEATURES, use_scaler=False)
mlp_model, mlp_info, mlp_hist = quick_search(
    "MLP", mlp_candidates, X_train_sc, y_train, X_val_sc, y_val
)
mlp_res = evaluate_sklearn(
    mlp_model, X_train_sc, y_train, X_val_sc, y_val, X_test_sc, y_test,
    "Best MLP (MFCC)", "mlp_best",
)
plot_confusion_matrix(
    y_test, mlp_res["test_pred"], "MLP — Test Confusion Matrix",
    save_path=os.path.join(SAVE_DIR, "cm_mlp.png"),
)
pickle.dump(mlp_model, open(os.path.join(SAVE_DIR, "checkpoints", "mlp_model.pkl"), "wb"))
results["mlp"] = (mlp_model, mlp_info["val_acc"])
summaries.append(dict(model="MLP", **{k: mlp_res[k] for k in
                       ["val_acc", "test_acc", "test_macro_f1"]}))

# 3d) Extra baselines (LogReg + ExtraTrees)
extra_candidates = build_extra_candidates(N_FEATURES, use_scaler=False)
extra_model, extra_info, extra_hist = quick_search(
    "Extra", extra_candidates, X_train_sc, y_train, X_val_sc, y_val
)
extra_res = evaluate_sklearn(
    extra_model, X_train_sc, y_train, X_val_sc, y_val, X_test_sc, y_test,
    "Best Extra (MFCC)", "extra_best",
)
plot_confusion_matrix(
    y_test, extra_res["test_pred"], "Extra — Test Confusion Matrix",
    save_path=os.path.join(SAVE_DIR, "cm_extra.png"),
)
pickle.dump(extra_model, open(os.path.join(SAVE_DIR, "checkpoints", "extra_model.pkl"), "wb"))
results["extra"] = (extra_model, extra_info["val_acc"])
summaries.append(dict(model="Extra (LogReg/ExtraTrees)", **{k: extra_res[k] for k in
                       ["val_acc", "test_acc", "test_macro_f1"]}))


# ── 4. Soft-voting ensemble ───────────────────────────────────────────────────

# Only include models that support predict_proba
proba_results = {k: v for k, v in results.items()
                 if k in ("svm", "mlp", "extra")}

ensemble = build_soft_ensemble(proba_results, X_train_sc, y_train)
ens_res = evaluate_sklearn(
    ensemble, X_train_sc, y_train, X_val_sc, y_val, X_test_sc, y_test,
    "Soft Voting Ensemble", "ensemble",
)
plot_confusion_matrix(
    y_test, ens_res["test_pred"], "Ensemble — Test Confusion Matrix",
    save_path=os.path.join(SAVE_DIR, "cm_ensemble.png"),
)
pickle.dump(ensemble, open(os.path.join(SAVE_DIR, "checkpoints", "best_ensemble.pkl"), "wb"))
summaries.append(dict(model="Soft Ensemble", **{k: ens_res[k] for k in
                       ["val_acc", "test_acc", "test_macro_f1"]}))


# ── 5. CNN on Mel-spectrograms ────────────────────────────────────────────────

train_mel_ds = MelDataset(train_df)
val_mel_ds   = MelDataset(val_df)
test_mel_ds  = MelDataset(test_df)

train_mel_loader = DataLoader(train_mel_ds, batch_size=CLF_BATCH_SIZE,
                               shuffle=True,  num_workers=2, pin_memory=True)
val_mel_loader   = DataLoader(val_mel_ds,   batch_size=CLF_BATCH_SIZE,
                               shuffle=False, num_workers=2, pin_memory=True)
test_mel_loader  = DataLoader(test_mel_ds,  batch_size=CLF_BATCH_SIZE,
                               shuffle=False, num_workers=2, pin_memory=True)

cnn       = EmotionCNN(NUM_CLASSES).to(device)
crit_cnn  = nn.CrossEntropyLoss()
opt_cnn   = optim.AdamW(cnn.parameters(), lr=LR, weight_decay=1e-2)
sched_cnn = optim.lr_scheduler.CosineAnnealingLR(opt_cnn, T_max=CNN_EPOCHS)

cnn_hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss, patience_cnt, cnn_best_ep = float("inf"), 0, 1
best_cnn_state = None

print("\n[CNN] Training on Mel-spectrograms...")
for ep in range(CNN_EPOCHS):
    tl, ta = cnn_run_epoch(cnn, train_mel_loader, crit_cnn, opt_cnn, device)
    vl, va = cnn_run_epoch(cnn, val_mel_loader,   crit_cnn, None,    device)
    sched_cnn.step()

    cnn_hist["train_loss"].append(tl); cnn_hist["train_acc"].append(ta)
    cnn_hist["val_loss"].append(vl);   cnn_hist["val_acc"].append(va)

    if vl < best_val_loss:
        best_val_loss = vl; patience_cnt = 0
        best_cnn_state = {k: v.clone() for k, v in cnn.state_dict().items()}
        cnn_best_ep = ep + 1
    else:
        patience_cnt += 1

    if (ep + 1) % 10 == 0 or ep == 0:
        print(f"  Ep {ep+1:3d} | Loss {tl:.4f} Acc {ta:.2%} | "
              f"ValLoss {vl:.4f} ValAcc {va:.2%}")

    if patience_cnt >= CLF_PATIENCE:
        print(f"  Early stopping at epoch {ep + 1} (best={cnn_best_ep})")
        break

cnn.load_state_dict(best_cnn_state)
torch.save(cnn.state_dict(), os.path.join(SAVE_DIR, "checkpoints", "cnn_model.pt"))

# Evaluate CNN on test
cnn.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_mel_loader:
        out = cnn(xb.to(device))
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(yb.numpy())

cnn_test_acc = accuracy_score(all_labels, all_preds)
cnn_test_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
print(f"\n[CNN] Test Accuracy: {cnn_test_acc:.2%}  Macro-F1: {cnn_test_f1:.4f}")
plot_confusion_matrix(
    all_labels, all_preds, "CNN (Mel-spectrogram) — Test Confusion Matrix",
    save_path=os.path.join(SAVE_DIR, "cm_cnn.png"),
)

cnn_hist_df = pd.DataFrame({
    "epoch":      range(1, len(cnn_hist["train_loss"]) + 1),
    "train_loss": cnn_hist["train_loss"], "val_loss": cnn_hist["val_loss"],
    "train_acc":  cnn_hist["train_acc"],  "val_acc":  cnn_hist["val_acc"],
})
# Re-use val_acc as val_f1 placeholder for the curve plotter
cnn_hist_df["val_f1"] = cnn_hist_df["val_acc"]
plot_training_curves(
    cnn_hist_df, cnn_best_ep,
    save_path=os.path.join(SAVE_DIR, "cnn_training_curves.png"),
    title="CNN (Mel-spectrogram)",
)
summaries.append(dict(model="CNN (Mel-spectrogram)",
                      val_acc=max(cnn_hist["val_acc"]),
                      test_acc=cnn_test_acc,
                      test_macro_f1=cnn_test_f1))


# ── 6. Wav2Vec2 embeddings (frozen, cached) ───────────────────────────────────

w2v_data = extract_pretrained_embeddings(
    train_df, val_df, test_df,
    model_name="facebook/wav2vec2-base",
)

print("\n[Wav2Vec2 pretrained] shapes:")
print(f"  Train: {w2v_data['X_train'].shape}")
print(f"  Val  : {w2v_data['X_val'].shape}")
print(f"  Test : {w2v_data['X_test'].shape}")


# ── 7. Fusion v1: MFCC + Wav2Vec2 ────────────────────────────────────────────

fusion_sets = build_fusion_features(
    X_train_sc, X_val_sc, X_test_sc, w2v_data
)

best_fusion_model = None
best_fusion_info  = None
best_fusion_key   = None
all_fusion_hists  = []

for feat_key, (Xtr, Xva, Xte) in fusion_sets.items():
    clf, info, hist = run_fusion_search(
        Xtr, Xva, Xte, y_train, y_val, feat_key
    )
    all_fusion_hists.append(hist)
    if best_fusion_info is None or info["score"] > best_fusion_info["score"]:
        best_fusion_info  = info
        best_fusion_model = clf
        best_fusion_key   = feat_key

fusion_history = (
    pd.concat(all_fusion_hists, ignore_index=True)
      .sort_values("score", ascending=False)
)
fusion_history.to_csv(os.path.join(SAVE_DIR, "fusion_v1_search.csv"), index=False)

_, _, Xte_best = fusion_sets[best_fusion_key]
fusion_test_pred = best_fusion_model.predict(Xte_best)
fusion_test_acc  = accuracy_score(y_test, fusion_test_pred)
fusion_test_f1   = f1_score(y_test, fusion_test_pred, average="macro", zero_division=0)
fusion_val_acc   = best_fusion_info["val_acc"]

print(f"\n[Fusion v1] Test Accuracy: {fusion_test_acc:.2%}  Macro-F1: {fusion_test_f1:.4f}")
plot_confusion_matrix(
    y_test, fusion_test_pred, "Fusion v1 — Test Confusion Matrix",
    save_path=os.path.join(SAVE_DIR, "cm_fusion_v1.png"),
)
pickle.dump(best_fusion_model,
            open(os.path.join(SAVE_DIR, "checkpoints", "fusion_v1_model.pkl"), "wb"))
summaries.append(dict(model=f"Fusion v1 ({best_fusion_key})",
                      val_acc=fusion_val_acc,
                      test_acc=fusion_test_acc,
                      test_macro_f1=fusion_test_f1))


# ── 8. Final summary ──────────────────────────────────────────────────────────

save_summary_csv(summaries, "classical_summary.csv", save_dir=SAVE_DIR)
print_final_summary(summaries)

model_names = [r["model"] for r in summaries]
test_accs   = [r["test_acc"] for r in summaries]
plot_model_comparison(
    model_names, test_accs,
    save_path=os.path.join(SAVE_DIR, "classical_model_comparison.png"),
    title="Classical + CNN + Fusion — Test Accuracy",
)

print("\n✓ All outputs saved to:", SAVE_DIR)
