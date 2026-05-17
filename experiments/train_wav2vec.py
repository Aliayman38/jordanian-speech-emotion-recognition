# experiments/train_wav2vec.py
"""
End-to-end training script — Notebook 1 pipeline.

Steps
-----
1.  Load and split dataset (speaker-independent, 70/15/15)
2.  Fine-tune Arabic Wav2Vec2 with train-only augmentation
3.  Evaluate best checkpoint on val + test
4.  Extract fusion features (feat256, pooled, probs)
5.  Search fusion classifiers (validation-based)
6.  Final test evaluation of best fusion model
7.  Save training curves, confusion matrices, and summary CSVs

Run
---
    python experiments/train_wav2vec.py

Outputs written to ``outputs/<EXPERIMENT_NAME>/``
"""

import os
import sys
import pickle

# ── Make the project root importable regardless of cwd ───────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import (
    BASE_MODEL, BATCH_SIZE, DATASET_DIR, EPOCHS, EXP_OUT_DIR,
    LABEL_SMOOTHING, LR_BACKBONE, LR_HEAD, NUM_CLASSES, PATIENCE,
    SAMPLE_RATE, SEED, WEIGHT_DECAY, device, set_seed,
)
from dataset import build_metadata, speaker_independent_split, verify_no_overlap
from features.wav2vec import SERAudioDataset, collate_fn
from models.wav2vec_clf import Wav2VecEmotionModel
from features.extract_wav2vec import extract_finetuned_features
from models.fusion_v2 import build_fusion_sets, search_fusion
from evaluation.metrics import (
    evaluate_wav2vec_checkpoint, save_summary_csv, print_final_summary,
)
from evaluation.plots import (
    plot_confusion_matrix, plot_training_curves, plot_model_comparison,
)


# ── Setup ─────────────────────────────────────────────────────────────────────

set_seed(SEED)
os.makedirs(EXP_OUT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(EXP_OUT_DIR, "best_arabic_wav2vec2_ser_aug.pt")

print(f"[INFO] Device     : {device}")
print(f"[INFO] Base model : {BASE_MODEL}")
print(f"[INFO] Output dir : {EXP_OUT_DIR}")


# ── 1. Dataset ────────────────────────────────────────────────────────────────

metadata = build_metadata(DATASET_DIR)
train_df, val_df, test_df = speaker_independent_split(metadata)
verify_no_overlap(train_df, val_df, test_df)


# ── 2. Data loaders ───────────────────────────────────────────────────────────

feature_extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)
_collate = lambda b: collate_fn(b, feature_extractor)

train_ds = SERAudioDataset(train_df, augment=True)
val_ds   = SERAudioDataset(val_df,   augment=False)
test_ds  = SERAudioDataset(test_df,  augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=_collate, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=_collate, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=_collate, num_workers=0)


# ── 3. Model, loss, optimiser ─────────────────────────────────────────────────

model = Wav2VecEmotionModel(BASE_MODEL, NUM_CLASSES).to(device)

total_p     = sum(p.numel() for p in model.parameters())
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[Model] Total params     : {total_p:,}")
print(f"[Model] Trainable params : {trainable_p:,}")

class_counts  = train_df["label"].value_counts().sort_index().values
class_weights = len(train_df) / (NUM_CLASSES * class_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

optimizer = torch.optim.AdamW(
    [
        {"params": [p for n, p in model.named_parameters()
                    if "wav2vec" in n and p.requires_grad], "lr": LR_BACKBONE},
        {"params": [p for n, p in model.named_parameters()
                    if "classifier" in n and p.requires_grad], "lr": LR_HEAD},
    ],
    weight_decay=WEIGHT_DECAY,
)

num_training_steps = EPOCHS * len(train_loader)
num_warmup_steps   = int(0.1 * num_training_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


# ── 4. Training loop ──────────────────────────────────────────────────────────

def run_epoch(loader, train_mode: bool = True):
    model.train() if train_mode else model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for batch in tqdm(loader, leave=False):
            iv     = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            am     = batch["attention_mask"]
            if am is not None:
                am = am.to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(iv, am)
                loss   = criterion(logits, labels)

            if train_mode:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, macro_f1, np.array(all_labels), np.array(all_preds)


best_val_score = -1.0
best_epoch     = 0
bad_epochs     = 0
history        = []

print("=" * 80)
print("  Training Arabic Wav2Vec2 SER (train-only augmentation)")
print("=" * 80)

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_f1, _, _ = run_epoch(train_loader, train_mode=True)
    va_loss, va_acc, va_f1, _, _ = run_epoch(val_loader,   train_mode=False)

    val_score = va_f1 + 0.25 * va_acc

    history.append(dict(epoch=epoch,
                        train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
                        val_loss=va_loss,   val_acc=va_acc,   val_f1=va_f1,
                        val_score=val_score))

    if val_score > best_val_score:
        best_val_score = val_score
        best_epoch     = epoch
        bad_epochs     = 0
        torch.save(model.state_dict(), CKPT_PATH)
        tag = "  ✅ new best saved"
    else:
        bad_epochs += 1
        tag = f"  no improvement ({bad_epochs}/{PATIENCE})"

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss={tr_loss:.4f} Acc={tr_acc:.2%} F1={tr_f1:.4f} | "
        f"Val Loss={va_loss:.4f} Acc={va_acc:.2%} F1={va_f1:.4f}" + tag
    )

    if bad_epochs >= PATIENCE:
        print(f"\n[Early Stopping] Stopped at epoch {epoch}. Best = {best_epoch}")
        break

history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(EXP_OUT_DIR, "training_history.csv"), index=False)


# ── 5. Checkpoint evaluation ──────────────────────────────────────────────────

va_acc, va_f1, yva, pva, te_acc, te_f1, yte, pte = evaluate_wav2vec_checkpoint(
    model, val_loader, test_loader, run_epoch, best_epoch, CKPT_PATH
)

plot_confusion_matrix(
    yte, pte,
    title="Arabic Wav2Vec2 — Test Confusion Matrix",
    save_path=os.path.join(EXP_OUT_DIR, "cm_wav2vec2_aug.png"),
)

plot_training_curves(
    history_df, best_epoch,
    save_path=os.path.join(EXP_OUT_DIR, "training_curves.png"),
    title="Wav2Vec2 Fine-Tuning",
)


# ── 6. Fusion feature extraction ──────────────────────────────────────────────

model.eval()
y_tr, p_tr, f_tr, pool_tr = extract_finetuned_features(
    train_df, "Train", model, feature_extractor)
y_va, p_va, f_va, pool_va = extract_finetuned_features(
    val_df,   "Val",   model, feature_extractor)
y_te, p_te, f_te, pool_te = extract_finetuned_features(
    test_df,  "Test",  model, feature_extractor)


# ── 7. Fusion search ──────────────────────────────────────────────────────────

fusion_sets = build_fusion_sets(
    f_tr, p_tr, pool_tr,
    f_va, p_va, pool_va,
    f_te, p_te, pool_te,
)

best_fusion_clf, best_fusion_info, best_feature_key, fusion_history = (
    search_fusion(fusion_sets, y_tr, y_va)
)

fusion_history.to_csv(
    os.path.join(EXP_OUT_DIR, "fusion_search_history.csv"), index=False
)

# ── 8. Final test evaluation ──────────────────────────────────────────────────

Xtr_best, Xva_best, Xte_best = fusion_sets[best_feature_key]

fusion_val_pred  = best_fusion_clf.predict(Xva_best)
fusion_test_pred = best_fusion_clf.predict(Xte_best)

fusion_val_acc  = accuracy_score(y_va, fusion_val_pred)
fusion_test_acc = accuracy_score(y_te, fusion_test_pred)
fusion_val_f1   = f1_score(y_va, fusion_val_pred, average="macro",    zero_division=0)
fusion_test_f1  = f1_score(y_te, fusion_test_pred, average="macro",   zero_division=0)
fusion_test_wf1 = f1_score(y_te, fusion_test_pred, average="weighted",zero_division=0)

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION — Best Fusion Model")
print("=" * 70)
print(f"  Feature set      : {best_feature_key}")
print(f"  Classifier       : {best_fusion_info['candidate']}")
print(f"  Val Accuracy     : {fusion_val_acc:.2%}")
print(f"  Val Macro-F1     : {fusion_val_f1:.4f}")
print(f"  Test Accuracy    : {fusion_test_acc:.2%}")
print(f"  Test Macro-F1    : {fusion_test_f1:.4f}")
print(f"  Test Weighted-F1 : {fusion_test_wf1:.4f}")
print(f"  Baseline (prev)  : 67.36%")
print(f"  Delta            : {(fusion_test_acc - 0.6736):+.2%}")
from config import EMOTION_NAMES as _EN
print("\n" + classification_report(y_te, fusion_test_pred, target_names=_EN, digits=4, zero_division=0))

plot_confusion_matrix(
    y_te, fusion_test_pred,
    title="Fusion — Test Confusion Matrix",
    save_path=os.path.join(EXP_OUT_DIR, "cm_fusion_test.png"),
)

plot_model_comparison(
    model_names=["Baseline\n(67.36%)", "Wav2Vec2\n(this run)", "Fusion\n(this run)"],
    test_accs=[0.6736, te_acc, fusion_test_acc],
    save_path=os.path.join(EXP_OUT_DIR, "model_comparison.png"),
)

# Save fusion model
fusion_model_path = os.path.join(EXP_OUT_DIR, "best_fusion_classifier.pkl")
pickle.dump(
    dict(model=best_fusion_clf, feature_key=best_feature_key,
         best_info=best_fusion_info, base_model=BASE_MODEL),
    open(fusion_model_path, "wb"),
)

# Save summary
summary_rows = [
    dict(model="Arabic_Wav2Vec2_Finetuned_Aug",
         val_acc=va_acc, val_macro_f1=va_f1,
         test_acc=te_acc, test_macro_f1=te_f1,
         best_epoch=best_epoch, augmentation="train-only",
         speaker_independent=True, test_leakage=False),
    dict(model=f"Fusion | {best_fusion_info['candidate']}",
         val_acc=fusion_val_acc, val_macro_f1=fusion_val_f1,
         test_acc=fusion_test_acc, test_macro_f1=fusion_test_f1,
         best_epoch=best_epoch, augmentation="train-only",
         speaker_independent=True, test_leakage=False),
]
save_summary_csv(summary_rows, "final_summary.csv", save_dir=EXP_OUT_DIR)
print_final_summary(summary_rows)

print("\n✓ All outputs saved to:", EXP_OUT_DIR)
