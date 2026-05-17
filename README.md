# Jordanian Arabic Speech Emotion Recognition

Speaker-independent SER for Jordanian Arabic dialect using a fine-tuned
Arabic Wav2Vec2 backbone, classical MFCC-based models, and two fusion strategies.

## Results

| Model | Test Acc | Macro-F1 |
|---|---|---|
| Baseline (previous) | 67.36% | — |
| SVM (MFCC) | — | — |
| CNN (Mel-spectrogram) | — | — |
| Wav2Vec2 fine-tuned | — | — |
| **Fusion v2 (best)** | **—** | **—** |

> Fill in results after running the experiments.

---

## Project Structure

```
.
├── config.py                   # All hyperparameters and paths (edit here)
├── dataset.py                  # Metadata scanning, speaker-independent split
├── requirements.txt
│
├── features/
│   ├── mfcc.py                 # MFCC extraction, audio loading, augmentation
│   ├── spectrogram.py          # MelDataset (for CNN)
│   ├── wav2vec.py              # SERAudioDataset + collate_fn (for fine-tuning)
│   └── extract_wav2vec.py      # Embedding extraction (frozen & fine-tuned)
│
├── models/
│   ├── wav2vec_clf.py          # Wav2VecEmotionModel architecture
│   ├── cnn_spec.py             # EmotionCNN on Mel-spectrograms
│   ├── classical.py            # SVM / KNN / MLP / ExtraTrees + quick_search
│   ├── fusion_v1.py            # MFCC + frozen Wav2Vec2 embeddings fusion
│   └── fusion_v2.py            # Fine-tuned Wav2Vec2 features + probs fusion
│
├── evaluation/
│   ├── metrics.py              # Accuracy, F1, classification reports
│   └── plots.py                # Confusion matrices, training curves, bar charts
│
├── experiments/
│   ├── train_wav2vec.py        # Full Notebook 1 pipeline (fine-tune + fusion v2)
│   └── train_classical.py      # Full Notebook 2 pipeline (classical + CNN + fusion v1)
│
├── Dataset/                    # Raw audio (not committed)
│   ├── female/
│   ├── male/
│   └── metadata.csv
│
└── outputs/                    # Generated artefacts (not committed)
    ├── checkpoints/            # .pt and .pkl model files
    └── features/               # Cached .npz feature arrays
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Configuration

Edit **`config.py`** before running anything:

```python
DATASET_DIR = "/path/to/Dataset_WAV_last/Dataset_WAV_last"
SAVE_DIR    = "outputs"   # or "/content/drive/MyDrive/SER_Results" on Colab
```

---

## Running Experiments

### Pipeline 1 — Arabic Wav2Vec2 fine-tuning + Fusion v2

```bash
python experiments/train_wav2vec.py
```

What it does:
1. Speaker-independent split (70 / 15 / 15)
2. Fine-tunes `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` with train-only augmentation
3. Extracts intermediate features (feat256, pooled, probs) from the best checkpoint
4. Searches fusion classifiers (LogReg, SVM, MLP) — selects by validation only
5. Evaluates the best fusion model on the test set **once**

### Pipeline 2 — Classical models + CNN + Fusion v1

```bash
python experiments/train_classical.py
```

What it does:
1. Speaker-independent split (70 / 15 / 15)
2. Extracts MFCC features (cached to `outputs/features/mfcc_features.npz`)
3. Searches SVM, KNN, MLP (sklearn), and ExtraTrees — validation-based selection
4. Builds a soft-voting ensemble from the best per-model classifiers
5. Trains a lightweight 2-D CNN on Mel-spectrograms
6. Extracts frozen `facebook/wav2vec2-base` embeddings (cached)
7. Searches fusion classifiers over MFCC + Wav2Vec2 feature combinations

---

## Key Design Principles

- **Speaker-independent** — no speaker appears in more than one split.
- **No test leakage** — all model selection is based on the validation set only.
  The test set is evaluated exactly once, at the very end of each pipeline.
- **Train-only augmentation** — Gaussian noise, volume gain, and time shift are
  applied exclusively to training samples; validation and test data are always
  loaded clean.
- **Reproducible** — all random seeds are fixed via `config.SEED`.

---

## Dataset Structure

```
Dataset/
    male/
        Speaker_01/
            Angry/*.wav
            Happy/*.wav
            Neutral/*.wav
            Sad/*.wav
    female/
        Speaker_23/
            ...
```

Emotion folder names are normalised automatically
(`anger` → `Angry`, `happiness` → `Happy`, etc.).
