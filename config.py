# config.py
"""
Central configuration — all hyperparameters and paths live here.
Every other module imports from this file; nothing is hard-coded elsewhere.
"""

import os
import random

import numpy as np
import torch


# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Paths ─────────────────────────────────────────────────────────────────────
# Point DATASET_DIR at the folder that contains 'male/' and 'female/'
DATASET_DIR = "Dataset"
SAVE_DIR    = "outputs"                       # local; override for Colab Drive

# Wav2Vec2 fine-tuning experiment writes here so the baseline is never
# accidentally overwritten.
EXPERIMENT_NAME = "wav2vec2_aug_fusion_final"
EXP_OUT_DIR     = os.path.join(SAVE_DIR, EXPERIMENT_NAME)

# ── Emotion classes ───────────────────────────────────────────────────────────
EMOTIONS    = ["Angry", "Happy", "Neutral", "Sad"]
EMOTION2ID  = {e: i for i, e in enumerate(EMOTIONS)}
ID2EMOTION  = {i: e for e, i in EMOTION2ID.items()}
NUM_CLASSES = len(EMOTIONS)
EMOTION_NAMES = [ID2EMOTION[i] for i in range(NUM_CLASSES)]

EMOTION_ALIASES = {
    "angry": "Angry", "anger": "Angry",
    "happy": "Happy", "happiness": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",    "sadness": "Sad",
}

# ── Audio ─────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16_000
MAX_SECONDS  = 4.0
MAX_LEN      = int(SAMPLE_RATE * MAX_SECONDS)

# MFCC / Mel (classical models & CNN)
N_MFCC       = 40
N_MELS       = 64
N_FFT        = 2048
HOP_LENGTH   = 512
MAX_DURATION = 5.0                               # seconds for CNN padding
MAX_FRAMES   = int(MAX_DURATION * SAMPLE_RATE / HOP_LENGTH) + 1  # ≈ 157

# ── Wav2Vec2 fine-tuning ──────────────────────────────────────────────────────
BASE_MODEL      = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
# Fallback if GPU OOM:  BASE_MODEL = "facebook/wav2vec2-base"

BATCH_SIZE      = 4        # reduce to 2 on small GPUs
EPOCHS          = 25
PATIENCE        = 5        # early-stopping patience (val-based)
LR_BACKBONE     = 1e-5     # unfrozen Wav2Vec2 layers
LR_HEAD         = 8e-4     # classification head
WEIGHT_DECAY    = 1e-2
UNFREEZE_LAST_N = 2        # how many encoder layers to unfreeze
LABEL_SMOOTHING = 0.08

# Train-only augmentation probabilities
AUG_NOISE_PROB  = 0.45    # Gaussian noise
AUG_GAIN_PROB   = 0.35    # volume gain
AUG_SHIFT_PROB  = 0.35    # time shift

# ── Classical / CNN ───────────────────────────────────────────────────────────
CLF_BATCH_SIZE  = 32
CNN_EPOCHS      = 100
MLP_EPOCHS      = 100
CLF_PATIENCE    = 15
LR              = 1e-3

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
