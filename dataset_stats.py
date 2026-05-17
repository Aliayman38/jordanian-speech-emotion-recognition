# dataset_stats.py
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import DATASET_DIR, SEED
from dataset import build_metadata, speaker_independent_split

metadata = build_metadata(DATASET_DIR)
train_df, val_df, test_df = speaker_independent_split(metadata)

emotions = sorted(metadata["emotion"].unique())
rows = []

for emotion in emotions:
    tr = (train_df["emotion"] == emotion).sum()
    va = (val_df["emotion"]   == emotion).sum()
    te = (test_df["emotion"]  == emotion).sum()
    rows.append((emotion, tr, va, te, tr + va + te))

# ── Print table ───────────────────────────────────────────────
header = f"{'Emotion':<12} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}"
sep    = "-" * len(header)

print("\n  TABLE — Dataset Distribution")
print(f"  {sep}")
print(f"  {header}")
print(f"  {sep}")
for emotion, tr, va, te, tot in rows:
    print(f"  {emotion:<12} {tr:>6} {va:>6} {te:>6} {tot:>7}")
print(f"  {sep}")

total_tr  = sum(r[1] for r in rows)
total_va  = sum(r[2] for r in rows)
total_te  = sum(r[3] for r in rows)
total_all = total_tr + total_va + total_te
print(f"  {'Total':<12} {total_tr:>6} {total_va:>6} {total_te:>6} {total_all:>7}")
print(f"  {sep}\n")
