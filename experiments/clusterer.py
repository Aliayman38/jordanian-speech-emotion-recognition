"""
experiments/clusterer.py — Speaker-independent splits with robust stratification.

Handles edge cases where some emotion classes have too few speakers
for sklearn's StratifiedShuffleSplit.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter

RANDOM_STATE = 42

def get_stratified_speakers(metadata_csv: Path):
    """
    Return three disjoint speaker-id sets for train/val/test.

    Strategy:
        1. Group speakers by dominant emotion label
        2. Allocate proportionally from each group to maintain label balance
        3. If a group has <2 speakers, merge it with similar groups

    Target ratios: ~70% train / 15% val / 15% test (speaker-level)
    """
    meta = pd.read_csv(metadata_csv)

    # Per-speaker dominant label
    speaker_dominant = {}
    speaker_samples = {}
    for spk_id, group in meta.groupby("speaker_id"):
        labels = group["label"].values
        dominant = Counter(labels).most_common(1)[0][0]
        speaker_dominant[spk_id] = dominant
        speaker_samples[spk_id] = len(labels)

    all_speakers = list(speaker_dominant.keys())
    n_total = len(all_speakers)

    # Target counts
    n_train = max(1, int(round(n_total * 0.70)))
    n_val = max(1, int(round(n_total * 0.15)))
    n_test = max(1, n_total - n_train - n_val)

    # Adjust for rounding
    while n_train + n_val + n_test > n_total:
        if n_train > n_val and n_train > n_test:
            n_train -= 1
        elif n_val > n_test:
            n_val -= 1
        else:
            n_test -= 1
    while n_train + n_val + n_test < n_total:
        n_train += 1

    # Group by dominant label
    label_groups = {}
    for spk, lbl in speaker_dominant.items():
        label_groups.setdefault(lbl, []).append(spk)

    np.random.seed(RANDOM_STATE)

    # Shuffle within each group
    for lbl in label_groups:
        spks = label_groups[lbl]
        np.random.shuffle(spks)
        label_groups[lbl] = spks

    # Proportional allocation
    train_speakers, val_speakers, test_speakers = [], [], []

    for lbl, spks in sorted(label_groups.items()):
        n_spks = len(spks)
        # Allocate ~70/15/15 within this label group
        n_tr = max(0, int(round(n_spks * 0.70)))
        n_va = max(0, int(round(n_spks * 0.15)))
        n_te = n_spks - n_tr - n_va

        # Ensure at least 1 per split if group is large enough
        if n_spks >= 3:
            if n_tr == 0: n_tr = 1; n_te -= 1
            if n_va == 0: n_va = 1; n_te -= 1
            if n_te == 0: n_te = 1; n_tr -= 1
            # Rebalance if negative
            if n_tr < 1: n_tr = 1; n_va = max(1, n_spks - n_tr - n_te)
            if n_va < 1: n_va = 1; n_tr = max(1, n_spks - n_va - n_te)
            if n_te < 1: n_te = 1; n_tr = max(1, n_spks - n_va - n_te)
        elif n_spks == 2:
            # 1 to train, 1 to val (test gets from another group)
            n_tr, n_va, n_te = 1, 1, 0
        else:
            # Single speaker — give to train
            n_tr, n_va, n_te = 1, 0, 0

        train_speakers.extend(spks[:n_tr])
        val_speakers.extend(spks[n_tr:n_tr + n_va])
        test_speakers.extend(spks[n_tr + n_va:])

    # --- Balance to exact counts ---
    def balance(target, others, target_count):
        while len(target) > target_count and others:
            other_lens = [len(o) for o in others]
            min_idx = other_lens.index(min(other_lens))
            others[min_idx].append(target.pop())
        while len(target) < target_count and others:
            other_lens = [len(o) for o in others]
            max_idx = other_lens.index(max(other_lens))
            if len(others[max_idx]) > 1:
                target.append(others[max_idx].pop())
            else:
                break
        return target, others

    all_lists = [train_speakers, val_speakers, test_speakers]
    targets = [n_train, n_val, n_test]

    for _ in range(10):
        changed = False
        for i in range(3):
            others = [all_lists[j] for j in range(3) if j != i]
            old_len = len(all_lists[i])
            all_lists[i], others = balance(all_lists[i], others, targets[i])
            idx = 0
            for j in range(3):
                if j != i:
                    all_lists[j] = others[idx]
                    idx += 1
            if len(all_lists[i]) != old_len:
                changed = True
        if not changed:
            break

    train_speakers, val_speakers, test_speakers = [sorted(s) for s in all_lists]

    # Verify
    assert len(set(train_speakers) & set(val_speakers)) == 0, "Overlap!"
    assert len(set(train_speakers) & set(test_speakers)) == 0, "Overlap!"
    assert len(set(val_speakers) & set(test_speakers)) == 0, "Overlap!"
    all_assigned = set(train_speakers) | set(val_speakers) | set(test_speakers)
    assert all_assigned == set(all_speakers), f"Missing: {set(all_speakers) - all_assigned}"

    return train_speakers, val_speakers, test_speakers


get_speaker_splits = get_stratified_speakers


if __name__ == "__main__":
    import sys
    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent
    candidates = [project_root / "data" / "metadata.csv", Path("data/metadata.csv"), Path("../data/metadata.csv")]
    meta_path = None
    for c in candidates:
        if c.exists():
            meta_path = c
            break
    if meta_path is None and len(sys.argv) > 1:
        meta_path = Path(sys.argv[1])
    if meta_path is None:
        print("Usage: python -m experiments.clusterer <metadata.csv>")
        sys.exit(1)

    print(f"Using: {meta_path}")
    train, val, test = get_stratified_speakers(meta_path)
    meta = pd.read_csv(meta_path)
    n_total = meta["speaker_id"].nunique()

    print(f"\nTotal: {n_total} speakers")
    for name, spks in [("Train", train), ("Val", val), ("Test", test)]:
        samples = len(meta[meta["speaker_id"].isin(spks)])
        print(f"{name:6}: {len(spks)} speakers ({len(spks)/n_total*100:.1f}%) → {samples} samples")
    print(f"\nTrain: {train}")
    print(f"Val:   {val}")
    print(f"Test:  {test}")