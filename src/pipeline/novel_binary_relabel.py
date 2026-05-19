"""
L1** Novel-Binary Detection: Relabel training data to neutralize the
attack entry process (Firefox) while keeping children labeled as attack.

This tests: "Can the model detect a malicious dropper (Firefox) from
behavioral patterns learned from known-malicious children (cache, clean, xtmp)?"

Adds `label_l1` column to labeled shard Parquets.

Usage:
    python -m src.pipeline.novel_binary_relabel --dataset theia
    python -m src.pipeline.novel_binary_relabel --dataset theia --dry-run
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ground_truth import load_ground_truth


# ============================================================
# Dataset paths
# ============================================================
DATASET_ROOTS = {
    "theia": Path("data/processed/darpa_tc_e3/theia"),
    "trace": Path("data/processed/darpa_tc_e3/trace"),
}

TRAIN_SHARDS = {
    "theia": list(range(7)),   # shards 0-6
    "trace": list(range(5)),   # shards 0-4
}

TEST_SHARDS = {
    "theia": [8, 9],
    "trace": [6],
}


def get_basename(path_str):
    """Extract basename from a process path."""
    if not path_str or pd.isna(path_str):
        return ""
    return str(path_str).rstrip("/").rsplit("/", 1)[-1].lower()


def main():
    parser = argparse.ArgumentParser(
        description="L1** Novel-Binary Detection: relabel for Firefox neutralization")
    parser.add_argument("--dataset", default="theia",
                        choices=list(DATASET_ROOTS.keys()))
    parser.add_argument("--dry-run", action="store_true",
                        help="Only analyze, don't modify files")
    parser.add_argument("--entry-process", default=None,
                        help="Override entry process name (default: from ground truth)")
    args = parser.parse_args()

    data_root = DATASET_ROOTS[args.dataset]
    labeled_dir = data_root / "labeled"
    train_shards = TRAIN_SHARDS[args.dataset]
    test_shards = TEST_SHARDS[args.dataset]

    gt = load_ground_truth(args.dataset)
    entry_process = args.entry_process or gt.attack_entry_process

    print("=" * 60)
    print(f"L1** NOVEL-BINARY RELABEL: {args.dataset.upper()}")
    print("=" * 60)
    print(f"  Entry process to neutralize: {entry_process}")
    print(f"  Train shards: {train_shards}")
    print(f"  Test shards:  {test_shards}")

    # ========================================================
    # Step 1: Load subject catalog & build basename map
    # ========================================================
    print("\n[Step 1] Loading subjects...")
    subj_df = pd.read_parquet(data_root / "subjects.parquet")
    subj_df["basename"] = subj_df["process_path"].fillna("").apply(get_basename)
    if subj_df["basename"].eq("").all() and "cmd_line" in subj_df.columns:
        # TRACE uses cmd_line
        subj_df["basename"] = subj_df["cmd_line"].fillna("").apply(
            lambda x: get_basename(x.split()[0]) if x.strip() else "")

    uuid_to_basename = dict(zip(subj_df["uuid"], subj_df["basename"]))

    # Extract basename from entry process (could be full path or just name)
    entry_basename = get_basename(entry_process)
    print(f"  Entry process basename: '{entry_basename}'")

    # Identify all UUIDs matching the entry process basename
    entry_uuids = set(
        subj_df[subj_df["basename"] == entry_basename]["uuid"])
    print(f"  Total subjects: {len(subj_df):,}")
    print(f"  '{entry_basename}' instances (all shards): {len(entry_uuids):,}")

    del subj_df
    gc.collect()

    # ========================================================
    # Step 2: Analyze overlap
    # ========================================================
    print("\n[Step 2] Analyzing attack structure...")

    # Test attack subjects by basename
    test_attack_basenames = set()
    test_attack_subjects = set()
    for sid in test_shards:
        df = pd.read_parquet(
            labeled_dir / f"labeled_shard{sid}.parquet",
            columns=["subject_uuid", "label_broad"])
        atk_uuids = df[df["label_broad"] == 1]["subject_uuid"].unique()
        test_attack_subjects.update(atk_uuids)
        for u in atk_uuids:
            bn = uuid_to_basename.get(u, "")
            if bn:
                test_attack_basenames.add(bn)
        del df

    print(f"  Test attack subjects: {len(test_attack_subjects):,}")
    print(f"  Test attack basenames: {test_attack_basenames}")

    # ========================================================
    # Step 3: Compute L1** labels (neutralize entry process only)
    # ========================================================
    print(f"\n[Step 3] Computing L1** labels (neutralize '{entry_process}')...")
    t0 = time.time()

    stats = {
        "total_train_events": 0,
        "original_attack": 0,
        "neutralized": 0,
        "remaining_attack": 0,
    }

    all_shards = sorted(labeled_dir.glob("labeled_shard*.parquet"))
    shard_indices = set()
    for f in all_shards:
        idx = int(f.stem.replace("labeled_shard", ""))
        shard_indices.add(idx)

    for sid in sorted(shard_indices):
        shard_path = labeled_dir / f"labeled_shard{sid}.parquet"
        df = pd.read_parquet(shard_path)

        is_train = sid in train_shards

        # Start from broad labels
        l1_labels = df["label_broad"].copy()

        if is_train:
            # In training: neutralize events from entry process
            is_entry = df["subject_uuid"].isin(entry_uuids)
            is_attack = l1_labels == 1
            neutralize_mask = is_entry & is_attack

            n_neutralized = int(neutralize_mask.sum())
            l1_labels[neutralize_mask] = 0

            stats["total_train_events"] += len(df)
            stats["original_attack"] += int(is_attack.sum())
            stats["neutralized"] += n_neutralized
            stats["remaining_attack"] += int((l1_labels == 1).sum())

            n_atk = int((l1_labels == 1).sum())
            tag = "TRAIN"
        else:
            # Test/val: keep original broad labels
            n_atk = int((l1_labels == 1).sum())
            n_neutralized = 0
            tag = "TEST "

        print(f"    Shard {sid:2d} [{tag}]: "
              f"{n_atk:>8,} attack / {len(df):>10,} total "
              f"({100*n_atk/len(df):.2f}%) "
              f"[neutralized: {n_neutralized:,}]")

        if not args.dry_run:
            df["label_l1"] = l1_labels.astype(np.int8)
            df.to_parquet(shard_path, index=False)

        del df, l1_labels
        gc.collect()

    elapsed = time.time() - t0

    # ========================================================
    # Summary
    # ========================================================
    print(f"\n{'='*60}")
    print(f"L1** SUMMARY")
    print(f"{'='*60}")
    print(f"  Entry process neutralized: {entry_process}")
    print(f"  Total train events:        {stats['total_train_events']:,}")
    print(f"  Original train attack:     {stats['original_attack']:,} "
          f"({100*stats['original_attack']/stats['total_train_events']:.2f}%)")
    print(f"  Neutralized (→ benign):    {stats['neutralized']:,}")
    print(f"  Remaining train attack:    {stats['remaining_attack']:,} "
          f"({100*stats['remaining_attack']/stats['total_train_events']:.2f}%)")
    print(f"  Time: {elapsed:.1f}s")

    if args.dry_run:
        print(f"\n  ⚠ DRY RUN — no files modified")
    else:
        print(f"\n  ✓ Added 'label_l1' column to all labeled shards")
        print(f"  → Use label_type: l1 in config YAML to train")

    # Quick sanity check
    if stats["remaining_attack"] == 0:
        print(f"\n  ⚠ WARNING: No attack events remaining in training!")
        print(f"    L1** is not viable for this dataset.")
    elif stats["remaining_attack"] / stats["total_train_events"] < 0.005:
        print(f"\n  ⚠ CAUTION: Very low attack rate ({100*stats['remaining_attack']/stats['total_train_events']:.3f}%).")
        print(f"    Consider increasing pos_weight in training config.")


if __name__ == "__main__":
    main()
