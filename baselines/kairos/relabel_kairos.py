"""
Re-extract KAIROS event labels using a different label column.

Replays the exact same filtering and ordering as data_converter.py
(event type filter → UUID mapping → timestamp sort) but reads a
different label column (e.g., label_l1 for L1** experiment).

This avoids re-running the expensive data converter. The KAIROS
graph structure and embeddings are unchanged — only labels differ.

Usage:
    python -m baselines.kairos.relabel_kairos --dataset theia --label-type l1
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baselines.kairos.config import (
    DATASET_CONFIGS, GRAPHS_DIR,
)


def relabel_dataset(dataset: str, label_type: str):
    """Re-extract labels for KAIROS events using a different label column."""

    cfg = DATASET_CONFIGS[dataset]
    data_dir = cfg["data_dir"]
    labeled_dir = data_dir / "labeled"
    include_types = set(cfg["include_edge_type"])
    graphs_dir = GRAPHS_DIR / dataset

    label_col = f"label_{label_type}"

    print("=" * 60)
    print(f"KAIROS RELABEL: {dataset.upper()} → {label_col}")
    print("=" * 60)

    # Load UUID→ID mapping from the saved graph metadata
    # We need to know which UUIDs were mapped during conversion
    # Rebuild from subjects/objects (same as data_converter Step 1)
    print("\n  Loading entity vocabulary...")
    t0 = time.time()

    subjects_df = pd.read_parquet(data_dir / "subjects.parquet", columns=["uuid"])
    objects_df = pd.read_parquet(data_dir / "objects.parquet", columns=["uuid"])

    uuid_to_id = {}
    next_id = 0
    for uid in subjects_df["uuid"].values:
        if uid not in uuid_to_id:
            uuid_to_id[uid] = next_id
            next_id += 1
    for uid in objects_df["uuid"].values:
        if uid not in uuid_to_id:
            uuid_to_id[uid] = next_id
            next_id += 1

    print(f"  Entities: {len(uuid_to_id):,} ({time.time()-t0:.1f}s)")

    del subjects_df, objects_df
    gc.collect()

    # Determine shard groups
    shard_groups = {}
    for idx in cfg["train_shards"]:
        shard_groups[idx] = "train"
    for idx in cfg["val_shards"]:
        shard_groups[idx] = "val"
    for idx in cfg["test_shards"]:
        shard_groups[idx] = "test"

    # For each split, replay the exact same filtering as data_converter
    for split in ["train", "val", "test"]:
        split_indices = sorted([i for i, g in shard_groups.items() if g == split])
        if not split_indices:
            continue

        # Check if TemporalData exists (we need it for the sort order)
        td_path = graphs_dir / f"{split}.TemporalData.pt"
        if not td_path.exists():
            print(f"\n  {split}: TemporalData not found, skipping")
            continue

        print(f"\n  {split} (shards {split_indices}):")

        all_labels = []
        all_timestamps = []

        for shard_idx in split_indices:
            shard_path = labeled_dir / f"labeled_shard{shard_idx}.parquet"
            if not shard_path.exists():
                print(f"    WARNING: {shard_path} not found")
                continue

            # Read same columns as data_converter + the new label column
            cols = ["type", "timestamp_nanos",
                    "subject_uuid", "predicate_object_uuid",
                    label_col]

            try:
                df = pd.read_parquet(shard_path, columns=cols)
            except Exception as e:
                print(f"    ERROR reading {label_col} from shard {shard_idx}: {e}")
                print(f"    Available columns: {pd.read_parquet(shard_path, columns=[]).columns.tolist()}")
                return

            # Same filtering as data_converter
            mask = df["type"].isin(include_types)
            df = df[mask].reset_index(drop=True)

            src_ids = df["subject_uuid"].map(uuid_to_id)
            dst_ids = df["predicate_object_uuid"].map(uuid_to_id)
            valid = src_ids.notna() & dst_ids.notna()
            df = df[valid].reset_index(drop=True)

            labels = df[label_col].values.astype(np.int8)
            timestamps = df["timestamp_nanos"].values.astype(np.int64)

            all_labels.append(labels)
            all_timestamps.append(timestamps)

            n_atk = int((labels == 1).sum())
            print(f"    Shard {shard_idx}: {len(labels):,} events, "
                  f"{n_atk:,} attack ({100*n_atk/len(labels):.1f}%)")

            del df
            gc.collect()

        if not all_labels:
            continue

        labels_concat = np.concatenate(all_labels)
        timestamps_concat = np.concatenate(all_timestamps)

        # Apply same timestamp sort as data_converter
        sort_idx = np.argsort(timestamps_concat)
        labels_sorted = labels_concat[sort_idx]

        # Verify alignment with existing broad labels
        broad_path = graphs_dir / f"{split}_labels.npy"
        if broad_path.exists():
            broad_labels = np.load(broad_path)
            if len(broad_labels) != len(labels_sorted):
                print(f"    ⚠ LENGTH MISMATCH: broad={len(broad_labels):,}, "
                      f"{label_col}={len(labels_sorted):,}")
                print(f"    This means the parquet data has changed. Re-run data_converter.")
                return
            # On test/val, labels should be identical to broad
            if split in ("val", "test"):
                diff = (broad_labels != labels_sorted).sum()
                print(f"    Alignment check ({split}): {diff:,} differences "
                      f"({'OK — same as broad' if diff == 0 else 'EXPECTED — L1 modifies this split too' if diff > 0 else ''})")

        # Save
        out_path = graphs_dir / f"{split}_labels_{label_type}.npy"
        np.save(out_path, labels_sorted)

        n_atk = int(labels_sorted.sum())
        print(f"    → Saved {out_path.name}: {len(labels_sorted):,} events, "
              f"{n_atk:,} attack ({100*n_atk/len(labels_sorted):.1f}%)")

    print(f"\n✓ Done. Use --train-labels {label_type} with supervised_head.py")


def main():
    parser = argparse.ArgumentParser(
        description="Re-extract KAIROS labels for L1** experiment")
    parser.add_argument("--dataset", default="theia",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--label-type", default="l1",
                        help="Label column suffix (reads label_{type} from parquets)")
    args = parser.parse_args()
    relabel_dataset(args.dataset, args.label_type)


if __name__ == "__main__":
    main()
