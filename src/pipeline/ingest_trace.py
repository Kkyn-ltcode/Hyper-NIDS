"""
TRACE Dataset Ingestion Pipeline.

Since TRACE data is already parsed into the same Parquet schema as
Theia, we can reuse the existing pipeline with dataset='trace'.

This script runs the full pipeline:
    1. Label events (broad + crossprocess)
    2. Extract features
    3. Build graph (entity vocab + incidence)
    4. Build per-subject sequences

Usage:
    python -m src.pipeline.ingest_trace
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ground_truth import (
    load_ground_truth,
    label_events,
    label_crossprocess_events,
)


DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3"
)


def discover_shards(dataset_dir: Path) -> list[int]:
    shard_dir = dataset_dir / "shards"
    indices = []
    for f in sorted(shard_dir.glob("events_shard*.parquet")):
        idx = int(f.stem.replace("events_shard", ""))
        indices.append(idx)
    return sorted(indices)


def label_shard(dataset_dir: Path, shard_idx: int, gt, output_dir: Path):
    """Label a single shard and save."""
    out_path = output_dir / f"labeled_shard{shard_idx}.parquet"
    if out_path.exists():
        df = pd.read_parquet(out_path, columns=["label_broad"])
        n = len(df)
        n_atk = int((df["label_broad"] == 1).sum())
        print(f"    SKIP (exists): {n:,} events, {n_atk:,} attack")
        return n, n_atk

    shard_dir = dataset_dir / "shards"
    events = pd.read_parquet(shard_dir / f"events_shard{shard_idx}.parquet")
    subjects = pd.read_parquet(shard_dir / f"subjects_shard{shard_idx}.parquet")
    objects = pd.read_parquet(shard_dir / f"objects_shard{shard_idx}.parquet")

    # Broad labels
    labels_broad = label_events(events, subjects, objects, gt)
    events["label_broad"] = labels_broad.values

    # Cross-process labels
    labels_xproc = label_crossprocess_events(events, subjects, objects, gt)
    events["label_crossprocess"] = labels_xproc.values

    # Placeholders for narrow/ioc
    events["label_narrow"] = np.int8(-1)
    events["label_ioc"] = np.int8(-1)

    # HE sizes
    has_sub = events["subject_uuid"].notna()
    has_obj = events["predicate_object_uuid"].notna()
    has_obj2 = events["predicate_object2_uuid"].notna()
    events["he_size"] = (has_sub.astype(int) + has_obj.astype(int)
                         + has_obj2.astype(int)).values.astype(np.int8)

    events.to_parquet(out_path, index=False)

    n = len(events)
    n_atk = int(labels_broad.sum())
    n_xproc = int(labels_xproc.sum())
    print(f"    Events: {n:,}, Broad: {n_atk:,} ({100*n_atk/n:.1f}%), "
          f"XProc: {n_xproc:,} ({100*n_xproc/n:.3f}%)")

    del events, subjects, objects
    gc.collect()
    return n, n_atk


def main():
    parser = argparse.ArgumentParser(description="Ingest TRACE dataset")
    parser.add_argument("--step", default="label",
                        choices=["label", "features", "graph", "sequences", "all"])
    args = parser.parse_args()

    dataset = "trace"
    dataset_dir = DATA_ROOT / dataset
    labeled_dir = dataset_dir / "labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(dataset)
    shard_ids = discover_shards(dataset_dir)

    print("=" * 60)
    print(f"TRACE INGESTION")
    print("=" * 60)
    print(f"  Shards: {shard_ids}")
    print(f"  Attack IPs: {len(gt.attack_ips)}")
    print(f"  Entry process: {gt.attack_entry_process}")
    print(f"  Malicious processes: {gt.malicious_process_basenames}")

    if args.step in ("label", "all"):
        print(f"\n── Step 1: Label ──")
        total_n, total_atk = 0, 0
        for sid in shard_ids:
            print(f"  Shard {sid}...")
            n, n_atk = label_shard(dataset_dir, sid, gt, labeled_dir)
            total_n += n
            total_atk += n_atk
        print(f"\n  Total: {total_n:,} events, "
              f"{total_atk:,} attack ({100*total_atk/total_n:.1f}%)")

    if args.step in ("features", "all"):
        print(f"\n── Step 2: Features ──")
        print("  Run: python -m src.pipeline.feature_extractor --dataset trace")

    if args.step in ("graph", "all"):
        print(f"\n── Step 3: Graph ──")
        print("  Run: python -m src.pipeline.build_graph --dataset trace")

    if args.step in ("sequences", "all"):
        print(f"\n── Step 4: Sequences ──")
        print("  Run: python -m src.pipeline.build_sequences --dataset trace")


if __name__ == "__main__":
    main()
