"""
Multi-Shard Batch Ingestion & Labeling for DARPA TC E3.

Processes all shards through the modular pipeline:
    1. Load each shard's events/subjects/objects
    2. Label with unified ground truth
    3. Build hypergraph stats and sequence stats per shard
    4. Save labeled shard with label columns
    5. Validate total counts against expected values

Usage:
    python -m src.pipeline.batch_ingest --dataset theia
    python -m src.pipeline.batch_ingest --dataset theia --validate-only
"""

import json
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ground_truth import load_ground_truth, label_events
from src.data.build_sequences import build_subject_sequences, sequence_stats
from src.data.ground_truth import label_narrow_events, label_ioc_events

DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3"
)


def discover_shards(dataset: str) -> list[int]:
    """Find all available shard indices for a dataset."""
    shard_dir = DATA_ROOT / dataset / "shards"
    if not shard_dir.exists():
        raise FileNotFoundError(f"Shard dir not found: {shard_dir}")

    indices = []
    for f in sorted(shard_dir.glob("events_shard*.parquet")):
        # Extract shard index from filename like events_shard0.parquet
        name = f.stem  # events_shard0
        idx = int(name.replace("events_shard", ""))
        indices.append(idx)
    return sorted(indices)


def process_shard(
    dataset: str,
    shard_idx: int,
    gt,
    output_dir: Path,
) -> dict:
    """
    Process a single shard: load, label, save.

    Returns stats dict.
    """
    shard_dir = DATA_ROOT / dataset / "shards"
    out_path = output_dir / f"labeled_shard{shard_idx}.parquet"

    REQUIRED_COLS = {"label_broad", "label_narrow", "label_ioc", "he_size"}
    if out_path.exists():
        import pyarrow.parquet as pq
        existing_cols = set(pq.read_schema(out_path).names)
        if not REQUIRED_COLS.issubset(existing_cols):
            print(f"    WARNING: Shard {shard_idx} missing columns "
                  f"{REQUIRED_COLS - existing_cols}, reprocessing...")
        else:
            df = pd.read_parquet(out_path, columns=["label_broad"])
            stats = {
                "shard": shard_idx,
                "n_events": len(df),
                "n_attack_broad": int((df["label_broad"] == 1).sum()),
                "skipped": True,
            }
            del df
            return stats

    # Load
    events_df = pd.read_parquet(shard_dir / f"events_shard{shard_idx}.parquet")
    subjects_df = pd.read_parquet(shard_dir / f"subjects_shard{shard_idx}.parquet")
    objects_df = pd.read_parquet(shard_dir / f"objects_shard{shard_idx}.parquet")

    n_events = len(events_df)
    if n_events == 0:
        print(f"    WARNING: Shard {shard_idx} has 0 events. Skipping...")
        return {
            "shard": shard_idx,
            "n_events": 0,
            "n_attack_broad": 0,
            "skipped": True,
        }

    # Label (broad = entire process tree)
    labels_broad  = label_events(events_df, subjects_df, objects_df, gt)
    labels_narrow = label_narrow_events(events_df, subjects_df, objects_df, gt)
    labels_ioc    = label_ioc_events(events_df, subjects_df, objects_df, gt)

    # Hyperedge sizes
    has_sub = events_df["subject_uuid"].notna()
    has_obj = events_df["predicate_object_uuid"].notna()
    has_obj2 = events_df["predicate_object2_uuid"].notna()
    he_sizes = has_sub.astype(int) + has_obj.astype(int) + has_obj2.astype(int)

    # Sequence stats
    sequences = build_subject_sequences(events_df)
    seq_s = sequence_stats(sequences)
    del sequences

    # Add label columns to events
    events_df["label_broad"]  = labels_broad.values
    events_df["label_narrow"] = labels_narrow.values
    events_df["label_ioc"]    = labels_ioc.values
    events_df["he_size"] = he_sizes.values.astype(np.int8)

    # Save labeled shard
    events_df.to_parquet(out_path, index=False)

    n_atk = int((labels_broad == 1).sum())
    stats = {
        "shard": shard_idx,
        "n_events": n_events,
        "n_attack_broad": n_atk,
        "n_benign": n_events - n_atk,
        "pct_attack": 100 * n_atk / n_events,
        "he_size_3_pct": 100 * (he_sizes == 3).sum() / n_events,
        "n_subjects": len(subjects_df),
        "n_objects": len(objects_df),
        "n_unique_obj2": events_df["predicate_object2_uuid"].nunique(),
        "seq_len_max": seq_s["seq_len_max"],
        "subjects_gt_10k": seq_s["subjects_gt_10000"],
        "skipped": False,
    }

    del events_df, subjects_df, objects_df, labels_broad
    gc.collect()

    return stats


def validate_totals(output_dir: Path, dataset: str, expected_total: int = None):
    """Validate total counts across all labeled shards."""
    if expected_total is None:
        summary_path = DATA_ROOT / dataset / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                expected_total = json.load(f).get("total_events", 0)
        else:
            expected_total = 0


    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")

    files = sorted(output_dir.glob("labeled_shard*.parquet"))
    if not files:
        print("  No labeled shards found!")
        return

    total_events = 0
    total_attack = 0

    for f in files:
        df = pd.read_parquet(f, columns=["label_broad"])
        n = len(df)
        n_atk = int((df["label_broad"] == 1).sum())
        total_events += n
        total_attack += n_atk
        del df

    total_benign = total_events - total_attack

    print(f"\n  Total events:  {total_events:,}")
    print(f"  Attack:        {total_attack:,} ({100*total_attack/total_events:.1f}%)")
    print(f"  Benign:        {total_benign:,} ({100*total_benign/total_events:.1f}%)")
    print(f"  Expected:      {expected_total:,}")

    diff = abs(total_events - expected_total)
    if diff == 0:
        print(f"  ✓ EXACT MATCH")
    elif diff < 1000:
        print(f"  ≈ CLOSE (diff={diff})")
    else:
        print(f"  ⚠ MISMATCH (diff={diff:,})")


def main():
    parser = argparse.ArgumentParser(
        description="Batch ingest and label all shards")
    parser.add_argument("--dataset", default="theia",
                        choices=["theia", "trace", "trace-1"])
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation on existing labeled shards")
    args = parser.parse_args()

    output_dir = DATA_ROOT / args.dataset / "labeled"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        validate_totals(output_dir)
        return

    print("=" * 60)
    print(f"BATCH INGEST: {args.dataset.upper()}")
    print("=" * 60)

    # Discover shards
    shard_indices = discover_shards(args.dataset)
    print(f"\n  Found {len(shard_indices)} shards: {shard_indices}")

    # Load ground truth once
    gt = load_ground_truth(args.dataset)
    print(f"  Ground truth: {gt.dataset}")
    print(f"    Attack IPs: {len(gt.attack_ips)}")
    print(f"    Malicious processes: {gt.malicious_process_basenames}")
    print(f"    Entry process: {gt.attack_entry_process}")

    # Process each shard
    all_stats = []
    t_total = time.time()

    for shard_idx in shard_indices:
        print(f"\n{'─'*40}")
        print(f"  Shard {shard_idx}...")
        t0 = time.time()

        stats = process_shard(args.dataset, shard_idx, gt, output_dir)
        all_stats.append(stats)

        elapsed = time.time() - t0
        if stats.get("skipped"):
            print(f"    SKIPPED (already exists): "
                  f"{stats['n_events']:,} events, "
                  f"{stats['n_attack_broad']:,} attack")
        else:
            print(f"    Events: {stats['n_events']:,}")
            print(f"    Attack: {stats['n_attack_broad']:,} "
                  f"({stats['pct_attack']:.1f}%)")
            print(f"    Size-3: {stats['he_size_3_pct']:.1f}%")
            print(f"    Max seq: {stats['seq_len_max']:,}, "
                  f">10K subjects: {stats['subjects_gt_10k']}")
            print(f"    Obj2 unique: {stats['n_unique_obj2']:,}")
            print(f"    Time: {elapsed:.1f}s")

    # Summary table
    print(f"\n{'='*60}")
    print("SHARD SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Shard':>5s} {'Events':>12s} {'Attack':>12s} {'%':>6s} "
          f"{'Size3%':>6s} {'MaxSeq':>8s}")
    print(f"  {'─'*55}")

    for s in all_stats:
        pct = s.get('pct_attack', 100*s['n_attack_broad']/max(s['n_events'],1))
        sz3 = s.get('he_size_3_pct', 0)
        maxseq = s.get('seq_len_max', 0)
        print(f"  {s['shard']:>5d} {s['n_events']:>12,} "
              f"{s['n_attack_broad']:>12,} {pct:>5.1f}% "
              f"{sz3:>5.1f}% {maxseq:>8,}")

    print(f"\n  Total time: {time.time()-t_total:.1f}s")

    # Validate
    validate_totals(output_dir, dataset=args.dataset)


if __name__ == "__main__":
    main()
