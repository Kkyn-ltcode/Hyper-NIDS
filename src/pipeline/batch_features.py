"""
Batch Feature Extraction for all shards.

Two-pass pipeline:
    Pass 1: Compute global statistics (type counts, first-seen timestamps)
    Pass 2: Extract features per shard with correct cross-shard continuity

Usage:
    python -m src.pipeline.batch_features --dataset theia
    python -m src.pipeline.batch_features --dataset theia --validate
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.feature_extractor import (
    GlobalStats,
    compute_global_stats,
    extract_features,
)


DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3"
)


def main():
    parser = argparse.ArgumentParser(
        description="Batch feature extraction across all shards")
    parser.add_argument("--dataset", default="theia",
                        choices=["theia", "trace"])
    parser.add_argument("--validate", action="store_true",
                        help="Validate feature distributions after extraction")
    args = parser.parse_args()

    labeled_dir = DATA_ROOT / args.dataset / "labeled"
    features_dir = DATA_ROOT / args.dataset / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(labeled_dir.glob("labeled_shard*.parquet"))
    if not shard_files:
        print(f"ERROR: No labeled shards in {labeled_dir}")
        print("  Run `python -m src.pipeline.batch_ingest` first.")
        return

    n_shards = len(shard_files)
    print("=" * 60)
    print(f"BATCH FEATURE EXTRACTION: {args.dataset.upper()}")
    print("=" * 60)
    print(f"  Labeled shards: {n_shards}")
    print(f"  Output dir:     {features_dir}")

    # ============================================================
    # Pass 1: Global statistics
    # ============================================================
    print(f"\n{'='*60}")
    print("PASS 1: Global Statistics")
    print(f"{'='*60}")

    global_stats_path = features_dir / "global_stats.npz"
    if global_stats_path.exists() and not args.validate:
        print("  Loading cached global stats...")
        data = np.load(global_stats_path, allow_pickle=True)
        global_stats = GlobalStats(
            total_events=int(data["total_events"]),
            type_counts=data["type_counts"].item(),
            subject_first_ts=data["subject_first_ts"].item(),
            object_first_ts=data["object_first_ts"].item(),
        )
        print(f"  {global_stats.total_events:,} events, "
              f"{len(global_stats.type_counts)} types, "
              f"{len(global_stats.subject_first_ts):,} subjects, "
              f"{len(global_stats.object_first_ts):,} objects")
    else:
        t0 = time.time()
        global_stats = compute_global_stats(labeled_dir)
        print(f"  Time: {time.time()-t0:.1f}s")

        # Cache for future runs
        np.savez(
            global_stats_path,
            total_events=global_stats.total_events,
            type_counts=global_stats.type_counts,
            subject_first_ts=global_stats.subject_first_ts,
            object_first_ts=global_stats.object_first_ts,
        )
        print(f"  Cached to {global_stats_path.name}")

    # ============================================================
    # Pass 2: Per-shard feature extraction
    # ============================================================
    print(f"\n{'='*60}")
    print("PASS 2: Per-Shard Feature Extraction")
    print(f"{'='*60}")

    t_total = time.time()
    subject_carry = {}  # Carry-over last timestamps across shards
    all_feat_names = None
    total_events = 0

    for i, shard_file in enumerate(shard_files):
        shard_name = shard_file.stem  # labeled_shard0
        shard_idx = int(shard_name.replace("labeled_shard", ""))
        npz_path = features_dir / f"thyne_shard{shard_idx}.npz"

        # Skip if already extracted
        if npz_path.exists() and not args.validate:
            data = np.load(npz_path, allow_pickle=True)
            n = len(data["y_broad"])
            total_events += n
            print(f"  Shard {shard_idx}: SKIPPED (exists, {n:,} events)")

            # Still need to update carry for subsequent shards
            df_carry = pd.read_parquet(
                shard_file,
                columns=["subject_uuid", "timestamp_nanos"],
            )
            last_ts = df_carry.groupby(
                "subject_uuid"
            )["timestamp_nanos"].max().to_dict()
            subject_carry.update(last_ts)
            del df_carry, last_ts, data
            gc.collect()
            continue

        print(f"\n  Shard {shard_idx}...")
        t0 = time.time()

        # Load labeled shard
        df = pd.read_parquet(shard_file)
        n = len(df)
        total_events += n

        # Extract features with global stats and carry-over
        X, feat_names, last_ts_out = extract_features(
            df,
            global_stats=global_stats,
            subject_last_ts_carry=subject_carry if subject_carry else None,
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Update carry for next shard
        subject_carry.update(last_ts_out)

        if all_feat_names is None:
            all_feat_names = feat_names
        else:
            # Ensure consistent feature names (types may vary across shards)
            if feat_names != all_feat_names:
                print(f"    ⚠ Feature name mismatch! Padding...")
                # Align columns: add missing types as zero columns
                X_aligned = np.zeros(
                    (n, len(all_feat_names)), dtype=np.float32)
                for j, name in enumerate(all_feat_names):
                    if name in feat_names:
                        X_aligned[:, j] = X[:, feat_names.index(name)]
                X = X_aligned

        # Save .npz with features + labels + timestamps
        np.savez_compressed(
            npz_path,
            X=X,
            y_broad=df["label_broad"].values,
            timestamp_nanos=df["timestamp_nanos"].values,
            subject_uuid=df["subject_uuid"].values,
        )

        elapsed = time.time() - t0
        n_atk = int((df["label_broad"] == 1).sum())
        print(f"    Events: {n:,}, Attack: {n_atk:,} "
              f"({100*n_atk/n:.1f}%)")
        print(f"    Features: {X.shape[1]}")
        print(f"    File: {npz_path.name} "
              f"({npz_path.stat().st_size/1e6:.0f} MB)")
        print(f"    Time: {elapsed:.1f}s")

        del df, X, last_ts_out
        gc.collect()

    # Save feature names
    if all_feat_names:
        names_path = features_dir / "feature_names.txt"
        with open(names_path, "w") as f:
            for name in all_feat_names:
                f.write(name + "\n")
        print(f"\n  Feature names saved to {names_path.name}")

    print(f"\n  Total events processed: {total_events:,}")
    print(f"  Total time: {time.time()-t_total:.1f}s")

    # ============================================================
    # Validation
    # ============================================================
    if args.validate or True:  # Always validate
        print(f"\n{'='*60}")
        print("VALIDATION")
        print(f"{'='*60}")

        npz_files = sorted(features_dir.glob("thyne_shard*.npz"))
        print(f"  Feature files: {len(npz_files)}")

        total_n = 0
        total_atk = 0
        feat_stats = {}

        for npz_file in npz_files:
            data = np.load(npz_file, allow_pickle=True)
            X = data["X"]
            y = data["y_broad"]
            n = len(y)
            total_n += n
            total_atk += int((y == 1).sum())

            # Spot-check feature ranges per shard
            shard_name = npz_file.stem
            for col_idx in [
                len(all_feat_names) - 6,  # type_rarity
                len(all_feat_names) - 4,  # time_gap_same_subject
                len(all_feat_names) - 1,  # has_path
            ]:
                if col_idx < 0 or col_idx >= X.shape[1]:
                    continue
                col_name = all_feat_names[col_idx]
                vals = X[:, col_idx]
                if col_name not in feat_stats:
                    feat_stats[col_name] = []
                feat_stats[col_name].append({
                    "shard": shard_name,
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                })

            del data, X, y
            gc.collect()

        print(f"\n  Total events: {total_n:,}")
        print(f"  Attack:       {total_atk:,} "
              f"({100*total_atk/total_n:.1f}%)")

        # Check feature consistency across shards
        print(f"\n  Feature distribution consistency:")
        for feat_name, shard_stats in feat_stats.items():
            means = [s["mean"] for s in shard_stats]
            stds = [s["std"] for s in shard_stats]
            mean_range = max(means) - min(means)
            print(f"    {feat_name}:")
            print(f"      mean range: {min(means):.4f} – {max(means):.4f} "
                  f"(span={mean_range:.4f})")
            if mean_range > 0.5:
                print(f"      ⚠ Large variation across shards!")
            else:
                print(f"      ✓ Consistent")


if __name__ == "__main__":
    main()
