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
    compute_subject_last_ts_per_shard,
)

def _process_shard_pass2(args_tuple):
    import pandas as pd
    import numpy as np
    import time
    from pathlib import Path
    
    (shard_idx, shard_file_str, npz_path_str, global_stats_path_str, objects_df, subject_carry, n_shards) = args_tuple
    
    shard_file = Path(shard_file_str)
    npz_path = Path(npz_path_str)
    
    t0 = time.time()
    df = pd.read_parquet(shard_file)
    n = len(df)
    
    # Load global_stats locally to avoid massive pickling overhead
    data = np.load(global_stats_path_str, allow_pickle=True)
    global_stats = GlobalStats(
        total_events=int(data["total_events"]),
        type_counts=data["type_counts"].item(),
        subject_first_ts=data["subject_first_ts"].item(),
        object_first_ts=data["object_first_ts"].item(),
    )
    
    X, feat_names, _ = extract_features(
        df,
        global_stats=global_stats,
        subject_last_ts_carry=subject_carry if subject_carry else None,
        objects_df=objects_df,
    )
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Save .npz with features + labels + timestamps
    np.savez_compressed(
        npz_path,
        X=X,
        y_broad=df["label_broad"].values,
        y_narrow=df["label_narrow"].values if "label_narrow" in df.columns else np.full(n, -1, dtype=np.int8),
        y_ioc=df["label_ioc"].values if "label_ioc" in df.columns else np.full(n, -1, dtype=np.int8),
        y_crossprocess=df["label_crossprocess"].values if "label_crossprocess" in df.columns else np.full(n, -1, dtype=np.int8),
        timestamp_nanos=df["timestamp_nanos"].values,
        subject_uuid=df["subject_uuid"].values,
        predicate_object_uuid=df["predicate_object_uuid"].values,
    )
    
    elapsed = time.time() - t0
    n_atk = int((df["label_broad"] == 1).sum())
    
    return shard_idx, n, n_atk, X.shape[1], feat_names, elapsed, npz_path.name, npz_path.stat().st_size



DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3"
)


def main():
    parser = argparse.ArgumentParser(
        description="Batch feature extraction across all shards")
    parser.add_argument("--dataset", default="theia",
                        choices=["theia", "trace", "cadets", "trace-1"])
    parser.add_argument("--validate", action="store_true",
                        help="Validate feature distributions after extraction")
    args = parser.parse_args()

    labeled_dir = DATA_ROOT / args.dataset / "labeled"
    features_dir = DATA_ROOT / args.dataset / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    def _extract_idx(f):
        return int(f.name.replace("labeled_shard", "").replace(".parquet", ""))
    shard_files = sorted(labeled_dir.glob("labeled_shard*.parquet"), key=_extract_idx)
    if not shard_files:
        print(f"ERROR: No labeled shards in {labeled_dir}")
        print("  Run `python -m src.pipeline.batch_ingest` first.")
        return

    n_shards = len(shard_files)

    # Load object metadata for object-aware features
    objects_path = DATA_ROOT / args.dataset / "objects.parquet"
    if objects_path.exists():
        objects_df = pd.read_parquet(
            objects_path, columns=["uuid", "object_type"])
        print(f"  Loaded {len(objects_df):,} objects from objects.parquet")
    else:
        objects_df = None
        print("  ⚠ objects.parquet not found; object type features will be zero")

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
    all_feat_names = None
    total_events = 0

    # Fast sequential pre-computation of carry state
    print("  Pre-computing cross-shard boundaries (fast)...")
    shard_carry_dicts, shard_unique_subs = compute_subject_last_ts_per_shard(labeled_dir)
    
    cumulative_carry = []
    current_carry = {}
    for i in range(len(shard_carry_dicts)):
        # Extract minimal carry for shard i BEFORE applying shard i's updates
        needed = shard_unique_subs[i]
        minimal_carry = {k: current_carry[k] for k in needed if k in current_carry}
        cumulative_carry.append(minimal_carry)
        
        # Update global state with shard i's last timestamps
        current_carry.update(shard_carry_dicts[i])
        
    del current_carry, shard_carry_dicts, shard_unique_subs

    # Build tasks
    tasks = []
    for i, shard_file in enumerate(shard_files):
        shard_name = shard_file.stem
        shard_idx = int(shard_name.replace("labeled_shard", ""))
        npz_path = features_dir / f"thyne_shard{shard_idx}.npz"
        
        # Skip if already extracted
        if npz_path.exists() and not args.validate:
            import pyarrow.parquet as pq
            try:
                data = np.load(npz_path, allow_pickle=True)
                n = len(data["y_broad"])
                del data
            except Exception:
                n = 0
            total_events += n
            print(f"  Shard {shard_idx}/{n_shards-1}: SKIPPED (exists, {n:,} events)")
            continue
            
        tasks.append((
            shard_idx, 
            str(shard_file), 
            str(npz_path), 
            str(global_stats_path), 
            objects_df, 
            cumulative_carry[i], 
            n_shards
        ))

    if tasks:
        import concurrent.futures
        print(f"\n  Extracting features for {len(tasks)} shards (parallelized)...")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(_process_shard_pass2, tasks):
                shard_idx, n, n_atk, n_feats, feat_names, elapsed, fname, fsize = result
                
                print(f"    Shard {shard_idx}/{n_shards-1}: Events: {n:,}, Attack: {n_atk:,} "
                      f"({100*n_atk/n:.1f}%) | Feats: {n_feats} | File: {fname} ({fsize/1e6:.0f} MB) | Time: {elapsed:.1f}s")
                      
                total_events += n
                
                if all_feat_names is None:
                    all_feat_names = feat_names
                elif feat_names != all_feat_names:
                    print(f"    ⚠ Feature name mismatch in Shard {shard_idx}! Run a consistent extraction.")


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
    if args.validate:
        if all_feat_names is None:
            # Load from saved file if all shards were skipped
            names_path = features_dir / "feature_names.txt"
            if names_path.exists():
                with open(names_path) as f:
                    all_feat_names = [l.strip() for l in f if l.strip()]
            else:
                print("  Cannot validate: feature_names.txt not found.")
                return

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
