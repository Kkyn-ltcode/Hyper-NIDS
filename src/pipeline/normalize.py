"""
Post-processing: Z-score normalization using training-only statistics.

Computes mean/std from chronologically first shards (training set),
then applies normalization to all shards. Binary/one-hot features
are auto-detected and left unnormalized.

Usage:
    python -m src.pipeline.normalize --dataset theia [--train-shards 0-6]
"""

import argparse
import gc
from pathlib import Path

import numpy as np


DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3"
)


def detect_binary_features(
    features_dir: Path,
    shard_indices: list[int],
    n_features: int,
) -> list[bool]:
    """
    Auto-detect binary features by checking if all values are in {0, 1}.

    Scans first 3 shards for efficiency.

    Returns:
        List of booleans, True = binary (skip normalization)
    """
    is_binary = [True] * n_features
    check_shards = shard_indices[:3]

    for idx in check_shards:
        data = np.load(features_dir / f"thyne_shard{idx}.npz")
        X = data["X"]
        for j in range(n_features):
            if not is_binary[j]:
                continue
            col = X[:, j]
            unique_vals = np.unique(col[~np.isnan(col)])
            if len(unique_vals) > 2 or not all(v in (0.0, 1.0, -1.0)
                                                for v in unique_vals):
                is_binary[j] = False
        del data, X
        gc.collect()

    return is_binary


def compute_train_stats(
    features_dir: Path,
    train_indices: list[int],
    n_features: int,
    is_binary: list[bool],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std from training shards only (online algorithm).

    Uses Welford's method for numerically stable single-pass computation.

    Returns:
        mean: (n_features,) array
        std: (n_features,) array
    """
    # Welford's online algorithm
    count = np.zeros(n_features, dtype=np.float64)
    mean = np.zeros(n_features, dtype=np.float64)
    m2 = np.zeros(n_features, dtype=np.float64)

    for idx in train_indices:
        data = np.load(features_dir / f"thyne_shard{idx}.npz")
        X = data["X"].astype(np.float64)

        for j in range(n_features):
            if is_binary[j]:
                continue
            col = X[:, j]
            valid = col[~np.isnan(col)]
            for val in valid:
                count[j] += 1
                delta = val - mean[j]
                mean[j] += delta / count[j]
                delta2 = val - mean[j]
                m2[j] += delta * delta2

        del data, X
        gc.collect()

    std = np.zeros(n_features, dtype=np.float64)
    for j in range(n_features):
        if count[j] > 1 and not is_binary[j]:
            std[j] = np.sqrt(m2[j] / count[j])
        else:
            std[j] = 1.0  # Avoid division by zero

    # Binary features: keep mean=0, std=1 (no-op normalization)
    for j in range(n_features):
        if is_binary[j]:
            mean[j] = 0.0
            std[j] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Normalize features using training-set statistics")
    parser.add_argument("--dataset", default="theia",
                        choices=["theia", "trace"])
    parser.add_argument("--train-shards", default="0-6",
                        help="Shard range for training (e.g., '0-6')")
    args = parser.parse_args()

    features_dir = DATA_ROOT / args.dataset / "features"
    norm_dir = DATA_ROOT / args.dataset / "features_norm"
    norm_dir.mkdir(parents=True, exist_ok=True)

    # Parse shard ranges
    start, end = map(int, args.train_shards.split("-"))
    all_files = sorted(features_dir.glob("thyne_shard*.npz"))
    all_indices = [int(f.stem.replace("thyne_shard", "")) for f in all_files]
    train_indices = [i for i in all_indices if start <= i <= end]
    test_indices = [i for i in all_indices if i not in train_indices]

    print("=" * 60)
    print(f"NORMALIZATION: {args.dataset.upper()}")
    print("=" * 60)
    print(f"  Train shards: {train_indices}")
    print(f"  Test shards:  {test_indices}")
    print(f"  Output dir:   {norm_dir}")

    # Load feature names
    names_file = features_dir / "feature_names.txt"
    feat_names = names_file.read_text().strip().split("\n")
    n_features = len(feat_names)
    print(f"  Features:     {n_features}")

    # ============================================================
    # Step 1: Detect binary features
    # ============================================================
    print(f"\n[1/3] Detecting binary features...")
    is_binary = detect_binary_features(features_dir, all_indices, n_features)

    n_binary = sum(is_binary)
    n_continuous = n_features - n_binary
    print(f"  Binary (skip normalization): {n_binary}")
    for j, name in enumerate(feat_names):
        if is_binary[j]:
            print(f"    {name}")
    print(f"  Continuous (z-score):        {n_continuous}")
    for j, name in enumerate(feat_names):
        if not is_binary[j]:
            print(f"    {name}")

    # ============================================================
    # Step 2: Compute training statistics
    # ============================================================
    print(f"\n[2/3] Computing training statistics (shards {train_indices})...")
    mean, std = compute_train_stats(
        features_dir, train_indices, n_features, is_binary)

    # Print stats for continuous features
    print(f"\n  {'Feature':35s} {'Mean':>10s} {'Std':>10s}")
    print(f"  {'─'*57}")
    for j, name in enumerate(feat_names):
        if not is_binary[j]:
            print(f"  {name:35s} {mean[j]:>10.4f} {std[j]:>10.4f}")

    # Save scaler params
    scaler_path = norm_dir / "scaler_params.npz"
    np.savez(
        scaler_path,
        mean=mean,
        std=std,
        is_binary=np.array(is_binary),
        feature_names=np.array(feat_names),
        train_shards=np.array(train_indices),
        test_shards=np.array(test_indices),
    )
    print(f"\n  Scaler saved to {scaler_path.name}")

    # ============================================================
    # Step 3: Normalize all shards
    # ============================================================
    print(f"\n[3/3] Normalizing all shards...")

    for idx in all_indices:
        src_path = features_dir / f"thyne_shard{idx}.npz"
        dst_path = norm_dir / f"thyne_shard{idx}.npz"

        data = np.load(src_path, allow_pickle=True)
        X = data["X"].astype(np.float32)

        # Z-score: (x - mean) / std, only for continuous features
        for j in range(n_features):
            if not is_binary[j] and std[j] > 0:
                X[:, j] = (X[:, j] - mean[j]) / std[j]

        split = "TRAIN" if idx in train_indices else "TEST"
        n = len(X)
        n_atk = int((data["y_broad"] == 1).sum())

        np.savez_compressed(
            dst_path,
            X=X,
            y_broad=data["y_broad"],
            timestamp_nanos=data["timestamp_nanos"],
            subject_uuid=data["subject_uuid"],
        )

        print(f"  Shard {idx} [{split}]: {n:>10,} events, "
              f"{n_atk:>8,} attack, "
              f"→ {dst_path.name} "
              f"({dst_path.stat().st_size/1e6:.0f} MB)")

        del data, X
        gc.collect()

    # ============================================================
    # Validation
    # ============================================================
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")

    # Check that training features have ~zero mean, ~unit std
    print(f"\n  Training set statistics (should be ~0 mean, ~1 std):")
    for idx in train_indices[:2]:  # Spot-check first 2 train shards
        data = np.load(norm_dir / f"thyne_shard{idx}.npz")
        X = data["X"]
        print(f"\n  Shard {idx}:")
        for j, name in enumerate(feat_names):
            if not is_binary[j]:
                m = X[:, j].mean()
                s = X[:, j].std()
                flag = "" if abs(m) < 0.5 and 0.5 < s < 2.0 else " ⚠"
                print(f"    {name:35s} mean={m:>7.3f}  std={s:>7.3f}{flag}")
        del data, X
        gc.collect()

    # Check test shards have non-zero mean (expected, no leakage)
    if test_indices:
        idx = test_indices[0]
        data = np.load(norm_dir / f"thyne_shard{idx}.npz")
        X = data["X"]
        print(f"\n  Test shard {idx} (mean may differ from 0 — expected):")
        for j, name in enumerate(feat_names):
            if not is_binary[j]:
                m = X[:, j].mean()
                s = X[:, j].std()
                print(f"    {name:35s} mean={m:>7.3f}  std={s:>7.3f}")
        del data, X
        gc.collect()

    print(f"\n  ✓ Normalization complete. No data leakage.")


if __name__ == "__main__":
    main()
