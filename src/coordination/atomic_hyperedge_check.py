"""
Atomic Hyperedge Sanity Check.

Each CDM event = one hyperedge connecting its entities (subject, object, object2).
No clustering needed — the provenance graph already defines hyperedges.

Steps:
    1. Count native hyperedge sizes (size-2 vs size-3)
    2. Label atomic hyperedges using ground truth
    3. RF sanity check on per-hyperedge features
    4. Single-feature confound check

Usage:
    python -m src.coordination.atomic_hyperedge_check
"""

import gc
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from src.coordination.ground_truth_e3 import label_events


DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3" / "theia"
)


def main():
    print("=" * 60)
    print("ATOMIC HYPEREDGE SANITY CHECK")
    print("=" * 60)

    # Load shard 0
    shard_dir = DATA_DIR / "shards"
    print("\nLoading shard 0...")
    events_df = pd.read_parquet(shard_dir / "events_shard0.parquet")
    subjects_df = pd.read_parquet(shard_dir / "subjects_shard0.parquet")
    objects_df = pd.read_parquet(shard_dir / "objects_shard0.parquet")
    print(f"  Events: {len(events_df):,}")
    print(f"  Time: {events_df['timestamp'].min()} → {events_df['timestamp'].max()}")

    # ============================================================
    # STEP 1: Count native hyperedge sizes
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 1: Native Hyperedge Sizes")
    print(f"{'='*60}")

    has_subject = events_df["subject_uuid"].notna()
    has_obj1 = events_df["predicate_object_uuid"].notna()
    has_obj2 = events_df["predicate_object2_uuid"].notna()

    # Size = number of non-null entity references
    sizes = has_subject.astype(int) + has_obj1.astype(int) + has_obj2.astype(int)
    size_counts = sizes.value_counts().sort_index()

    print(f"\n  Hyperedge size distribution:")
    for s, cnt in size_counts.items():
        pct = 100 * cnt / len(events_df)
        bar = "█" * int(pct)
        print(f"    size {s}: {cnt:>10,} ({pct:5.1f}%) {bar}")

    n_size3 = (sizes == 3).sum()
    n_size2 = (sizes == 2).sum()
    print(f"\n  Size-3 (subject + obj + obj2): {n_size3:,} ({100*n_size3/len(events_df):.1f}%)")
    print(f"  Size-2 (subject + obj):        {n_size2:,} ({100*n_size2/len(events_df):.1f}%)")

    # Which event types produce size-3?
    if n_size3 > 0:
        size3_events = events_df[sizes == 3]
        print(f"\n  Event types producing size-3 hyperedges:")
        for etype, cnt in size3_events["type"].value_counts().head(10).items():
            pct = 100 * cnt / n_size3
            print(f"    {etype:30s} {cnt:>10,} ({pct:5.1f}%)")

    # ============================================================
    # STEP 2: Label atomic hyperedges
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 2: Label Atomic Hyperedges")
    print(f"{'='*60}")

    print("\n  Labeling events with ground truth...")
    event_labels = label_events(events_df, subjects_df, objects_df)
    labels = event_labels.values

    n_atk = (labels == 1).sum()
    n_ben = (labels == 0).sum()
    print(f"  Attack hyperedges: {n_atk:,} ({100*n_atk/len(events_df):.1f}%)")
    print(f"  Benign hyperedges: {n_ben:,} ({100*n_ben/len(events_df):.1f}%)")
    print(f"  Imbalance ratio:   1:{n_ben/max(n_atk,1):.1f}")

    # Attack by event type
    print(f"\n  Attack hyperedges by event type:")
    atk_events = events_df[labels == 1]
    for etype, cnt in atk_events["type"].value_counts().head(10).items():
        total_of_type = (events_df["type"] == etype).sum()
        print(f"    {etype:30s} {cnt:>10,} / {total_of_type:>10,} "
              f"({100*cnt/total_of_type:.1f}% of type)")

    # ============================================================
    # STEP 3: RF sanity check
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 3: RF on Atomic Hyperedge Features")
    print(f"{'='*60}")

    print("\n  Extracting features...")
    t0 = time.time()

    # --- Feature engineering ---
    # 1. Event type (one-hot)
    event_type_dummies = pd.get_dummies(events_df["type"], prefix="etype")

    # 2. Hour of day
    hour = events_df["timestamp"].dt.hour.values.astype(np.float32)

    # 3. Hyperedge size
    he_size = sizes.values.astype(np.float32)

    # 4. Event type rarity (fraction of total events with this type)
    type_counts = events_df["type"].value_counts()
    type_freq = events_df["type"].map(type_counts).values.astype(np.float64)
    type_rarity = 1.0 - (type_freq / len(events_df))

    # 5. Event size field
    event_size = pd.to_numeric(events_df["size"], errors="coerce").fillna(0).values.astype(np.float32)

    # 6. Time gap from previous event sharing same subject
    ts_nanos = events_df["timestamp_nanos"].values.astype(np.float64)
    subject_uuids = events_df["subject_uuid"].values

    # Compute per-subject time gaps efficiently
    # Sort is already done (events sorted by timestamp)
    subject_last_ts = {}
    time_gap_same_subject = np.full(len(events_df), np.nan, dtype=np.float64)
    for i in range(len(events_df)):
        s = subject_uuids[i]
        if pd.notna(s):
            if s in subject_last_ts:
                time_gap_same_subject[i] = (ts_nanos[i] - subject_last_ts[s]) / 1e9
            subject_last_ts[s] = ts_nanos[i]

    # 7. Subject is "new" (first seen in last hour)
    subject_first_ts = {}
    subject_is_new = np.zeros(len(events_df), dtype=np.float32)
    for i in range(len(events_df)):
        s = subject_uuids[i]
        if pd.notna(s):
            if s not in subject_first_ts:
                subject_first_ts[s] = ts_nanos[i]
                subject_is_new[i] = 1.0
            elif (ts_nanos[i] - subject_first_ts[s]) < 3600e9:  # 1 hour
                subject_is_new[i] = 1.0

    # 8. Object is "new"
    obj_uuids = events_df["predicate_object_uuid"].values
    obj_first_ts = {}
    object_is_new = np.zeros(len(events_df), dtype=np.float32)
    for i in range(len(events_df)):
        o = obj_uuids[i]
        if pd.notna(o):
            if o not in obj_first_ts:
                obj_first_ts[o] = ts_nanos[i]
                object_is_new[i] = 1.0
            elif (ts_nanos[i] - obj_first_ts[o]) < 3600e9:
                object_is_new[i] = 1.0

    # 9. Has predicate_object_path (file path present)
    has_path = events_df["predicate_object_path"].notna().astype(np.float32).values

    # Combine all features
    X_parts = [
        event_type_dummies.values.astype(np.float32),
        hour.reshape(-1, 1),
        he_size.reshape(-1, 1),
        type_rarity.reshape(-1, 1),
        event_size.reshape(-1, 1),
        np.nan_to_num(time_gap_same_subject, nan=-1.0).reshape(-1, 1),
        subject_is_new.reshape(-1, 1),
        object_is_new.reshape(-1, 1),
        has_path.reshape(-1, 1),
    ]
    X = np.hstack(X_parts).astype(np.float32)

    feat_names = (
        list(event_type_dummies.columns) +
        ["hour", "he_size", "type_rarity", "event_size",
         "time_gap_same_subject", "subject_is_new", "object_is_new",
         "has_path"]
    )

    del event_type_dummies, type_freq; gc.collect()

    print(f"  Features: {X.shape[1]} ({len(feat_names)} named)")
    print(f"  Feature extraction time: {time.time()-t0:.1f}s")

    # Balanced subsample
    rng = np.random.default_rng(42)
    atk_idx = np.where(labels == 1)[0]
    ben_idx = np.where(labels == 0)[0]
    n_sample = min(len(atk_idx), len(ben_idx), 100_000)

    s_atk = rng.choice(atk_idx, n_sample, replace=False)
    s_ben = rng.choice(ben_idx, n_sample, replace=False)
    sample = np.sort(np.concatenate([s_atk, s_ben]))

    X_s = X[sample]
    y_s = labels[sample]

    print(f"  Sample: {len(X_s):,} ({n_sample:,} per class)")

    X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("  Training RF (5-fold CV)...")
    t0 = time.time()
    scores = cross_val_score(clf, X_s, y_s, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Feature importances
    clf.fit(X_s, y_s)
    imps = sorted(zip(feat_names, clf.feature_importances_), key=lambda x: -x[1])
    print(f"\n  Top 10 features:")
    for name, imp in imps[:10]:
        print(f"    {name:35s} {imp:.4f}")

    # ============================================================
    # STEP 4: Single-feature confound check
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 4: Single-Feature Confound Check")
    print(f"{'='*60}")

    # Use the balanced sample for fair AUC
    print(f"\n  Checking single-feature AUCs on balanced sample...")
    confounds = []
    for i, feat in enumerate(feat_names):
        vals = X_s[:, i]
        if np.std(vals) == 0:
            continue
        try:
            auc = roc_auc_score(y_s, vals)
            if auc < 0.5:
                auc = 1 - auc
            if auc > 0.70:
                confounds.append((feat, auc))
        except ValueError:
            pass

    confounds.sort(key=lambda x: -x[1])
    if confounds:
        print(f"\n  Features with single-feature AUC > 0.70:")
        for feat, auc in confounds[:15]:
            marker = " ⚠ CONFOUND" if auc > 0.95 else ""
            print(f"    {feat:35s} AUC={auc:.4f}{marker}")
    else:
        print(f"  No features with AUC > 0.70 — no trivial confounds.")

    # Free memory
    del X, X_s; gc.collect()

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total atomic hyperedges: {len(events_df):,}")
    print(f"  Size-2 (S+O):           {n_size2:,} ({100*n_size2/len(events_df):.1f}%)")
    print(f"  Size-3 (S+O+O2):        {n_size3:,} ({100*n_size3/len(events_df):.1f}%)")
    print(f"  Attack:                  {n_atk:,} ({100*n_atk/len(events_df):.1f}%)")
    print(f"  Benign:                  {n_ben:,}")
    print(f"  RF AUC:                  {scores.mean():.4f}")

    # Add this to the sanity check or run standalone:
    n_unique_obj2 = events_df["predicate_object2_uuid"].nunique()
    print(f"Unique predicateObject2: {n_unique_obj2:,} out of {len(events_df):,} events")
    # Also: what fraction of events have obj2 == obj1?
    same = (events_df["predicate_object_uuid"] == events_df["predicate_object2_uuid"]).sum()
    print(f"obj == obj2: {same:,} ({100*same/len(events_df):.1f}%)")


if __name__ == "__main__":
    main()
