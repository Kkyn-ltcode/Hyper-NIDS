"""
Hyperedge feature extraction for atomic provenance hyperedges.

Supports both single-shard (standalone) and multi-shard (global stats)
modes. When global stats are provided, features like type_rarity and
subject_is_new use corpus-wide frequencies instead of per-shard.

Usage:
    # Single shard (standalone)
    X, names = extract_features(events_df)

    # Multi-shard (with global stats)
    global_stats = compute_global_stats(shard_dirs)
    X, names = extract_features(events_df, global_stats=global_stats,
                                 subject_last_ts_carry=carry_dict)
"""

import gc
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class GlobalStats:
    """Pre-computed corpus-wide statistics for feature extraction."""
    # Total number of events across all shards
    total_events: int = 0
    # Event type -> count across all shards
    type_counts: dict = field(default_factory=dict)
    # Subject UUID -> first seen timestamp (nanos) across all shards
    subject_first_ts: dict = field(default_factory=dict)
    # Object UUID -> first seen timestamp (nanos) across all shards
    object_first_ts: dict = field(default_factory=dict)


def compute_global_stats(labeled_dir) -> GlobalStats:
    """
    Compute corpus-wide statistics by scanning all labeled shards.

    Only loads the columns needed (type, subject_uuid,
    predicate_object_uuid, predicate_object2_uuid, timestamp_nanos).
    Processes one shard at a time to limit memory.

    Args:
        labeled_dir: Path to directory containing labeled_shard*.parquet

    Returns:
        GlobalStats with type_counts, subject_first_ts, object_first_ts
    """
    from pathlib import Path
    from collections import Counter

    labeled_dir = Path(labeled_dir)
    files = sorted(labeled_dir.glob("labeled_shard*.parquet"))
    if not files:
        raise FileNotFoundError(f"No labeled shards in {labeled_dir}")

    stats = GlobalStats()
    type_counter = Counter()

    print(f"  Computing global stats from {len(files)} shards...")

    for f in files:
        shard_name = f.stem
        print(f"    Scanning {shard_name}...")

        # Type counts
        df_type = pd.read_parquet(f, columns=["type"])
        for t, c in df_type["type"].value_counts().items():
            type_counter[t] += c
        stats.total_events += len(df_type)
        del df_type

        # Subject first-seen timestamps
        df_sub = pd.read_parquet(
            f, columns=["subject_uuid", "timestamp_nanos"])
        sub_first = df_sub.groupby("subject_uuid")["timestamp_nanos"].min()
        for uuid, ts in sub_first.items():
            if uuid not in stats.subject_first_ts:
                stats.subject_first_ts[uuid] = ts
            else:
                stats.subject_first_ts[uuid] = min(
                    stats.subject_first_ts[uuid], ts)
        del df_sub, sub_first

        # Object first-seen timestamps (both obj and obj2)
        for col in ["predicate_object_uuid", "predicate_object2_uuid"]:
            df_obj = pd.read_parquet(f, columns=[col, "timestamp_nanos"])
            df_obj = df_obj.dropna(subset=[col])
            if len(df_obj) > 0:
                obj_first = df_obj.groupby(col)["timestamp_nanos"].min()
                for uuid, ts in obj_first.items():
                    if uuid not in stats.object_first_ts:
                        stats.object_first_ts[uuid] = ts
                    else:
                        stats.object_first_ts[uuid] = min(
                            stats.object_first_ts[uuid], ts)
                del obj_first
            del df_obj

        gc.collect()

    stats.type_counts = dict(type_counter)
    print(f"  Done. {stats.total_events:,} events, "
          f"{len(stats.type_counts)} types, "
          f"{len(stats.subject_first_ts):,} subjects, "
          f"{len(stats.object_first_ts):,} objects")

    return stats


def compute_subject_last_ts_per_shard(labeled_dir) -> list[dict]:
    """
    Compute each subject's last timestamp in each shard.

    Used to carry over time_gap_same_subject across shard boundaries.

    Returns:
        List of dicts, one per shard (ordered by shard index).
        Each dict maps subject_uuid -> last_timestamp_nanos in that shard.
    """
    from pathlib import Path

    labeled_dir = Path(labeled_dir)
    files = sorted(labeled_dir.glob("labeled_shard*.parquet"))

    carry_dicts = []
    for f in files:
        df = pd.read_parquet(f, columns=["subject_uuid", "timestamp_nanos"])
        last_ts = df.groupby("subject_uuid")["timestamp_nanos"].max()
        carry_dicts.append(last_ts.to_dict())
        del df, last_ts
        gc.collect()

    return carry_dicts


def extract_features(
    events_df: pd.DataFrame,
    global_stats: GlobalStats | None = None,
    subject_last_ts_carry: dict | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract per-hyperedge feature matrix from events DataFrame.

    Features (26 total):
        - Event type one-hot (~18 types)
        - hour: hour of day
        - he_size: number of non-null entity references (2 or 3)
        - type_rarity: 1 - (freq of this type / total events)
        - event_size: CDM event size field
        - time_gap_same_subject: seconds since last event by same subject
        - subject_is_new: 1 if subject first seen within last hour
        - object_is_new: 1 if object first seen within last hour
        - has_path: 1 if predicate_object_path is non-null

    Args:
        events_df: DataFrame sorted by timestamp_nanos.
        global_stats: Pre-computed corpus stats. If None, computes per-shard.
        subject_last_ts_carry: Dict mapping subject_uuid -> last timestamp
            from a previous shard, used to seed time_gap computation.

    Returns:
        X: np.ndarray of shape (n_events, n_features), dtype float32
        feature_names: list of feature name strings
    """
    n = len(events_df)

    # 1. Event type one-hot
    event_type_dummies = pd.get_dummies(events_df["type"], prefix="etype")

    # 2. Hour of day
    hour = events_df["timestamp"].dt.hour.values.astype(np.float32)

    # 3. Hyperedge size
    has_sub = events_df["subject_uuid"].notna().astype(int)
    has_obj = events_df["predicate_object_uuid"].notna().astype(int)
    has_obj2 = events_df["predicate_object2_uuid"].notna().astype(int)
    he_size = (has_sub + has_obj + has_obj2).values.astype(np.float32)

    # 4. Type rarity
    if global_stats is not None:
        # Use corpus-wide frequencies
        total = global_stats.total_events
        type_freq = events_df["type"].map(
            global_stats.type_counts
        ).astype(np.float64).fillna(1).values
    else:
        # Per-shard fallback
        total = n
        tc = events_df["type"].value_counts()
        type_freq = events_df["type"].map(tc).values.astype(np.float64)

    type_rarity = (1.0 - (type_freq / total)).astype(np.float32)

    # 5. Event size field
    event_size = pd.to_numeric(
        events_df["size"], errors="coerce"
    ).fillna(0).values.astype(np.float32)

    # 6. Time gap from previous event by same subject
    ts_nanos = events_df["timestamp_nanos"].values.astype(np.float64)
    subject_uuids = events_df["subject_uuid"].values

    # Seed with carry-over from previous shard
    subject_last_ts = {}
    if subject_last_ts_carry is not None:
        subject_last_ts = dict(subject_last_ts_carry)

    time_gap = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        s = subject_uuids[i]
        if pd.notna(s):
            if s in subject_last_ts:
                time_gap[i] = (ts_nanos[i] - subject_last_ts[s]) / 1e9
            subject_last_ts[s] = ts_nanos[i]

    # Record last timestamp per subject (for next shard's carry)
    last_ts_out = {s: subject_last_ts[s] for s in subject_last_ts}

    # 7. Subject is "new" (first seen within last hour)
    subject_is_new = np.zeros(n, dtype=np.float32)
    if global_stats is not None:
        # Use corpus-wide first-seen
        for i in range(n):
            s = subject_uuids[i]
            if pd.notna(s):
                first = global_stats.subject_first_ts.get(s, 0)
                if (ts_nanos[i] - first) < 3600e9:
                    subject_is_new[i] = 1.0
    else:
        # Per-shard fallback
        subject_first = {}
        for i in range(n):
            s = subject_uuids[i]
            if pd.notna(s):
                if s not in subject_first:
                    subject_first[s] = ts_nanos[i]
                    subject_is_new[i] = 1.0
                elif (ts_nanos[i] - subject_first[s]) < 3600e9:
                    subject_is_new[i] = 1.0

    # 8. Object is "new"
    obj_uuids = events_df["predicate_object_uuid"].values
    object_is_new = np.zeros(n, dtype=np.float32)
    if global_stats is not None:
        for i in range(n):
            o = obj_uuids[i]
            if pd.notna(o):
                first = global_stats.object_first_ts.get(o, 0)
                if (ts_nanos[i] - first) < 3600e9:
                    object_is_new[i] = 1.0
    else:
        obj_first = {}
        for i in range(n):
            o = obj_uuids[i]
            if pd.notna(o):
                if o not in obj_first:
                    obj_first[o] = ts_nanos[i]
                    object_is_new[i] = 1.0
                elif (ts_nanos[i] - obj_first[o]) < 3600e9:
                    object_is_new[i] = 1.0

    # 9. Has predicate_object_path
    has_path = events_df[
        "predicate_object_path"
    ].notna().astype(np.float32).values

    # Combine
    X_parts = [
        event_type_dummies.values.astype(np.float32),
        hour.reshape(-1, 1),
        he_size.reshape(-1, 1),
        type_rarity.reshape(-1, 1),
        event_size.reshape(-1, 1),
        np.nan_to_num(time_gap, nan=-1.0).astype(np.float32).reshape(-1, 1),
        subject_is_new.reshape(-1, 1),
        object_is_new.reshape(-1, 1),
        has_path.reshape(-1, 1),
    ]

    feature_names = (
        list(event_type_dummies.columns) +
        ["hour", "he_size", "type_rarity", "event_size",
         "time_gap_same_subject", "subject_is_new", "object_is_new",
         "has_path"]
    )

    X = np.hstack(X_parts).astype(np.float32)

    del event_type_dummies, type_freq
    gc.collect()

    return X, feature_names, last_ts_out
