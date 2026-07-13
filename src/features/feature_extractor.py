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
    # Object UUID -> total event count across all shards
    object_event_counts: dict = field(default_factory=dict)


def _process_single_shard(f):
    import pandas as pd
    import numpy as np
    from collections import Counter
    import gc
    
    type_counter = Counter()
    df_type = pd.read_parquet(f, columns=["type"])
    for t, c in df_type["type"].value_counts().items():
        type_counter[t] += c
    total_events = len(df_type)
    del df_type
    
    df_sub = pd.read_parquet(f, columns=["subject_uuid", "timestamp_nanos"])
    shard_sub_first = df_sub.groupby("subject_uuid")["timestamp_nanos"].min()
    del df_sub
    
    obj_firsts = []
    for col in ["predicate_object_uuid", "predicate_object2_uuid"]:
        df_obj = pd.read_parquet(f, columns=[col, "timestamp_nanos"])
        df_obj = df_obj.dropna(subset=[col])
        if len(df_obj) > 0:
            obj_firsts.append(df_obj.groupby(col)["timestamp_nanos"].min())
        del df_obj
        
    if obj_firsts:
        if len(obj_firsts) == 2:
            combined = pd.concat(obj_firsts)
            shard_obj_first = combined.groupby(combined.index).min()
        else:
            shard_obj_first = obj_firsts[0]
    else:
        shard_obj_first = pd.Series(dtype=np.float64)
        
    gc.collect()
    return type_counter, total_events, shard_sub_first, shard_obj_first

def compute_global_stats(labeled_dir) -> GlobalStats:
    """
    Compute corpus-wide statistics by scanning all labeled shards.
    Uses ProcessPoolExecutor for parallel processing and avoids O(N^2) merge.
    """
    from pathlib import Path
    from collections import Counter
    import concurrent.futures
    import gc

    labeled_dir = Path(labeled_dir)
    def _extract_idx(f):
        return int(f.name.replace("labeled_shard", "").replace(".parquet", ""))
    files = sorted(labeled_dir.glob("labeled_shard*.parquet"), key=_extract_idx)
    if not files:
        raise FileNotFoundError(f"No labeled shards in {labeled_dir}")

    stats = GlobalStats()
    type_counter = Counter()

    print(f"  Computing global stats from {len(files)} shards (parallelized)...")

    sub_first_list = []
    obj_first_list = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(_process_single_shard, f): f for f in files}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            f = futures[future]
            try:
                t_counter, t_events, s_sub_first, s_obj_first = future.result()
                type_counter.update(t_counter)
                stats.total_events += t_events
                sub_first_list.append(s_sub_first)
                obj_first_list.append(s_obj_first)
                print(f"    Finished {f.stem} ({i+1}/{len(files)})")
            except Exception as exc:
                print(f"    {f.stem} generated an exception: {exc}")

    print("  Merging results (this should be fast)...")
    if sub_first_list:
        combined_sub = pd.concat(sub_first_list)
        stats.subject_first_ts = combined_sub.groupby(combined_sub.index).min().to_dict()
        del combined_sub
        
    if obj_first_list:
        combined_obj = pd.concat(obj_first_list)
        stats.object_first_ts = combined_obj.groupby(combined_obj.index).min().to_dict()
        del combined_obj
        
    stats.type_counts = dict(type_counter)

    del sub_first_list, obj_first_list
    gc.collect()

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
    def _extract_idx(f):
        return int(f.name.replace("labeled_shard", "").replace(".parquet", ""))
    files = sorted(labeled_dir.glob("labeled_shard*.parquet"), key=_extract_idx)

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
    objects_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract per-hyperedge feature matrix from events DataFrame.

    Features (35 total):
        - Event type one-hot (~18 types)
        - hour: hour of day
        - he_size: number of non-null entity references (2 or 3)
        - type_rarity: 1 - (freq of this type / total events)
        - event_size: CDM event size field
        - time_gap_same_subject: seconds since last event by same subject
        - subject_is_new: 1 if subject first seen within last hour
        - object_is_new: 1 if object first seen within last hour
        - has_path: 1 if predicate_object_path is non-null
        - obj_is_file: 1 if predicate_object is FILE type
        - obj_is_netflow: 1 if predicate_object is NETFLOW type
        - obj_is_memory: 1 if predicate_object is MEMORY type
        - path_depth: number of '/' segments in predicate_object_path
        - path_has_tmp: 1 if path contains /tmp or /var/tmp
        - path_has_home: 1 if path contains /home/
        - path_has_log: 1 if path contains /var/log or /log/
        - obj_event_count: log(count+1) of events referencing this object
        - is_new_pair: 1 if (subject, object) pair not seen earlier in shard

    Args:
        events_df: DataFrame sorted by timestamp_nanos.
        global_stats: Pre-computed corpus stats. If None, computes per-shard.
        subject_last_ts_carry: Dict mapping subject_uuid -> last timestamp
            from a previous shard, used to seed time_gap computation.
        objects_df: DataFrame with object metadata (uuid, object_type, etc.).
            If None, object type features default to 0.

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

    # 6. Time gap from previous event by same subject (vectorized)
    ts_nanos = events_df["timestamp_nanos"].values.astype(np.float64)
    subject_uuids = events_df["subject_uuid"].values

    # Build a temporary DataFrame for groupby operations
    _tmp = pd.DataFrame({
        "subject_uuid": subject_uuids,
        "ts": ts_nanos,
    })

    # Compute time gap as diff within each subject group
    _tmp["prev_ts"] = _tmp.groupby("subject_uuid")["ts"].shift(1)

    # Seed first events with carry-over from previous shard
    if subject_last_ts_carry:
        first_event_mask = _tmp["prev_ts"].isna() & _tmp["subject_uuid"].notna()
        if first_event_mask.any():
            carry_ts = _tmp.loc[first_event_mask, "subject_uuid"].map(
                subject_last_ts_carry)
            _tmp.loc[first_event_mask, "prev_ts"] = carry_ts

    time_gap = ((_tmp["ts"] - _tmp["prev_ts"]) / 1e9).values

    # Record last timestamp per subject (for next shard's carry)
    last_ts_out = _tmp.groupby("subject_uuid")["ts"].last().to_dict()
    del _tmp

    # 7. Subject is "new" (first seen within last hour) — vectorized
    subject_is_new = np.zeros(n, dtype=np.float32)
    if global_stats is not None:
        # Use corpus-wide first-seen
        sub_series = events_df["subject_uuid"]
        sub_first_ts = sub_series.map(global_stats.subject_first_ts)
        valid_sub = sub_first_ts.notna()
        age = ts_nanos - sub_first_ts.values.astype(np.float64)
        subject_is_new[valid_sub.values & (age < 3600e9)] = 1.0
        del sub_first_ts, valid_sub, age
    else:
        # Per-shard fallback: first occurrence per subject
        sub_df = pd.DataFrame({
            "subject_uuid": events_df["subject_uuid"].values,
            "ts": ts_nanos,
        })
        sub_first = sub_df.groupby("subject_uuid")["ts"].transform("first")
        age = sub_df["ts"] - sub_first
        valid = sub_df["subject_uuid"].notna()
        subject_is_new[valid.values & (age.values < 3600e9)] = 1.0
        del sub_df, sub_first, age

    # 8. Object is "new" — vectorized
    obj_uuids = events_df["predicate_object_uuid"].values
    object_is_new = np.zeros(n, dtype=np.float32)
    if global_stats is not None:
        obj_series = events_df["predicate_object_uuid"]
        obj_first_ts = obj_series.map(global_stats.object_first_ts)
        valid_obj = obj_first_ts.notna()
        age = ts_nanos - obj_first_ts.values.astype(np.float64)
        object_is_new[valid_obj.values & (age < 3600e9)] = 1.0
        del obj_first_ts, valid_obj, age
    else:
        obj_df = pd.DataFrame({
            "obj_uuid": obj_uuids,
            "ts": ts_nanos,
        })
        obj_df = obj_df[obj_df["obj_uuid"].notna()]
        obj_first = obj_df.groupby("obj_uuid")["ts"].transform("first")
        age = obj_df["ts"] - obj_first
        valid_idx = obj_df.index[age.values < 3600e9]
        object_is_new[valid_idx] = 1.0
        del obj_df, obj_first, age

    # 9. Has predicate_object_path
    has_path = events_df[
        "predicate_object_path"
    ].notna().astype(np.float32).values

    # ------------------------------------------------------------------
    # Object-aware features (10-18)
    # ------------------------------------------------------------------
    obj_uuid_col = events_df["predicate_object_uuid"]

    # 10-12. Object type indicators (file / netflow / memory)
    obj_is_file = np.zeros(n, dtype=np.float32)
    obj_is_netflow = np.zeros(n, dtype=np.float32)
    obj_is_memory = np.zeros(n, dtype=np.float32)

    if objects_df is not None:
        # Build uuid -> object_type mapping (vectorized)
        obj_type_map = pd.Series(
            objects_df["object_type"].astype(str).values,
            index=objects_df["uuid"].values,
        )
        obj_types = obj_uuid_col.map(obj_type_map).fillna("UNKNOWN")
        obj_is_file = (obj_types == "FILE").values.astype(np.float32)
        obj_is_netflow = (obj_types == "NETFLOW").values.astype(np.float32)
        obj_is_memory = (obj_types == "MEMORY").values.astype(np.float32)
        del obj_type_map, obj_types

    # 13. Path depth (number of '/' segments)
    path_col = events_df["predicate_object_path"].fillna("").astype(str)
    path_depth = path_col.str.count("/").values.astype(np.float32)

    # 14-16. Path content indicators
    path_has_tmp = (
        path_col.str.contains("/tmp", na=False)
    ).values.astype(np.float32)
    path_has_home = (
        path_col.str.contains("/home/", na=False)
    ).values.astype(np.float32)
    path_has_log = (
        path_col.str.contains("/var/log|/log/", regex=True, na=False)
    ).values.astype(np.float32)

    del path_col

    # 17. Object event count: log(count + 1) per predicate_object_uuid
    obj_counts = obj_uuid_col.map(
        obj_uuid_col.value_counts()
    ).fillna(0).values.astype(np.float32)
    obj_event_count = np.log1p(obj_counts)
    del obj_counts

    # 18. Is new (subject, object) pair in this shard
    pair_series = events_df["subject_uuid"].astype(str) + "||" + obj_uuid_col.astype(str)
    is_new_pair = (~pair_series.duplicated(keep="first")).values.astype(np.float32)
    del pair_series

    gc.collect()

    # ------------------------------------------------------------------
    # Combine all features
    # ------------------------------------------------------------------
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
        obj_is_file.reshape(-1, 1),
        obj_is_netflow.reshape(-1, 1),
        obj_is_memory.reshape(-1, 1),
        path_depth.reshape(-1, 1),
        path_has_tmp.reshape(-1, 1),
        path_has_home.reshape(-1, 1),
        path_has_log.reshape(-1, 1),
        obj_event_count.reshape(-1, 1),
        is_new_pair.reshape(-1, 1),
    ]

    feature_names = (
        list(event_type_dummies.columns) +
        ["hour", "he_size", "type_rarity", "event_size",
         "time_gap_same_subject", "subject_is_new", "object_is_new",
         "has_path",
         "obj_is_file", "obj_is_netflow", "obj_is_memory",
         "path_depth", "path_has_tmp", "path_has_home", "path_has_log",
         "obj_event_count", "is_new_pair"]
    )

    X = np.hstack(X_parts).astype(np.float32)

    del event_type_dummies, type_freq
    gc.collect()

    return X, feature_names, last_ts_out

