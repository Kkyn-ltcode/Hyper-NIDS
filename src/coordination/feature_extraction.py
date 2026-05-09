"""
Feature extraction for the kill switch experiment.

Extracts features at three levels (identity-excluded):
  - Individual: per-event features (12 features)
  - Pairwise: per-consecutive-pair features (30 features)
  - Group: per-time-window features (18 features)

Identity-excluded means NO raw UUIDs as features. We use event type,
temporal patterns, structural indicators, and aggregate statistics only.

Usage:
    python -m src.coordination.feature_extraction
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.coordination.cd_measure import compute_cd_components


# ============================================================
# Event type classification helpers
# ============================================================

READ_TYPES = {"EVENT_READ", "EVENT_RECVFROM", "EVENT_RECVMSG", "EVENT_READ_SOCKET_PARAMS"}
WRITE_TYPES = {"EVENT_WRITE", "EVENT_SENDTO", "EVENT_SENDMSG", "EVENT_WRITE_SOCKET_PARAMS"}
EXEC_TYPES = {"EVENT_EXECUTE", "EVENT_CLONE", "EVENT_FORK"}
NETWORK_TYPES = {
    "EVENT_CONNECT", "EVENT_ACCEPT", "EVENT_SENDTO", "EVENT_RECVFROM",
    "EVENT_SENDMSG", "EVENT_RECVMSG", "EVENT_READ_SOCKET_PARAMS",
    "EVENT_WRITE_SOCKET_PARAMS",
}
FILE_TYPES = {
    "EVENT_READ", "EVENT_WRITE", "EVENT_OPEN", "EVENT_CLOSE",
    "EVENT_UNLINK", "EVENT_RENAME", "EVENT_MODIFY_FILE_ATTRIBUTES",
}


def _compute_event_type_rarity(events_df: pd.DataFrame) -> dict[str, float]:
    """Compute rarity score for each event type: -log2(frequency)."""
    counts = events_df["type"].value_counts()
    total = counts.sum()
    rarity = {}
    for etype, count in counts.items():
        freq = count / total
        rarity[etype] = -np.log2(freq) if freq > 0 else 0.0
    return rarity


# ============================================================
# Individual features (per event, 12 features)
# ============================================================

def extract_individual_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 12 identity-excluded features per event.

    Features:
        0. event_type_code:    Ordinal encoding of event type
        1. is_read:            Read-class event (binary)
        2. is_write:           Write-class event (binary)
        3. is_exec:            Execution-class event (binary)
        4. is_network:         Network-class event (binary)
        5. is_file:            File-class event (binary)
        6. hour_of_day:        Hour from timestamp (0-23)
        7. time_since_start:   Seconds since first event
        8. event_size:         Size field (0 if null)
        9. has_pred_obj2:      Has secondary object (binary)
        10. event_rarity:      -log2(type_frequency)
        11. has_name:          Has non-null name field (binary)
    """
    types = events_df["type"].astype(str)
    type_codes = types.astype("category").cat.codes

    rarity_map = _compute_event_type_rarity(events_df)

    ts = events_df["timestamp_nanos"].values.astype(np.float64)
    t_start = ts[np.isfinite(ts)].min() if len(ts) > 0 else 0
    time_since_start = (ts - t_start) / 1e9  # seconds

    # Hour of day from nanosecond timestamps
    timestamps_dt = pd.to_datetime(ts, unit="ns", errors="coerce")
    hour = timestamps_dt.hour.fillna(0).astype(np.float32)

    features = pd.DataFrame({
        "event_type_code": type_codes.values,
        "is_read": types.isin(READ_TYPES).astype(np.float32).values,
        "is_write": types.isin(WRITE_TYPES).astype(np.float32).values,
        "is_exec": types.isin(EXEC_TYPES).astype(np.float32).values,
        "is_network": types.isin(NETWORK_TYPES).astype(np.float32).values,
        "is_file": types.isin(FILE_TYPES).astype(np.float32).values,
        "hour_of_day": hour.values,
        "time_since_start": time_since_start.astype(np.float32),
        "event_size": events_df["size"].fillna(0).astype(np.float32).values,
        "has_pred_obj2": events_df["predicate_object2_uuid"].notna().astype(np.float32).values,
        "event_rarity": types.map(rarity_map).fillna(0).astype(np.float32).values,
        "has_name": events_df["name"].notna().astype(np.float32).values,
    })

    return features


# ============================================================
# Pairwise features (per consecutive event pair, 30 features)
# ============================================================

def extract_pairwise_features(
    events_df: pd.DataFrame,
    individual_features: pd.DataFrame,
    max_pairs: int = 0,
) -> pd.DataFrame:
    """
    Extract 30 features per consecutive event pair.

    Features = individual_i (12) + individual_j (12) + interaction (6):
        24-29: time_gap, same_type, same_subject, same_object,
               type_pair_code, size_ratio

    Args:
        events_df: Events DataFrame sorted by timestamp.
        individual_features: Output from extract_individual_features.
        max_pairs: Max pairs to extract (0 = all). Use for memory control.

    Returns:
        DataFrame with 30 columns, one row per consecutive pair.
    """
    n = len(events_df)
    if n < 2:
        return pd.DataFrame()

    if max_pairs > 0 and max_pairs < n - 1:
        # Random sample of consecutive pair indices
        rng = np.random.default_rng(42)
        pair_starts = rng.choice(n - 1, size=max_pairs, replace=False)
        pair_starts.sort()
    else:
        pair_starts = np.arange(n - 1)

    # Individual features for event i and j
    feat_i = individual_features.iloc[pair_starts].reset_index(drop=True)
    feat_j = individual_features.iloc[pair_starts + 1].reset_index(drop=True)

    feat_i.columns = [f"{c}_i" for c in feat_i.columns]
    feat_j.columns = [f"{c}_j" for c in feat_j.columns]

    # Interaction features
    ts = events_df["timestamp_nanos"].values.astype(np.float64)
    time_gap = (ts[pair_starts + 1] - ts[pair_starts]) / 1e9  # seconds

    types = events_df["type"].values.astype(str)
    same_type = (types[pair_starts] == types[pair_starts + 1]).astype(np.float32)

    subjects = events_df["subject_uuid"].values.astype(str)
    same_subject = (subjects[pair_starts] == subjects[pair_starts + 1]).astype(np.float32)

    objects = events_df["predicate_object_uuid"].values.astype(str)
    same_object = (objects[pair_starts] == objects[pair_starts + 1]).astype(np.float32)

    # Type pair encoding (ordinal of concatenated types)
    type_pairs = np.char.add(np.char.add(types[pair_starts], "_"), types[pair_starts + 1])
    type_pair_codes = pd.Categorical(type_pairs).codes.astype(np.float32)

    sizes = events_df["size"].fillna(0).values.astype(np.float64)
    size_j = sizes[pair_starts + 1]
    size_i = sizes[pair_starts]
    # Safe ratio: avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        size_ratio = np.where(size_j > 0, size_i / size_j, 0.0).astype(np.float32)
    size_ratio = np.nan_to_num(size_ratio, nan=0.0, posinf=0.0, neginf=0.0)

    interaction = pd.DataFrame({
        "time_gap": time_gap.astype(np.float32),
        "same_type": same_type,
        "same_subject": same_subject,
        "same_object": same_object,
        "type_pair_code": type_pair_codes,
        "size_ratio": size_ratio,
    })

    return pd.concat([feat_i, feat_j, interaction], axis=1)


# ============================================================
# Group features (per time window, 18 features)
# ============================================================

def extract_group_features(
    events_df: pd.DataFrame,
    window_seconds: float = 300.0,
    min_events: int = 10,
) -> pd.DataFrame:
    """
    Extract 18 features per time window.

    Features:
        0-3:   tc, bd, es, cd (Coordination Degree components)
        4-7:   n_events, n_subjects, n_objects, n_event_types
        8-11:  frac_read, frac_write, frac_exec, frac_network
        12-13: mean_time_gap, std_time_gap
        14-15: mean_event_size, max_event_size
        16-17: events_per_subject, events_per_object
    """
    if len(events_df) == 0:
        return pd.DataFrame()

    df = events_df.sort_values("timestamp_nanos").copy()
    ts = df["timestamp_nanos"].values.astype(np.float64)
    window_nanos = int(window_seconds * 1e9)
    t_min, t_max = ts.min(), ts.max()
    window_starts = np.arange(t_min, t_max + 1, window_nanos)

    types = df["type"].values.astype(str)
    subjects = df["subject_uuid"].values.astype(str)
    objects = df["predicate_object_uuid"].values.astype(str)
    sizes = df["size"].fillna(0).values.astype(np.float64)

    rows = []
    for w_start in window_starts:
        w_end = w_start + window_nanos
        mask = (ts >= w_start) & (ts < w_end)
        n = mask.sum()

        if n < min_events:
            continue

        w_ts = ts[mask]
        w_types = types[mask]
        w_subjects = subjects[mask]
        w_objects = objects[mask]
        w_sizes = sizes[mask]

        # CD components
        w_df = df[mask]
        cd_result = compute_cd_components(w_df)

        # Type fractions
        n_total = float(n)
        frac_read = np.isin(w_types, list(READ_TYPES)).sum() / n_total
        frac_write = np.isin(w_types, list(WRITE_TYPES)).sum() / n_total
        frac_exec = np.isin(w_types, list(EXEC_TYPES)).sum() / n_total
        frac_network = np.isin(w_types, list(NETWORK_TYPES)).sum() / n_total

        # Time gap stats
        sorted_ts = np.sort(w_ts)
        delta_t = np.diff(sorted_ts).astype(np.float64) / 1e9  # seconds
        mean_gap = np.mean(delta_t) if len(delta_t) > 0 else 0.0
        std_gap = np.std(delta_t) if len(delta_t) > 1 else 0.0

        # Size stats
        mean_size = np.mean(w_sizes)
        max_size = np.max(w_sizes)

        # Entity ratios
        n_subj = len(np.unique(w_subjects))
        n_obj = len(np.unique(w_objects))
        eps = events_per_subject = n / max(n_subj, 1)
        epo = events_per_object = n / max(n_obj, 1)

        rows.append({
            "window_start": w_start,
            "window_end": w_end,
            "tc": cd_result["tc"],
            "bd": cd_result["bd"],
            "es": cd_result["es"],
            "cd": cd_result["cd"],
            "n_events": n,
            "n_subjects": n_subj,
            "n_objects": n_obj,
            "n_event_types": cd_result["n_event_types"],
            "frac_read": frac_read,
            "frac_write": frac_write,
            "frac_exec": frac_exec,
            "frac_network": frac_network,
            "mean_time_gap": mean_gap,
            "std_time_gap": std_gap,
            "mean_event_size": mean_size,
            "max_event_size": max_size,
            "events_per_subject": eps,
            "events_per_object": epo,
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["window_start_dt"] = pd.to_datetime(result["window_start"], unit="ns")
    return result


# ============================================================
# Feature column names (for reference)
# ============================================================

INDIVIDUAL_FEATURE_NAMES = [
    "event_type_code", "is_read", "is_write", "is_exec", "is_network",
    "is_file", "hour_of_day", "time_since_start", "event_size",
    "has_pred_obj2", "event_rarity", "has_name",
]

GROUP_FEATURE_NAMES = [
    "tc", "bd", "es", "cd", "n_events", "n_subjects", "n_objects",
    "n_event_types", "frac_read", "frac_write", "frac_exec", "frac_network",
    "mean_time_gap", "std_time_gap", "mean_event_size", "max_event_size",
    "events_per_subject", "events_per_object",
]


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    import time

    processed_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "darpa_tc_e3" / "theia"
    events_path = processed_dir / "events.parquet"

    if not events_path.exists():
        print(f"Run parser first: python -m src.data.darpa_tc_parser --shards 0")
        exit(1)

    print("Loading events...")
    events_df = pd.read_parquet(events_path)
    print(f"  {len(events_df):,} events\n")

    # --- Individual features ---
    print("Extracting individual features (12 per event)...")
    t0 = time.time()
    indiv = extract_individual_features(events_df)
    print(f"  Shape: {indiv.shape}, Time: {time.time()-t0:.1f}s")
    print(f"  Columns: {list(indiv.columns)}")
    print(f"  Sample:\n{indiv.head(3).to_string()}\n")

    # --- Pairwise features (sample 500K pairs) ---
    print("Extracting pairwise features (30 per pair, 500K sample)...")
    t0 = time.time()
    pairs = extract_pairwise_features(events_df, indiv, max_pairs=500_000)
    print(f"  Shape: {pairs.shape}, Time: {time.time()-t0:.1f}s")
    print(f"  Columns: {list(pairs.columns)}\n")

    # --- Group features (5-min windows) ---
    print("Extracting group features (18 per window, 5-min windows)...")
    t0 = time.time()
    groups = extract_group_features(events_df, window_seconds=300)
    print(f"  Shape: {groups.shape}, Time: {time.time()-t0:.1f}s")
    print(f"  Columns: {list(groups.columns)}")
    if len(groups) > 0:
        print(f"\n  Group feature stats:")
        for col in GROUP_FEATURE_NAMES:
            if col in groups.columns:
                vals = groups[col]
                print(f"    {col:20s}  mean={vals.mean():10.4f}  std={vals.std():10.4f}")

    print(f"\n✓ Feature extraction verified.")
    print(f"  Individual: {indiv.shape[1]} features × {indiv.shape[0]:,} events")
    print(f"  Pairwise:   {pairs.shape[1]} features × {pairs.shape[0]:,} pairs")
    print(f"  Group:      {groups.shape[1]} features × {groups.shape[0]:,} windows")
