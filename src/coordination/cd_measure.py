"""
Coordination Degree (CD) computation.

CD = TC × BD × ES where:
  TC (Temporal Clustering):  Burstiness of event timing
  BD (Behavioral Diversity): Diversity of action types
  ES (Entity Spread):        Diversity of entities involved

Each component is bounded in [0, 1].
CD = 0 if ANY component is 0 (unanimity axiom).
CD increases monotonically in each component (monotonicity axiom).
CD ∈ [0, 1] (boundedness axiom).

This module computes CD for arbitrary groups of events.
A "group" is any set of events (typically a time window or a
set of events linked by provenance).

References:
    - Burstiness: Goh & Barabási, "Burstiness and memory in complex
      systems," EPL 2008.
    - Normalized entropy: standard information-theoretic measure.

Usage:
    from src.coordination.cd_measure import compute_cd, compute_cd_components

    # Compute CD for a DataFrame of events
    tc, bd, es, cd = compute_cd_components(events_df)

    # Compute CD for each time window
    window_cds = compute_cd_windowed(events_df, window_seconds=300)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ============================================================
# Core component functions
# ============================================================

def temporal_clustering(timestamps_nanos: np.ndarray) -> float:
    """
    Compute Temporal Clustering (TC) using the burstiness parameter.

    Uses the Goh-Barabási burstiness coefficient:
        B = (σ_Δt - μ_Δt) / (σ_Δt + μ_Δt)
    where Δt are inter-event times.

    B ∈ [-1, 1]:
        B = -1: perfectly periodic (no burstiness)
        B =  0: Poisson random process
        B =  1: maximally bursty

    We map to [0, 1]:
        TC = (B + 1) / 2

    Args:
        timestamps_nanos: Sorted array of event timestamps in nanoseconds.

    Returns:
        TC ∈ [0, 1]. Returns 0.0 if fewer than 2 events.
    """
    if len(timestamps_nanos) < 2:
        return 0.0

    # Compute inter-event times
    sorted_ts = np.sort(timestamps_nanos)
    delta_t = np.diff(sorted_ts).astype(np.float64)

    # Remove zero inter-event times (simultaneous events)
    delta_t = delta_t[delta_t > 0]

    if len(delta_t) < 2:
        return 0.0

    mu = np.mean(delta_t)
    sigma = np.std(delta_t, ddof=1)  # sample std

    # Handle edge case: all inter-event times identical
    if mu + sigma == 0:
        return 0.0

    # Burstiness coefficient B ∈ [-1, 1]
    B = (sigma - mu) / (sigma + mu)

    # Map to [0, 1]
    tc = (B + 1.0) / 2.0

    return float(np.clip(tc, 0.0, 1.0))


def behavioral_diversity(event_types: np.ndarray) -> float:
    """
    Compute Behavioral Diversity (BD) as normalized Shannon entropy
    of the event type distribution.

        BD = H(types) / log(K)

    where K is the number of distinct event types observed.

    BD = 0: all events are the same type (no diversity)
    BD = 1: all event types equally frequent (maximum diversity)

    Args:
        event_types: Array of event type labels (strings or categorical).

    Returns:
        BD ∈ [0, 1]. Returns 0.0 if fewer than 2 distinct types.
    """
    if len(event_types) == 0:
        return 0.0

    # Count occurrences of each type
    _, counts = np.unique(event_types, return_counts=True)
    K = len(counts)

    if K <= 1:
        return 0.0  # Only one type → no diversity

    # Compute probabilities
    probs = counts / counts.sum()

    # Shannon entropy
    H = -np.sum(probs * np.log2(probs))

    # Normalize by maximum possible entropy
    H_max = np.log2(K)

    bd = H / H_max

    return float(np.clip(bd, 0.0, 1.0))


def entity_spread(subject_uuids: np.ndarray, object_uuids: np.ndarray) -> float:
    """
    Compute Entity Spread (ES) as normalized Shannon entropy of the
    entity participation distribution.

    We treat each unique entity (subject or object UUID) as a category,
    count how many events involve each entity, and compute normalized entropy.

        ES = H(entities) / log(N)

    where N is the number of distinct entities.

    ES = 0: all events involve the same entity (no spread)
    ES = 1: events spread uniformly across entities (maximum spread)

    Args:
        subject_uuids: Array of subject UUIDs from events.
        object_uuids: Array of object UUIDs from events.

    Returns:
        ES ∈ [0, 1]. Returns 0.0 if fewer than 2 distinct entities.
    """
    if len(subject_uuids) == 0:
        return 0.0

    # Combine subjects and objects into a single entity list
    # Each event contributes both its subject and its object
    all_entities = np.concatenate([
        subject_uuids.astype(str),
        object_uuids.astype(str),
    ])

    # Remove null/None/empty entries
    mask = (all_entities != "None") & (all_entities != "") & (all_entities != "nan")
    all_entities = all_entities[mask]

    if len(all_entities) == 0:
        return 0.0

    # Count occurrences of each entity
    _, counts = np.unique(all_entities, return_counts=True)
    N = len(counts)

    if N <= 1:
        return 0.0

    # Compute probabilities
    probs = counts / counts.sum()

    # Shannon entropy
    H = -np.sum(probs * np.log2(probs))

    # Normalize
    H_max = np.log2(N)

    es = H / H_max

    return float(np.clip(es, 0.0, 1.0))


# ============================================================
# CD computation
# ============================================================

def compute_cd_components(
    events_df: pd.DataFrame,
    timestamp_col: str = "timestamp_nanos",
    type_col: str = "type",
    subject_col: str = "subject_uuid",
    object_col: str = "predicate_object_uuid",
) -> dict:
    """
    Compute all CD components for a group of events.

    Args:
        events_df: DataFrame of events (must have the specified columns).
        timestamp_col: Column name for timestamps (nanoseconds).
        type_col: Column name for event type.
        subject_col: Column name for subject UUID.
        object_col: Column name for object UUID.

    Returns:
        Dictionary with keys: 'tc', 'bd', 'es', 'cd', 'n_events',
        'n_subjects', 'n_objects', 'n_event_types'.
    """
    if len(events_df) == 0:
        return {
            "tc": 0.0, "bd": 0.0, "es": 0.0, "cd": 0.0,
            "n_events": 0, "n_subjects": 0, "n_objects": 0, "n_event_types": 0,
        }

    # Extract arrays
    timestamps = events_df[timestamp_col].dropna().values
    event_types = events_df[type_col].dropna().values.astype(str)
    subjects = events_df[subject_col].dropna().values
    objects = events_df[object_col].dropna().values

    # Compute components
    tc = temporal_clustering(timestamps)
    bd = behavioral_diversity(event_types)
    es = entity_spread(subjects, objects)

    # CD = TC × BD × ES (product aggregation)
    cd = tc * bd * es

    return {
        "tc": round(tc, 6),
        "bd": round(bd, 6),
        "es": round(es, 6),
        "cd": round(cd, 6),
        "n_events": len(events_df),
        "n_subjects": len(np.unique(subjects)),
        "n_objects": len(np.unique(objects)),
        "n_event_types": len(np.unique(event_types)),
    }


def compute_cd_windowed(
    events_df: pd.DataFrame,
    window_seconds: float = 300.0,
    timestamp_col: str = "timestamp_nanos",
    type_col: str = "type",
    subject_col: str = "subject_uuid",
    object_col: str = "predicate_object_uuid",
    min_events: int = 10,
) -> pd.DataFrame:
    """
    Compute CD for each non-overlapping time window.

    Args:
        events_df: DataFrame of events sorted by timestamp.
        window_seconds: Window duration in seconds.
        timestamp_col: Column name for timestamps (nanoseconds).
        type_col: Column name for event type.
        subject_col: Column name for subject UUID.
        object_col: Column name for object UUID.
        min_events: Minimum events per window to compute CD
                    (windows with fewer events get CD=0).

    Returns:
        DataFrame with one row per window, columns:
        window_start, window_end, tc, bd, es, cd, n_events,
        n_subjects, n_objects, n_event_types.
    """
    if len(events_df) == 0:
        return pd.DataFrame()

    # Ensure sorted by timestamp
    df = events_df.sort_values(timestamp_col).copy()
    timestamps = df[timestamp_col].values

    # Define window boundaries
    window_nanos = int(window_seconds * 1e9)
    t_min = timestamps.min()
    t_max = timestamps.max()

    window_starts = np.arange(t_min, t_max + 1, window_nanos)

    results = []
    for w_start in window_starts:
        w_end = w_start + window_nanos

        # Select events in this window
        mask = (timestamps >= w_start) & (timestamps < w_end)
        window_events = df[mask]

        if len(window_events) < min_events:
            cd_result = {
                "tc": 0.0, "bd": 0.0, "es": 0.0, "cd": 0.0,
                "n_events": len(window_events),
                "n_subjects": 0, "n_objects": 0, "n_event_types": 0,
            }
        else:
            cd_result = compute_cd_components(
                window_events,
                timestamp_col=timestamp_col,
                type_col=type_col,
                subject_col=subject_col,
                object_col=object_col,
            )

        cd_result["window_start"] = w_start
        cd_result["window_end"] = w_end
        results.append(cd_result)

    result_df = pd.DataFrame(results)

    # Add human-readable timestamps
    result_df["window_start_dt"] = pd.to_datetime(result_df["window_start"], unit="ns")
    result_df["window_end_dt"] = pd.to_datetime(result_df["window_end"], unit="ns")

    # Reorder columns for readability
    cols = [
        "window_start_dt", "window_end_dt",
        "tc", "bd", "es", "cd",
        "n_events", "n_subjects", "n_objects", "n_event_types",
        "window_start", "window_end",
    ]
    result_df = result_df[cols]

    return result_df


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    """Quick test: compute CD on the parsed Theia E3 data."""
    import time
    from pathlib import Path

    processed_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "darpa_tc_e3" / "theia"

    events_path = processed_dir / "events.parquet"
    if not events_path.exists():
        print(f"Events parquet not found at {events_path}")
        print("Run: python -m src.data.darpa_tc_parser --shards 0")
        exit(1)

    print("Loading events...")
    events_df = pd.read_parquet(events_path)
    print(f"  {len(events_df):,} events loaded")

    # Test 1: Global CD (all events)
    print(f"\n{'='*60}")
    print("Test 1: Global CD (all 4.7M events)")
    print(f"{'='*60}")
    t0 = time.time()
    result = compute_cd_components(events_df)
    elapsed = time.time() - t0
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"  Computed in {elapsed:.2f}s")

    # Test 2: Windowed CD (5-minute windows)
    print(f"\n{'='*60}")
    print("Test 2: Windowed CD (5-minute windows)")
    print(f"{'='*60}")
    t0 = time.time()
    window_df = compute_cd_windowed(events_df, window_seconds=300)
    elapsed = time.time() - t0
    print(f"  {len(window_df)} windows computed in {elapsed:.2f}s")

    if len(window_df) > 0:
        print(f"\n  CD statistics across windows:")
        for col in ["tc", "bd", "es", "cd", "n_events"]:
            vals = window_df[col]
            print(f"    {col:12s}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}")

        print(f"\n  Top 5 windows by CD:")
        top5 = window_df.nlargest(5, "cd")
        for _, row in top5.iterrows():
            print(f"    {row['window_start_dt']} | TC={row['tc']:.3f} BD={row['bd']:.3f} "
                  f"ES={row['es']:.3f} CD={row['cd']:.3f} | {int(row['n_events']):,} events")

        print(f"\n  Bottom 5 windows by CD (non-zero):")
        nonzero = window_df[window_df["cd"] > 0]
        if len(nonzero) > 0:
            bot5 = nonzero.nsmallest(5, "cd")
            for _, row in bot5.iterrows():
                print(f"    {row['window_start_dt']} | TC={row['tc']:.3f} BD={row['bd']:.3f} "
                      f"ES={row['es']:.3f} CD={row['cd']:.3f} | {int(row['n_events']):,} events")

    print(f"\n✓ CD computation verified.")
