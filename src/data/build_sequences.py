"""
Per-subject event sequence builder for temporal modeling.

Groups events by subject UUID, orders by timestamp, and returns
the sequences needed by the Mamba/GRU temporal encoder.

Usage:
    from src.data.build_sequences import build_subject_sequences
    sequences = build_subject_sequences(events_df)
"""

import numpy as np
import pandas as pd
from collections import defaultdict


def build_subject_sequences(
    events_df: pd.DataFrame,
) -> dict[str, list[int]]:
    """
    Build per-subject event sequences.

    Groups events by subject_uuid and orders by timestamp.
    Returns a dict mapping subject UUID to a list of positional
    event indices (into events_df), sorted by time.

    Args:
        events_df: DataFrame with columns [subject_uuid, timestamp_nanos].
                   Must be sorted by timestamp_nanos.

    Returns:
        sequences: dict mapping subject_uuid -> list of event indices
                   (positional indices into events_df)
    """
    sub_uuids = events_df["subject_uuid"].values
    sequences = defaultdict(list)

    for i in range(len(events_df)):
        s = sub_uuids[i]
        if pd.notna(s):
            sequences[s].append(i)

    # Convert to regular dict (already sorted because events_df is
    # sorted by timestamp)
    return dict(sequences)


def sequence_stats(
    sequences: dict[str, list[int]],
) -> dict:
    """Compute summary statistics of per-subject sequences."""
    lengths = [len(seq) for seq in sequences.values()]
    lengths_arr = np.array(lengths)

    return {
        "n_subjects": len(sequences),
        "total_events": sum(lengths),
        "seq_len_mean": lengths_arr.mean(),
        "seq_len_median": np.median(lengths_arr),
        "seq_len_min": lengths_arr.min(),
        "seq_len_max": lengths_arr.max(),
        "seq_len_p95": np.percentile(lengths_arr, 95),
        "seq_len_p99": np.percentile(lengths_arr, 99),
        "subjects_gt_1000": int((lengths_arr > 1000).sum()),
        "subjects_gt_10000": int((lengths_arr > 10000).sum()),
        "subjects_gt_100000": int((lengths_arr > 100000).sum()),
    }
