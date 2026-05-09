"""
Ground truth labeling for DARPA TC E3 Theia dataset.

Labels events as attack/benign based on known IoCs from the official
ground truth report and data-driven discovery on the 1r topic.

Theia E3-1r Attack Scenario:
    The Firefox process (/home/admin/Downloads/firefox/firefox) is the
    exploited entry point. Events involving Firefox-spawned processes
    and their network connections constitute the attack activity.

Labeling strategy (conservative, two-tier):
    Tier 1 (DEFINITE attack): Events by/on known malicious entities
        - Processes: firefox and its children
        - Network: connections to non-local external IPs from firefox
        - Files: files written by firefox-tree processes
    Tier 2 (BENIGN): All other system activity

Usage:
    python -m src.coordination.ground_truth_e3
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Known IoCs for Theia E3-1r
# ============================================================

# The exploited browser (attack entry point)
ATTACK_ENTRY_PROCESS = "/home/admin/Downloads/firefox/firefox"

# Known benign local IPs (not attack targets)
LOCAL_IPS = {"127.0.0.1", "10.0.6.60", "LOCAL", "NA", "NETLINK", ""}


def _build_attack_subject_uuids(
    subjects_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> set:
    """
    Build the set of attack-related subject UUIDs by tracing the
    process tree rooted at Firefox.

    Firefox → child processes (via EVENT_CLONE/FORK) → grandchildren → ...
    All descendants are considered attack-related.
    """
    # Start with Firefox subjects
    firefox_mask = subjects_df["process_path"] == ATTACK_ENTRY_PROCESS
    attack_uuids = set(subjects_df.loc[firefox_mask, "uuid"].values)

    if not attack_uuids:
        return set()

    # Trace children via parent_uuid
    # Iteratively add children until no new ones found
    for _ in range(20):  # Max depth 20 (more than enough)
        children = subjects_df[
            subjects_df["parent_uuid"].isin(attack_uuids)
            & ~subjects_df["uuid"].isin(attack_uuids)
        ]["uuid"].values
        if len(children) == 0:
            break
        attack_uuids.update(children)

    return attack_uuids


def label_events(
    events_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    objects_df: pd.DataFrame = None,
) -> pd.Series:
    """
    Label events as attack (1) or benign (0).

    An event is attack-related if its subject is Firefox or a
    descendant of Firefox in the process tree.

    Args:
        events_df: Events DataFrame.
        subjects_df: Subjects DataFrame.
        objects_df: Objects DataFrame (unused, kept for API compat).

    Returns:
        Integer Series: 1 = attack, 0 = benign.
    """
    attack_uuids = _build_attack_subject_uuids(subjects_df, events_df)

    event_labels = pd.Series(0, index=events_df.index, dtype=np.int8)
    if attack_uuids:
        mask = events_df["subject_uuid"].isin(attack_uuids)
        event_labels[mask] = 1

    return event_labels


def label_windows(
    events_df: pd.DataFrame,
    event_labels: pd.Series,
    window_seconds: float = 300.0,
) -> pd.DataFrame:
    """
    Assign attack/benign labels to time windows.

    A window is "attack" if it contains ANY attack event.
    Also computes attack_fraction for analysis.
    """
    ts = events_df["timestamp_nanos"].values.astype(np.float64)
    window_nanos = int(window_seconds * 1e9)
    t_min, t_max = ts.min(), ts.max()
    window_starts = np.arange(t_min, t_max + 1, window_nanos)

    labels_arr = event_labels.values

    rows = []
    for w_start in window_starts:
        w_end = w_start + window_nanos
        mask = (ts >= w_start) & (ts < w_end)
        n_total = mask.sum()
        if n_total == 0:
            continue
        n_attack = int(labels_arr[mask].sum())
        rows.append({
            "window_start": w_start,
            "window_end": w_end,
            "label": 1 if n_attack > 0 else 0,
            "attack_fraction": n_attack / n_total,
            "n_attack": n_attack,
            "n_total": int(n_total),
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["window_start_dt"] = pd.to_datetime(result["window_start"], unit="ns")
    return result


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    processed_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "data" / "processed" / "darpa_tc_e3" / "theia"
    )

    print("Loading data...")
    events_df = pd.read_parquet(processed_dir / "events.parquet")
    subjects_df = pd.read_parquet(processed_dir / "subjects.parquet")
    objects_df = pd.read_parquet(processed_dir / "objects.parquet")
    print(f"  Events: {len(events_df):,}, Subjects: {len(subjects_df):,}, Objects: {len(objects_df):,}")

    # Build attack subject tree
    attack_uuids = _build_attack_subject_uuids(subjects_df, events_df)
    print(f"\nAttack-related subjects (Firefox tree): {len(attack_uuids)}")

    # Show attack process paths
    attack_subs = subjects_df[subjects_df["uuid"].isin(attack_uuids)]
    print(f"  Unique process paths in attack tree:")
    for path, count in attack_subs["process_path"].value_counts().items():
        print(f"    {path}: {count}")

    # Label events
    print("\nLabeling events...")
    event_labels = label_events(events_df, subjects_df)
    n_attack = (event_labels == 1).sum()
    n_benign = (event_labels == 0).sum()
    print(f"  Attack events: {n_attack:,} ({100*n_attack/len(events_df):.2f}%)")
    print(f"  Benign events: {n_benign:,} ({100*n_benign/len(events_df):.2f}%)")

    # Event type breakdown
    if n_attack > 0:
        print(f"\n  Attack event types:")
        atk = events_df[event_labels == 1]
        for etype, count in atk["type"].value_counts().items():
            pct = 100 * count / n_attack
            print(f"    {etype:30s} {count:>8,} ({pct:5.1f}%)")

        print(f"\n  Benign event types:")
        ben = events_df[event_labels == 0]
        for etype, count in ben["type"].value_counts().head(5).items():
            pct = 100 * count / n_benign
            print(f"    {etype:30s} {count:>8,} ({pct:5.1f}%)")

    # Window labels
    print("\nLabeling 5-min windows...")
    wl = label_windows(events_df, event_labels, window_seconds=300)
    n_atk_win = (wl["label"] == 1).sum()
    n_ben_win = (wl["label"] == 0).sum()
    print(f"  Attack windows: {n_atk_win}, Benign windows: {n_ben_win}")

    if n_atk_win > 0:
        print(f"\n  Attack windows detail:")
        for _, row in wl[wl["label"] == 1].iterrows():
            print(f"    {row['window_start_dt']} | {row['n_attack']:>7,} atk / "
                  f"{row['n_total']:>7,} total ({100*row['attack_fraction']:5.2f}%)")

    print(f"\n✓ Ground truth labeling complete.")
