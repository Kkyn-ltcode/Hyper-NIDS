"""
Ground truth labeling for DARPA TC E3 Theia dataset.

Uses EXACT IoCs from the official ground truth report:
    TC_Ground_Truth_Report_E3_Update.pdf

Theia E3 Attack Scenario (Firefox Backdoor w/ Drakon APT):
    Date: April 3, 2018
    1. Firefox exploit via www.allstate.com/www.gatech.edu
    2. Drakon implant runs in Firefox memory
    3. Writes /home/admin/cache (drakon binary), /var/log/xtmp (libdrakon)
    4. Executes drakon as root ("clean" process), connects to C2
    5. Port scanning via micro APT

Official C2/Attack IPs:
    161.116.88.72   - drakon C2
    146.153.68.151  - loaderDrakon
    104.228.117.212 - webserver (exploit)
    141.43.176.203  - shellcode_server
    149.52.198.23   - micro APT

Phishing IPs (April 10):
    62.83.155.175   - phishing email
    208.75.117.3    - www.nasa.ng (phishing link)
    208.75.117.2    - www.foo1.com (credential harvester)

Usage:
    python -m src.coordination.ground_truth_e3
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Official IoCs from TC_Ground_Truth_Report_E3_Update.pdf
# ============================================================

# C2 and attack infrastructure IPs (from ground truth §3 and §4)
ATTACK_IPS = {
    "161.116.88.72",     # THEIA drakon.linux.x64 C2
    "146.153.68.151",    # THEIA loaderDrakon.linux.x64
    "104.228.117.212",   # THEIA webserver (exploit delivery)
    "141.43.176.203",    # THEIA shellcode_server
    "149.52.198.23",     # THEIA micro APT
    "62.83.155.175",     # Phishing email source
    "208.75.117.3",      # www.nasa.ng (phishing link)
    "208.75.117.2",      # www.foo1.com (credential harvester)
}

# Malicious files written to disk (from ground truth §3.5, §3.10)
MALICIOUS_FILE_PATHS = {
    "/home/admin/cache",    # drakon.linux.x64 binary
    "/var/log/xtmp",        # libdrakon.linux.x64.so
    "memtrace.so",          # libdrakon (alternate name)
}

# Malicious file substrings (for partial path matching)
MALICIOUS_FILE_SUBSTRINGS = [
    "/home/admin/cache",
    "/var/log/xtmp",
    "memtrace.so",
    "tcexec",               # Failed phishing executable
]

# Malicious process basenames (exact match on last path component)
MALICIOUS_PROCESS_BASENAMES = {
    "clean",    # Drakon implant executed as root
    "cache",    # Drakon binary (alternate name)
    "xtmp",     # libdrakon
}

# Firefox is the exploited entry point
ATTACK_ENTRY_PROCESS = "/home/admin/Downloads/firefox/firefox"


def _get_basename(path_str: str) -> str:
    """Get the last component of a path."""
    if not path_str:
        return ""
    return path_str.rstrip("/").rsplit("/", 1)[-1].lower()


def _build_attack_subject_uuids(subjects_df: pd.DataFrame) -> set:
    """
    Build the set of attack-related subject UUIDs using two methods:

    1. Exact basename match for known malicious processes (clean, cache, xtmp)
    2. Firefox process tree (Firefox + all descendants via parent_uuid)

    Returns set of attack-related subject UUIDs.
    """
    attack_uuids = set()

    # Method 1: Exact basename match for malicious binaries
    if "process_path" in subjects_df.columns:
        basenames = subjects_df["process_path"].fillna("").apply(_get_basename)
        malicious_mask = basenames.isin(MALICIOUS_PROCESS_BASENAMES)
        attack_uuids.update(subjects_df.loc[malicious_mask, "uuid"].values)

    # Method 2: Firefox process tree
    firefox_mask = subjects_df["process_path"] == ATTACK_ENTRY_PROCESS
    firefox_uuids = set(subjects_df.loc[firefox_mask, "uuid"].values)
    attack_uuids.update(firefox_uuids)

    # Trace children via parent_uuid (iterative BFS)
    for _ in range(20):
        children = subjects_df[
            subjects_df["parent_uuid"].isin(attack_uuids)
            & ~subjects_df["uuid"].isin(attack_uuids)
        ]["uuid"].values
        if len(children) == 0:
            break
        attack_uuids.update(children)

    return attack_uuids


def _build_attack_object_uuids(objects_df: pd.DataFrame) -> set:
    """
    Build set of attack-related object UUIDs using:

    1. Exact file path match for known malicious files
    2. Substring match for malicious file indicators
    3. Exact IP match for C2/attack infrastructure
    """
    attack_uuids = set()

    # File objects: exact path and substring matching
    if "filename" in objects_df.columns:
        fnames = objects_df["filename"].fillna("")
        # Exact match
        exact_mask = fnames.isin(MALICIOUS_FILE_PATHS)
        attack_uuids.update(objects_df.loc[exact_mask, "uuid"].values)
        # Substring match
        for substr in MALICIOUS_FILE_SUBSTRINGS:
            substr_mask = fnames.str.contains(substr, na=False, case=False)
            attack_uuids.update(objects_df.loc[substr_mask, "uuid"].values)

    # Network objects: exact IP match
    for col in ["remote_address", "local_address"]:
        if col in objects_df.columns:
            addrs = objects_df[col].fillna("")
            ip_mask = addrs.isin(ATTACK_IPS)
            attack_uuids.update(objects_df.loc[ip_mask, "uuid"].values)

    return attack_uuids


def label_events(
    events_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    objects_df: pd.DataFrame,
) -> pd.Series:
    """
    Label events as attack (1) or benign (0).

    An event is attack-related if ANY of:
    - Its subject is in the attack subject set (Firefox tree + malicious procs)
    - Its predicate_object is an attack object (malicious file or C2 connection)
    - Its predicate_object2 is an attack object
    """
    attack_subject_uuids = _build_attack_subject_uuids(subjects_df)
    attack_object_uuids = _build_attack_object_uuids(objects_df)

    print(f"  Attack subjects: {len(attack_subject_uuids):,}")
    print(f"  Attack objects:  {len(attack_object_uuids):,}")

    event_labels = pd.Series(0, index=events_df.index, dtype=np.int8)

    # Match by subject UUID
    sub_mask = events_df["subject_uuid"].isin(attack_subject_uuids)
    event_labels[sub_mask] = 1

    # Match by predicate object UUID
    obj_mask = events_df["predicate_object_uuid"].isin(attack_object_uuids)
    event_labels[obj_mask] = 1

    # Match by predicate object 2 UUID
    if "predicate_object2_uuid" in events_df.columns:
        obj2_mask = events_df["predicate_object2_uuid"].isin(attack_object_uuids)
        event_labels[obj2_mask] = 1

    return event_labels


def label_windows(
    events_df: pd.DataFrame,
    event_labels: pd.Series,
    window_seconds: float = 300.0,
) -> pd.DataFrame:
    """Assign attack/benign labels to time windows."""
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
            "window_start": w_start, "window_end": w_end,
            "label": 1 if n_attack > 0 else 0,
            "attack_fraction": n_attack / n_total,
            "n_attack": n_attack, "n_total": int(n_total),
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
    print(f"  Events: {len(events_df):,}, Subjects: {len(subjects_df):,}, "
          f"Objects: {len(objects_df):,}")
    print(f"  Time range: {events_df['timestamp'].min()} to {events_df['timestamp'].max()}")

    # Label events
    print("\nLabeling events...")
    event_labels = label_events(events_df, subjects_df, objects_df)
    n_attack = (event_labels == 1).sum()
    n_benign = (event_labels == 0).sum()
    print(f"  Attack: {n_attack:,} ({100*n_attack/len(events_df):.2f}%)")
    print(f"  Benign: {n_benign:,} ({100*n_benign/len(events_df):.2f}%)")

    # Attack event type breakdown
    if n_attack > 0:
        print(f"\n  Attack event types:")
        atk = events_df[event_labels == 1]
        for etype, count in atk["type"].value_counts().head(10).items():
            pct = 100 * count / n_attack
            print(f"    {etype:30s} {count:>10,} ({pct:5.1f}%)")

    # Window labels (5-min)
    print("\nLabeling 5-min windows...")
    wl = label_windows(events_df, event_labels, window_seconds=300)
    n_atk_win = (wl["label"] == 1).sum()
    n_ben_win = (wl["label"] == 0).sum()
    print(f"  Attack windows: {n_atk_win}, Benign windows: {n_ben_win}")

    print(f"\n✓ Ground truth labeling complete.")
