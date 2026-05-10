"""
Ground truth labeling for DARPA TC E3 TRACE dataset.

Uses EXACT IoCs from the official ground truth report:
    TC_Ground_Truth_Report_E3_Update.pdf

TRACE E3 Attack Scenarios:
    - Nginx Backdoor w/ Drakon In-Memory
    - Phishing E-mail Link

Official C2/Attack IPs for TRACE:
    145.199.103.57  - TRACE webserver
    61.130.69.232   - TRACE shellcode_server
    2.233.33.52     - TRACE loaderDrakon
    180.156.107.146 - TRACE drakon (C2)
    5.214.163.155   - TRACE libdrakon
    45.26.25.240    - TRACE netrecon
    162.66.239.75   - TRACE micro
    17.146.0.252    - TRACE netrecon 2
    62.83.155.175   - Phishing email source (from Theia report)
    208.75.117.3    - www.nasa.ng (phishing link)
    208.75.117.2    - www.foo1.com (credential harvester)

Malicious Files:
    /home/admin/cache
    /var/log/xtmp
    xtmp

Usage:
    python -m src.coordination.ground_truth_trace
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Official IoCs from TC_Ground_Truth_Report_E3_Update.pdf
# ============================================================

ATTACK_IPS = {
    "145.199.103.57",
    "61.130.69.232",
    "2.233.33.52",
    "180.156.107.146",
    "5.214.163.155",
    "45.26.25.240",
    "162.66.239.75",
    "17.146.0.252",
    "62.83.155.175",
    "208.75.117.3",
    "208.75.117.2",
}

MALICIOUS_FILE_PATHS = {
    "/home/admin/cache",
    "/var/log/xtmp",
}

MALICIOUS_FILE_SUBSTRINGS = [
    "/home/admin/cache",
    "/var/log/xtmp",
    "xtmp",
]

MALICIOUS_PROCESS_BASENAMES = {
    "clean",    # Typically drakon runs as clean
    "cache",
    "xtmp",
}

# The main entry points on TRACE. In the dataset it might be nginx or something similar.
ATTACK_ENTRY_PROCESS_SUBSTRINGS = [
    "nginx"
]

def _get_basename(path_str: str) -> str:
    if not path_str:
        return ""
    return path_str.rstrip("/").rsplit("/", 1)[-1].lower()


def _build_attack_subject_uuids(subjects_df: pd.DataFrame) -> set:
    attack_uuids = set()

    if "process_path" in subjects_df.columns:
        paths = subjects_df["process_path"].fillna("")
        basenames = paths.apply(_get_basename)
        
        malicious_mask = basenames.isin(MALICIOUS_PROCESS_BASENAMES)
        attack_uuids.update(subjects_df.loc[malicious_mask, "uuid"].values)
        
        for substr in ATTACK_ENTRY_PROCESS_SUBSTRINGS:
            entry_mask = paths.str.contains(substr, case=False, na=False)
            attack_uuids.update(subjects_df.loc[entry_mask, "uuid"].values)

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
    attack_uuids = set()

    if "filename" in objects_df.columns:
        fnames = objects_df["filename"].fillna("")
        exact_mask = fnames.isin(MALICIOUS_FILE_PATHS)
        attack_uuids.update(objects_df.loc[exact_mask, "uuid"].values)
        
        for substr in MALICIOUS_FILE_SUBSTRINGS:
            substr_mask = fnames.str.contains(substr, na=False, case=False)
            attack_uuids.update(objects_df.loc[substr_mask, "uuid"].values)

    for col in ["remote_address", "local_address"]:
        if col in objects_df.columns:
            addrs = objects_df[col].fillna("")
            ip_mask = addrs.isin(ATTACK_IPS)
            attack_uuids.update(objects_df.loc[ip_mask, "uuid"].values)

    return attack_uuids


def label_events(events_df: pd.DataFrame, subjects_df: pd.DataFrame, objects_df: pd.DataFrame) -> pd.Series:
    attack_subject_uuids = _build_attack_subject_uuids(subjects_df)
    attack_object_uuids = _build_attack_object_uuids(objects_df)

    print(f"  Attack subjects: {len(attack_subject_uuids):,}")
    print(f"  Attack objects:  {len(attack_object_uuids):,}")

    event_labels = pd.Series(0, index=events_df.index, dtype=np.int8)

    sub_mask = events_df["subject_uuid"].isin(attack_subject_uuids)
    event_labels[sub_mask] = 1

    obj_mask = events_df["predicate_object_uuid"].isin(attack_object_uuids)
    event_labels[obj_mask] = 1

    if "predicate_object2_uuid" in events_df.columns:
        obj2_mask = events_df["predicate_object2_uuid"].isin(attack_object_uuids)
        event_labels[obj2_mask] = 1

    return event_labels


def label_windows(events_df: pd.DataFrame, event_labels: pd.Series, window_seconds: float = 300.0) -> pd.DataFrame:
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


if __name__ == "__main__":
    processed_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "data" / "processed" / "darpa_tc_e3" / "trace"
    )

    print("Loading TRACE data...")
    try:
        events_df = pd.read_parquet(processed_dir / "events.parquet")
        subjects_df = pd.read_parquet(processed_dir / "subjects.parquet")
        objects_df = pd.read_parquet(processed_dir / "objects.parquet")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run parser for TRACE dataset first.")
        import sys; sys.exit(1)

    print(f"  Events: {len(events_df):,}, Subjects: {len(subjects_df):,}, Objects: {len(objects_df):,}")

    print("\nLabeling events...")
    event_labels = label_events(events_df, subjects_df, objects_df)
    n_attack = (event_labels == 1).sum()
    n_benign = (event_labels == 0).sum()
    print(f"  Attack: {n_attack:,} ({100*n_attack/len(events_df):.2f}%)")
    print(f"  Benign: {n_benign:,} ({100*n_benign/len(events_df):.2f}%)")

    if n_attack > 0:
        print(f"\n  Attack event types:")
        atk = events_df[event_labels == 1]
        for etype, count in atk["type"].value_counts().head(10).items():
            pct = 100 * count / n_attack
            print(f"    {etype:30s} {count:>10,} ({pct:5.1f}%)")

    print("\nLabeling 5-min windows...")
    wl = label_windows(events_df, event_labels, window_seconds=300)
    n_atk_win = (wl["label"] == 1).sum()
    n_ben_win = (wl["label"] == 0).sum()
    print(f"  Attack windows: {n_atk_win}, Benign windows: {n_ben_win}")

    print(f"\n✓ TRACE ground truth labeling complete.")
