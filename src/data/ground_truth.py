"""
Ground truth labeling for DARPA TC E3 datasets.

Supports Theia and TRACE datasets via dataset-specific IoC configurations.
Each function takes DataFrames and returns labels — no file I/O.

Usage:
    from src.data.ground_truth import load_ground_truth, label_events
    gt = load_ground_truth("theia")
    labels = label_events(events_df, subjects_df, objects_df, gt)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class GroundTruth:
    """Container for dataset-specific attack IoCs."""
    dataset: str
    attack_ips: set = field(default_factory=set)
    malicious_file_paths: set = field(default_factory=set)
    malicious_file_substrings: list = field(default_factory=list)
    malicious_process_basenames: set = field(default_factory=set)
    attack_entry_process: str = ""


# ============================================================
# Dataset-specific IoC configurations
# ============================================================

_THEIA_GT = GroundTruth(
    dataset="theia",
    attack_ips={
        "161.116.88.72",      # THEIA drakon.linux.x64 C2
        "146.153.68.151",     # THEIA loaderDrakon.linux.x64
        "104.228.117.212",    # THEIA webserver (exploit delivery)
        "141.43.176.203",     # THEIA shellcode_server
        "149.52.198.23",      # THEIA micro APT
        "62.83.155.175",      # Phishing email source
        "208.75.117.3",       # www.nasa.ng (phishing link)
        "208.75.117.2",       # www.foo1.com (credential harvester)
    },
    malicious_file_paths={
        "/home/admin/cache",
        "/var/log/xtmp",
        "memtrace.so",
    },
    malicious_file_substrings=[
        "/home/admin/cache",
        "/var/log/xtmp",
        "memtrace.so",
        "tcexec",
    ],
    malicious_process_basenames={"clean", "cache", "xtmp"},
    attack_entry_process="/home/admin/Downloads/firefox/firefox",
)

_TRACE_GT = GroundTruth(
    dataset="trace",
    attack_ips={
        "145.199.103.57",     # TRACE webserver
        "61.130.69.232",      # TRACE shellcode_server
        "2.233.33.52",        # TRACE loaderDrakon
        "180.156.107.146",    # TRACE drakon
        "5.214.163.155",      # TRACE libdrakon
        "45.26.25.240",       # TRACE netrecon
        "162.66.239.75",      # TRACE micro
        "17.146.0.252",       # TRACE netrecon 2
        "62.83.155.175",      # Phishing
        "208.75.117.3",       # www.nasa.ng
        "208.75.117.2",       # www.foo1.com
    },
    malicious_file_substrings=[
        "xtmp", "ztmp", "cache",
    ],
    malicious_process_basenames={"xtmp", "ztmp", "cache"},
    attack_entry_process="firefox",
)

_TRACE_1_GT = GroundTruth(
    dataset="trace",
    attack_ips={
        "145.199.103.57",     # TRACE webserver
        "61.130.69.232",      # TRACE shellcode_server
        "2.233.33.52",        # TRACE loaderDrakon
        "180.156.107.146",    # TRACE drakon
        "5.214.163.155",      # TRACE libdrakon
        "45.26.25.240",       # TRACE netrecon
        "162.66.239.75",      # TRACE micro
        "17.146.0.252",       # TRACE netrecon 2
        "62.83.155.175",      # Phishing
        "208.75.117.3",       # www.nasa.ng
        "208.75.117.2",       # www.foo1.com
    },
    malicious_file_substrings=[
        "xtmp", "ztmp", "cache",
    ],
    malicious_process_basenames={"xtmp", "ztmp", "cache"},
    attack_entry_process="firefox",
)

def load_ground_truth(dataset: str = "theia") -> GroundTruth:
    """Load ground truth IoCs for a given dataset."""
    configs = {
        "theia": _THEIA_GT,
        "trace": _TRACE_GT,
        "trace-1": _TRACE_1_GT
    }
    if dataset not in configs:
        raise ValueError(f"Unknown dataset: {dataset}. "
                         f"Available: {list(configs.keys())}")
    return configs[dataset]


# ============================================================
# Labeling functions (stateless, pure)
# ============================================================

def _get_basename(path_str: str) -> str:
    """Get the last component of a path."""
    if not path_str:
        return ""
    return path_str.rstrip("/").rsplit("/", 1)[-1].lower()


def _get_process_basenames(subjects_df: pd.DataFrame) -> pd.Series:
    """Extract process basenames from whichever column is available."""
    if "process_path" in subjects_df.columns and subjects_df["process_path"].notna().any():
        return subjects_df["process_path"].fillna("").apply(_get_basename)
    elif "cmd_line" in subjects_df.columns:
        # TRACE: cmd_line is like "/usr/sbin/sshd -D -R", extract first token
        def _cmd_basename(cmd):
            if not cmd:
                return ""
            first_token = cmd.strip().split()[0] if cmd.strip() else ""
            return _get_basename(first_token)
        return subjects_df["cmd_line"].fillna("").apply(_cmd_basename)
    return pd.Series("", index=subjects_df.index)


def _match_entry_process(subjects_df: pd.DataFrame, entry: str) -> pd.Series:
    """Match the attack entry process against available columns."""
    if "/" in entry:
        # Exact path match — try process_path first, then cmd_line
        if "process_path" in subjects_df.columns and subjects_df["process_path"].notna().any():
            return subjects_df["process_path"] == entry
        elif "cmd_line" in subjects_df.columns:
            return subjects_df["cmd_line"].fillna("").str.startswith(entry)
    # Basename match
    basenames = _get_process_basenames(subjects_df)
    return basenames == _get_basename(entry)

NIL_UUID = "00000000-0000-0000-0000-000000000000"

def _build_children_map(subjects_df: pd.DataFrame) -> dict:
    """Build parent->children mapping, excluding nil UUIDs."""
    children = {}
    mask = subjects_df["parent_uuid"].notna() & (subjects_df["parent_uuid"] != NIL_UUID)
    for _, row in subjects_df[mask][["uuid", "parent_uuid"]].iterrows():
        children.setdefault(row["parent_uuid"], []).append(row["uuid"])
    return children

def _bfs_descendants(start_uuids: set, children_map: dict) -> set:
    """BFS to find all descendants of start_uuids."""
    visited = set(start_uuids)
    queue = list(start_uuids)
    while queue:
        curr = queue.pop()
        for child in children_map.get(curr, []):
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return visited

def build_attack_subject_uuids(
    subjects_df: pd.DataFrame,
    gt: GroundTruth,
) -> set:
    """
    Build set of attack-related subject UUIDs.
    Works with both Theia (process_path) and TRACE (cmd_line) schemas.
    """
    attack_uuids = set()

    # Method 1: Exact basename match for malicious processes
    basenames = _get_process_basenames(subjects_df)
    malicious_mask = basenames.isin(gt.malicious_process_basenames)
    attack_uuids.update(subjects_df.loc[malicious_mask, "uuid"].values)

    # Method 2: Attack entry process tree
    if gt.attack_entry_process:
        entry_mask = _match_entry_process(subjects_df, gt.attack_entry_process)
        entry_uuids = set(subjects_df.loc[entry_mask, "uuid"].values)
        attack_uuids.update(entry_uuids)

        # BFS on parent_uuid to find descendants
        children_map = _build_children_map(subjects_df)
        attack_uuids = _bfs_descendants(attack_uuids, children_map)

    return attack_uuids


def build_attack_object_uuids(
    objects_df: pd.DataFrame,
    gt: GroundTruth,
) -> set:
    """
    Build set of attack-related object UUIDs.

    Uses:
        1. Exact and substring file path matching
        2. Exact IP address matching
    """
    attack_uuids = set()
    objects_df = objects_df[objects_df["uuid"] != NIL_UUID]

    # File objects
    if "filename" in objects_df.columns:
        fnames = objects_df["filename"].fillna("")
        # Exact match
        if gt.malicious_file_paths:
            exact_mask = fnames.isin(gt.malicious_file_paths)
            attack_uuids.update(objects_df.loc[exact_mask, "uuid"].values)
        # Substring match
        for substr in gt.malicious_file_substrings:
            substr_mask = fnames.str.contains(substr, na=False, case=False)
            attack_uuids.update(objects_df.loc[substr_mask, "uuid"].values)

    # Network objects
    for col in ["remote_address", "local_address"]:
        if col in objects_df.columns:
            addrs = objects_df[col].fillna("")
            ip_mask = addrs.isin(gt.attack_ips)
            attack_uuids.update(objects_df.loc[ip_mask, "uuid"].values)

    return attack_uuids


def label_events(
    events_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    gt: GroundTruth,
) -> pd.Series:
    """
    BROAD labels: event is attack if ANY of its entities are attack-related.
    """
    attack_sub = build_attack_subject_uuids(subjects_df, gt)
    attack_obj = build_attack_object_uuids(objects_df, gt)

    print(f"  Attack subjects: {len(attack_sub):,}")
    print(f"  Attack objects:  {len(attack_obj):,}")

    labels = pd.Series(0, index=events_df.index, dtype=np.int8)

    # Subject match
    labels[events_df["subject_uuid"].isin(attack_sub)] = 1

    # Object match
    labels[events_df["predicate_object_uuid"].isin(attack_obj)] = 1

    # Object2 match
    if "predicate_object2_uuid" in events_df.columns:
        labels[events_df["predicate_object2_uuid"].isin(attack_obj)] = 1

    return labels


def build_child_only_subject_uuids(
    subjects_df: pd.DataFrame,
    gt: GroundTruth,
) -> set:
    """
    Build set of attack child process UUIDs, EXCLUDING the entry
    process (Firefox) itself. Only malicious binaries spawned by
    the attack chain are included (clean, cache, xtmp, etc.).

    This tests cross-process campaign propagation detection.
    """
    child_uuids = set()

    # Method 1: Exact basename match for malicious binaries
    basenames = _get_process_basenames(subjects_df)
    malicious_mask = basenames.isin(gt.malicious_process_basenames)
    child_uuids.update(subjects_df.loc[malicious_mask, "uuid"].values)

    # Method 2: Get Firefox's children but NOT Firefox itself
    if gt.attack_entry_process:
        entry_mask = _match_entry_process(subjects_df, gt.attack_entry_process)

        entry_uuids = set(subjects_df.loc[entry_mask, "uuid"].values)

        # BFS from Firefox to find descendants (but don't include Firefox)
        children_map = _build_children_map(subjects_df)
        all_descendants = _bfs_descendants(entry_uuids, children_map)
        # Exclude entry process itself
        child_uuids.update(all_descendants - entry_uuids)

    return child_uuids


def label_crossprocess_events(
    events_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    gt: GroundTruth,
) -> pd.Series:
    """
    CROSS-PROCESS labels: event is attack only if the subject is a
    child of the attack entry process (NOT the entry process itself).

    Firefox events are labeled 0 (benign). Only events from spawned
    malicious processes (clean, cache, xtmp) are labeled 1.

    This tests whether the model can detect campaign propagation
    across process boundaries — the harder, more realistic task.
    """
    child_uuids = build_child_only_subject_uuids(subjects_df, gt)

    labels = pd.Series(0, index=events_df.index, dtype=np.int8)
    labels[events_df["subject_uuid"].isin(child_uuids)] = 1

    n_atk = int(labels.sum())
    print(f"  Cross-process labels: {n_atk:,} attack / {len(labels):,} "
          f"total ({100*n_atk/len(labels):.3f}%)")

    return labels


def label_narrow_events(
    events_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    gt: GroundTruth,
) -> pd.Series:
    """
    NARROW labels: event is attack only if it directly involves an
    IoC object (malicious file, C2 IP, malicious process output).

    An event must satisfy BOTH:
        - Subject is in the attack process tree (broad condition)
        - At least one object is a known IoC (file path, IP address)

    This filters out the ~90% of attack-subject events that are
    routine benign operations (reads, mprotects) by the attack process.
    """
    attack_sub = build_attack_subject_uuids(subjects_df, gt)
    attack_obj = build_attack_object_uuids(objects_df, gt)

    labels = pd.Series(0, index=events_df.index, dtype=np.int8)

    # Must be from attack subject AND touch an IoC object
    is_attack_sub = events_df["subject_uuid"].isin(attack_sub)
    touches_ioc_obj = events_df["predicate_object_uuid"].isin(attack_obj)
    if "predicate_object2_uuid" in events_df.columns:
        touches_ioc_obj = touches_ioc_obj | \
            events_df["predicate_object2_uuid"].isin(attack_obj)

    labels[is_attack_sub & touches_ioc_obj] = 1

    n_atk = int(labels.sum())
    print(f"  Narrow labels: {n_atk:,} attack / {len(labels):,} total "
          f"({100*n_atk/len(labels):.3f}%)")

    return labels


def label_ioc_events(
    events_df: pd.DataFrame,
    subjects_df: pd.DataFrame,
    objects_df: pd.DataFrame,
    gt: GroundTruth,
) -> pd.Series:
    """
    IoC-ONLY labels: event is attack only if it directly interacts
    with a known-malicious IP address or file path.

    No subject filter — any event touching an IoC is labeled.
    This is the strictest, most realistic labeling.
    """
    attack_obj = build_attack_object_uuids(objects_df, gt)

    labels = pd.Series(0, index=events_df.index, dtype=np.int8)

    touches_ioc = events_df["predicate_object_uuid"].isin(attack_obj)
    if "predicate_object2_uuid" in events_df.columns:
        touches_ioc = touches_ioc | \
            events_df["predicate_object2_uuid"].isin(attack_obj)

    labels[touches_ioc] = 1

    n_atk = int(labels.sum())
    print(f"  IoC labels: {n_atk:,} attack / {len(labels):,} total "
          f"({100*n_atk/len(labels):.4f}%)")

    return labels

