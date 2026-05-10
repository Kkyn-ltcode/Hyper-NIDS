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


def load_ground_truth(dataset: str = "theia") -> GroundTruth:
    """Load ground truth IoCs for a given dataset."""
    configs = {
        "theia": _THEIA_GT,
        "trace": _TRACE_GT,
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


def build_attack_subject_uuids(
    subjects_df: pd.DataFrame,
    gt: GroundTruth,
) -> set:
    """
    Build set of attack-related subject UUIDs.

    Uses:
        1. Exact basename match for known malicious processes
        2. Firefox/attack entry process tree (BFS on parent_uuid)
    """
    attack_uuids = set()

    # Method 1: Exact basename match
    if "process_path" in subjects_df.columns:
        basenames = subjects_df["process_path"].fillna("").apply(_get_basename)
        malicious_mask = basenames.isin(gt.malicious_process_basenames)
        attack_uuids.update(subjects_df.loc[malicious_mask, "uuid"].values)

    # Method 2: Attack entry process tree
    if gt.attack_entry_process:
        if "/" in gt.attack_entry_process:
            # Exact path match
            entry_mask = subjects_df["process_path"] == gt.attack_entry_process
        else:
            # Basename match
            basenames = subjects_df["process_path"].fillna("").apply(_get_basename)
            entry_mask = basenames == gt.attack_entry_process
        entry_uuids = set(subjects_df.loc[entry_mask, "uuid"].values)
        attack_uuids.update(entry_uuids)

        # BFS on parent_uuid to find descendants
        for _ in range(20):
            children = subjects_df[
                subjects_df["parent_uuid"].isin(attack_uuids)
                & ~subjects_df["uuid"].isin(attack_uuids)
            ]["uuid"].values
            if len(children) == 0:
                break
            attack_uuids.update(children)

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
    Label each event as attack (1) or benign (0).

    An event is attack if ANY of:
        - Its subject is in the attack subject set
        - Its predicate_object is an attack object
        - Its predicate_object2 is an attack object
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
