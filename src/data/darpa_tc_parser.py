"""
Parse DARPA TC E3 JSON data into structured DataFrames.

Reads the extracted JSON shard files line-by-line (memory-efficient),
extracts Events, Subjects, and Objects into clean pandas DataFrames,
and saves them as parquet files for fast reloading.

Memory-safe: parses one shard at a time, saves per-shard parquets,
then merges. Supports 8 GB RAM machines with 35+ GB of raw JSON.

Usage:
    # Parse first shard only (for testing)
    python -m src.data.darpa_tc_parser --shards 0

    # Parse all shards (incremental, memory-safe)
    python -m src.data.darpa_tc_parser --shards all

    # Load previously parsed data
    python -m src.data.darpa_tc_parser --load-only

Output:
    data/processed/darpa_tc_e3/theia/
        events.parquet       # All events (merged, sorted)
        subjects.parquet     # All subjects (deduplicated)
        objects.parquet      # All objects (deduplicated)
        shards/              # Per-shard parquets (intermediate)
        summary.json         # Parsing statistics
"""

import argparse
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

def get_raw_dir(dataset: str) -> Path:
    """Return path to raw DARPA TC E3 data for a dataset."""
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "data" / "raw" / "darpa_tc_e3" / dataset


def get_processed_dir(dataset: str) -> Path:
    """Return path to processed output, creating it if needed."""
    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = project_root / "data" / "processed" / "darpa_tc_e3" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ============================================================
# Avro union unwrapping
# ============================================================

def unwrap(value):
    """
    Unwrap Avro union-encoded values to plain Python types.

    The CDM JSON uses Avro's union encoding:
        {"string": "hello"}   -> "hello"
        {"long": 12345}       -> 12345
        {"int": 42}           -> 42
        {"com.bbn...UUID": "ABC-123"} -> "ABC-123"
        {"map": {"k": "v"}}  -> {"k": "v"}
        null                  -> None
    """
    if value is None:
        return None
    if isinstance(value, dict) and len(value) == 1:
        return next(iter(value.values()))
    return value


# ============================================================
# Record extraction functions
# ============================================================

def extract_event(record_data: dict) -> dict:
    """Extract relevant fields from a CDM Event record."""
    return {
        "uuid": record_data.get("uuid"),
        "type": record_data.get("type"),
        "timestamp_nanos": record_data.get("timestampNanos"),
        "subject_uuid": unwrap(record_data.get("subject")),
        "predicate_object_uuid": unwrap(record_data.get("predicateObject")),
        "predicate_object2_uuid": unwrap(record_data.get("predicateObject2")),
        "predicate_object_path": unwrap(record_data.get("predicateObjectPath")),
        "predicate_object2_path": unwrap(record_data.get("predicateObject2Path")),
        "name": unwrap(record_data.get("name")),
        "size": unwrap(record_data.get("size")),
        "thread_id": unwrap(record_data.get("threadId")),
        "host_id": record_data.get("hostId"),
    }


def extract_subject(record_data: dict) -> dict:
    """Extract relevant fields from a CDM Subject record."""
    props = unwrap(record_data.get("properties")) or {}
    return {
        "uuid": record_data.get("uuid"),
        "type": record_data.get("type"),
        "cid": record_data.get("cid"),  # process ID
        "parent_uuid": unwrap(record_data.get("parentSubject")),
        "cmd_line": unwrap(record_data.get("cmdLine")),
        "start_timestamp_nanos": record_data.get("startTimestampNanos"),
        "local_principal": record_data.get("localPrincipal"),
        "host_id": record_data.get("hostId"),
        # From properties
        "process_path": props.get("path"),
        "ppid": props.get("ppid"),
        "tgid": props.get("tgid"),
    }


def extract_file_object(record_data: dict) -> dict:
    """Extract relevant fields from a CDM FileObject record."""
    base = record_data.get("baseObject", {}) or {}
    base_props = unwrap(base.get("properties")) or {}
    return {
        "uuid": record_data.get("uuid"),
        "object_type": "FILE",
        "filename": base_props.get("filename"),
        "dev": base_props.get("dev"),
        "inode": base_props.get("inode"),
        "host_id": base.get("hostId"),
    }


def extract_netflow_object(record_data: dict) -> dict:
    """Extract relevant fields from a CDM NetFlowObject record."""
    return {
        "uuid": record_data.get("uuid"),
        "object_type": "NETFLOW",
        "local_address": unwrap(record_data.get("localAddress")),
        "local_port": unwrap(record_data.get("localPort")),
        "remote_address": unwrap(record_data.get("remoteAddress")),
        "remote_port": unwrap(record_data.get("remotePort")),
        "filename": None,
        "dev": None,
        "inode": None,
        "host_id": record_data.get("baseObject", {}).get("hostId") if record_data.get("baseObject") else None,
    }


def extract_memory_object(record_data: dict) -> dict:
    """Extract relevant fields from a CDM MemoryObject record."""
    base = record_data.get("baseObject", {}) or {}
    return {
        "uuid": record_data.get("uuid"),
        "object_type": "MEMORY",
        "filename": None,
        "dev": None,
        "inode": None,
        "host_id": base.get("hostId"),
    }


def extract_srcsink_object(record_data: dict) -> dict:
    """Extract relevant fields from a CDM SrcSinkObject record."""
    return {
        "uuid": record_data.get("uuid"),
        "object_type": "SRCSINK",
        "filename": None,
        "dev": None,
        "inode": None,
        "host_id": record_data.get("baseObject", {}).get("hostId") if record_data.get("baseObject") else None,
    }


def extract_pipe_object(record_data: dict) -> dict:
    """Extract relevant fields from a CDM UnnamedPipeObject record."""
    return {
        "uuid": record_data.get("uuid"),
        "object_type": "PIPE",
        "filename": None,
        "dev": None,
        "inode": None,
        "host_id": record_data.get("baseObject", {}).get("hostId") if record_data.get("baseObject") else None,
    }


# Map from CDM fully-qualified type to (short_name, extraction_function)
RECORD_HANDLERS = {
    "com.bbn.tc.schema.avro.cdm18.Event": ("event", extract_event),
    "com.bbn.tc.schema.avro.cdm18.Subject": ("subject", extract_subject),
    "com.bbn.tc.schema.avro.cdm18.FileObject": ("object", extract_file_object),
    "com.bbn.tc.schema.avro.cdm18.NetFlowObject": ("object", extract_netflow_object),
    "com.bbn.tc.schema.avro.cdm18.MemoryObject": ("object", extract_memory_object),
    "com.bbn.tc.schema.avro.cdm18.SrcSinkObject": ("object", extract_srcsink_object),
    "com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject": ("object", extract_pipe_object),
}


# ============================================================
# Core parsing
# ============================================================

def count_lines(filepath: Path) -> int:
    """Count lines in a file efficiently for progress bar."""
    count = 0
    with open(filepath, "rb") as f:
        buf = f.raw.read(65536)
        while buf:
            count += buf.count(b"\n")
            buf = f.raw.read(65536)
    return count


def parse_shard(filepath: Path, show_progress: bool = True) -> dict:
    """
    Parse a single JSON shard file into lists of extracted records.

    Args:
        filepath: Path to the JSON shard file.
        show_progress: Whether to show a tqdm progress bar.

    Returns:
        Dictionary with keys 'events', 'subjects', 'objects'
        each containing a list of extracted record dicts.
    """
    events = []
    subjects = []
    objects = []
    parse_errors = 0
    skipped_types = Counter()


    total_lines = None

    with open(filepath, "r") as f:
        iterator = tqdm(f, total=total_lines, desc=filepath.name, unit=" records") \
            if show_progress else f
        for line in iterator:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            datum = record.get("datum")
            if not datum or not isinstance(datum, dict):
                continue

            record_type = next(iter(datum))
            record_data = datum[record_type]

            handler = RECORD_HANDLERS.get(record_type)
            if handler is None:
                short_name = record_type.split(".")[-1]
                skipped_types[short_name] += 1
                continue

            category, extract_fn = handler
            extracted = extract_fn(record_data)

            if category == "event":
                events.append(extracted)
            elif category == "subject":
                subjects.append(extracted)
            elif category == "object":
                objects.append(extracted)

    if parse_errors:
        print(f"  Parse errors: {parse_errors}")
    if skipped_types:
        print(f"  Skipped record types: {dict(skipped_types)}")

    return {
        "events": events,
        "subjects": subjects,
        "objects": objects,
        "parse_errors": parse_errors,
    }


def build_dataframes(parsed: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert parsed record lists into pandas DataFrames with appropriate dtypes.

    Returns:
        (events_df, subjects_df, objects_df)
    """
    # --- Events ---
    events_df = pd.DataFrame(parsed["events"])
    if len(events_df) > 0 and "type" in events_df.columns:
        events_df["timestamp_nanos"] = pd.to_numeric(
            events_df["timestamp_nanos"], errors="coerce"
        )
        valid_ts = events_df["timestamp_nanos"].notna()
        events_df.loc[valid_ts, "timestamp"] = pd.to_datetime(
            events_df.loc[valid_ts, "timestamp_nanos"], unit="ns"
        )
        events_df = events_df.sort_values("timestamp_nanos").reset_index(drop=True)
        events_df["type"] = events_df["type"].astype("category")

    # --- Subjects ---
    subjects_df = pd.DataFrame(parsed["subjects"])
    if len(subjects_df) > 0 and "type" in subjects_df.columns:
        subjects_df["type"] = subjects_df["type"].astype("category")
        subjects_df["start_timestamp_nanos"] = pd.to_numeric(
            subjects_df["start_timestamp_nanos"], errors="coerce"
        )

    # --- Objects ---
    objects_df = pd.DataFrame(parsed["objects"])
    if len(objects_df) > 0 and "object_type" in objects_df.columns:
        objects_df["object_type"] = objects_df["object_type"].astype("category")

    return events_df, subjects_df, objects_df


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parse DARPA TC E3 JSON into structured DataFrames"
    )
    parser.add_argument(
        "--shards", default="0",
        help="Which shards to parse: '0', '0,1,2', or 'all' (default: '0')",
    )
    parser.add_argument(
        "--load-only", action="store_true",
        help="Just load and summarize previously saved parquet files",
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Skip parsing, just merge existing per-shard parquets",
    )
    parser.add_argument(
        "--dataset", type=str, default="theia", choices=["theia", "trace", "cadets", "trace-1"],
        help="Which dataset to process",
    )
    args = parser.parse_args()

    raw_dir = get_raw_dir(args.dataset)
    processed_dir = get_processed_dir(args.dataset)

    # --- Load-only mode ---
    if args.load_only:
        print(f"Loading from: {processed_dir}")
        for name in ["events", "subjects", "objects"]:
            path = processed_dir / f"{name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                print(f"\n{name}: {len(df):,} rows, "
                      f"{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
                print(f"  Columns: {list(df.columns)}")
                if name == "events" and "type" in df.columns:
                    print(f"  Event types: {df['type'].value_counts().to_dict()}")
                if name == "events" and "timestamp" in df.columns:
                    print(f"  Time range: {df['timestamp'].min()} to "
                          f"{df['timestamp'].max()}")
                if name == "objects" and "object_type" in df.columns:
                    print(f"  Object types: "
                          f"{df['object_type'].value_counts().to_dict()}")
            else:
                print(f"\n{name}: NOT FOUND at {path}")
        return

    # --- Find JSON shards ---
    all_shards = sorted([
        f for f in raw_dir.iterdir()
        if f.is_file() and ".json" in f.name and ".tar.gz" not in f.name
    ])

    if not all_shards and not args.merge_only:
        print(f"No JSON shards found in {raw_dir}")
        sys.exit(1)

    if not args.merge_only:
        print(f"Available shards ({len(all_shards)}):")
        for i, s in enumerate(all_shards):
            size_gb = s.stat().st_size / (1024**3)
            print(f"  [{i}] {s.name}: {size_gb:.2f} GB")

    if args.shards == "all":
        selected = list(range(len(all_shards)))
    else:
        selected = [int(x.strip()) for x in args.shards.split(",")]

    shard_dir = processed_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # ---- STEP 1: Parse each shard independently (memory-safe) ----
    if not args.merge_only:
        selected_shards = [all_shards[i] for i in selected]
        print(f"\nParsing {len(selected_shards)} shard(s) incrementally...")

        for shard_idx, shard_path in zip(selected, selected_shards):
            shard_parquet = shard_dir / f"events_shard{shard_idx}.parquet"

            # Skip if already parsed
            if shard_parquet.exists():
                n = len(pd.read_parquet(shard_parquet, columns=["uuid"]))
                print(f"\n[Shard {shard_idx}] Already parsed ({n:,} events) → skip")
                continue

            print(f"\n{'='*60}")
            print(f"Parsing shard {shard_idx}: {shard_path.name}")
            print(f"{'='*60}")

            parsed = parse_shard(shard_path, show_progress=True)
            print(f"  Events: {len(parsed['events']):,}, "
                  f"Subjects: {len(parsed['subjects']):,}, "
                  f"Objects: {len(parsed['objects']):,}")

            events_df, subjects_df, objects_df = build_dataframes(parsed)
            del parsed

            events_df.to_parquet(shard_parquet, index=False)
            subjects_df.to_parquet(
                shard_dir / f"subjects_shard{shard_idx}.parquet", index=False)
            objects_df.to_parquet(
                shard_dir / f"objects_shard{shard_idx}.parquet", index=False)

            print(f"  ✓ Saved shard {shard_idx} parquets")
            del events_df, subjects_df, objects_df
            gc.collect()

    elapsed_parse = time.time() - start_time
    print(f"\nParsing phase done in {elapsed_parse:.1f}s")

    # ---- STEP 2: Merge per-shard parquets ----
    print(f"\n{'='*60}")
    print("MERGING SHARDS")
    print(f"{'='*60}")

    event_parts = sorted(shard_dir.glob("events_shard*.parquet"))
    print(f"\nMerging {len(event_parts)} event shards...")

    # Merge events
    events_all = pd.concat(
        [pd.read_parquet(p) for p in event_parts], ignore_index=True
    )
    events_all = events_all.sort_values("timestamp_nanos").reset_index(drop=True)
    events_all["type"] = events_all["type"].astype("category")
    events_all.to_parquet(processed_dir / "events.parquet", index=False)
    n_ev = len(events_all)
    mem_ev = events_all.memory_usage(deep=True).sum() / 1e6
    ts_min = events_all["timestamp"].min()
    ts_max = events_all["timestamp"].max()
    ev_types = events_all["type"].value_counts().to_dict()
    del events_all; gc.collect()
    print(f"  events.parquet: {n_ev:,} rows ({mem_ev:.0f} MB)")

    # Merge subjects (dedup)
    subj_parts = sorted(shard_dir.glob("subjects_shard*.parquet"))
    print(f"Merging {len(subj_parts)} subject shards...")
    subjects_all = pd.concat(
        [pd.read_parquet(p) for p in subj_parts], ignore_index=True
    ).drop_duplicates(subset=["uuid"], keep="first")
    subjects_all.to_parquet(processed_dir / "subjects.parquet", index=False)
    n_sub = len(subjects_all)
    del subjects_all; gc.collect()
    print(f"  subjects.parquet: {n_sub:,} rows (deduplicated)")

    # Merge objects (dedup)
    obj_parts = sorted(shard_dir.glob("objects_shard*.parquet"))
    print(f"Merging {len(obj_parts)} object shards...")
    objects_all = pd.concat(
        [pd.read_parquet(p) for p in obj_parts], ignore_index=True
    ).drop_duplicates(subset=["uuid"], keep="first")
    objects_all.to_parquet(processed_dir / "objects.parquet", index=False)
    n_obj = len(objects_all)
    del objects_all; gc.collect()
    print(f"  objects.parquet: {n_obj:,} rows (deduplicated)")

    # ---- Summary ----
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"PARSING SUMMARY")
    print(f"{'='*60}")
    print(f"  Events:   {n_ev:,}")
    print(f"  Subjects: {n_sub:,}")
    print(f"  Objects:  {n_obj:,}")
    print(f"  Time range: {ts_min} to {ts_max}")
    print(f"\n  Event type distribution:")
    for etype, count in sorted(ev_types.items(), key=lambda x: -x[1]):
        pct = 100 * count / n_ev
        print(f"    {etype:30s} {count:>10,}  ({pct:5.1f}%)")

    summary = {
        "shards_parsed": [p.name for p in event_parts],
        "total_events": n_ev,
        "total_subjects": n_sub,
        "total_objects": n_obj,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(processed_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✓ Done. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
