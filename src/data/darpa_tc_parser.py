"""
Parse DARPA TC E3 JSON data into structured DataFrames.

Reads the extracted JSON shard files line-by-line (memory-efficient),
extracts Events, Subjects, and Objects into clean pandas DataFrames,
and saves them as parquet files for fast reloading.

Usage:
    # Parse first shard only (for testing)
    python -m src.data.darpa_tc_parser --shards 0

    # Parse all shards
    python -m src.data.darpa_tc_parser --shards all

    # Load previously parsed data
    python -m src.data.darpa_tc_parser --load-only

Output:
    data/processed/darpa_tc_e3/theia/
        events.parquet       # All events with flattened fields
        subjects.parquet     # All subjects (processes)
        objects.parquet      # All objects (files, network, memory)
        summary.json         # Parsing statistics
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

def get_raw_dir() -> Path:
    """Return path to raw DARPA TC E3 theia data."""
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "data" / "raw" / "darpa_tc_e3" / "theia"


def get_processed_dir() -> Path:
    """Return path to processed output, creating it if needed."""
    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = project_root / "data" / "processed" / "darpa_tc_e3" / "theia"
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
        "filename": None,  # Consistent schema with FileObject
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
        # Read in 64KB chunks for speed
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

    # Count lines for progress bar
    if show_progress:
        print(f"Counting lines in {filepath.name}...")
        total_lines = count_lines(filepath)
        print(f"  {total_lines:,} lines")
    else:
        total_lines = None

    iterator = open(filepath, "r")
    if show_progress:
        iterator = tqdm(iterator, total=total_lines, desc=filepath.name, unit=" records")

    try:
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

            # Get the record type (single key in datum dict)
            record_type = next(iter(datum))
            record_data = datum[record_type]

            handler = RECORD_HANDLERS.get(record_type)
            if handler is None:
                # Skip metadata records (Host, Principal, TimeMarker, etc.)
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

    finally:
        if show_progress and hasattr(iterator, "close"):
            iterator.close()

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
    # --- Events DataFrame ---
    events_df = pd.DataFrame(parsed["events"])
    if len(events_df) > 0:
        # Convert timestamp from nanoseconds to datetime for readability,
        # but keep the raw nanos for precise computation
        events_df["timestamp_nanos"] = pd.to_numeric(
            events_df["timestamp_nanos"], errors="coerce"
        )
        # Convert nanos to seconds for datetime
        valid_ts = events_df["timestamp_nanos"].notna()
        events_df.loc[valid_ts, "timestamp"] = pd.to_datetime(
            events_df.loc[valid_ts, "timestamp_nanos"], unit="ns"
        )
        # Sort by timestamp
        events_df = events_df.sort_values("timestamp_nanos").reset_index(drop=True)

        # Use categorical dtype for event type (saves memory, 18 unique values)
        events_df["type"] = events_df["type"].astype("category")

    # --- Subjects DataFrame ---
    subjects_df = pd.DataFrame(parsed["subjects"])
    if len(subjects_df) > 0:
        subjects_df["type"] = subjects_df["type"].astype("category")
        subjects_df["start_timestamp_nanos"] = pd.to_numeric(
            subjects_df["start_timestamp_nanos"], errors="coerce"
        )

    # --- Objects DataFrame ---
    objects_df = pd.DataFrame(parsed["objects"])
    if len(objects_df) > 0:
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
        "--shards",
        default="0",
        help="Which shards to parse: '0', '0,1,2', or 'all' (default: '0')",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Just load and summarize previously saved parquet files",
    )
    args = parser.parse_args()

    raw_dir = get_raw_dir()
    processed_dir = get_processed_dir()

    # --- Load-only mode ---
    if args.load_only:
        print(f"Loading from: {processed_dir}")
        for name in ["events", "subjects", "objects"]:
            path = processed_dir / f"{name}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                print(f"\n{name}: {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
                print(f"  Columns: {list(df.columns)}")
                if name == "events" and "type" in df.columns:
                    print(f"  Event types: {df['type'].value_counts().to_dict()}")
                if name == "objects" and "object_type" in df.columns:
                    print(f"  Object types: {df['object_type'].value_counts().to_dict()}")
            else:
                print(f"\n{name}: NOT FOUND at {path}")
        return

    # --- Find JSON shards ---
    all_shards = sorted([
        f for f in raw_dir.iterdir()
        if f.is_file() and ".json" in f.name and ".tar.gz" not in f.name
    ])

    if not all_shards:
        print(f"No JSON shards found in {raw_dir}")
        print("Run: python -m src.data.download_darpa_tc --extract")
        sys.exit(1)

    print(f"Available shards ({len(all_shards)}):")
    for i, s in enumerate(all_shards):
        size_gb = s.stat().st_size / (1024**3)
        print(f"  [{i}] {s.name}: {size_gb:.2f} GB")

    # Select shards to parse
    if args.shards == "all":
        selected = list(range(len(all_shards)))
    else:
        selected = [int(x.strip()) for x in args.shards.split(",")]

    selected_shards = [all_shards[i] for i in selected]
    print(f"\nParsing {len(selected_shards)} shard(s): {[s.name for s in selected_shards]}")

    # --- Parse each shard ---
    all_events = []
    all_subjects = []
    all_objects = []
    total_parse_errors = 0

    start_time = time.time()

    for shard_path in selected_shards:
        print(f"\n{'='*60}")
        print(f"Parsing: {shard_path.name}")
        print(f"{'='*60}")

        parsed = parse_shard(shard_path, show_progress=True)

        print(f"  Events:   {len(parsed['events']):,}")
        print(f"  Subjects: {len(parsed['subjects']):,}")
        print(f"  Objects:  {len(parsed['objects']):,}")

        all_events.extend(parsed["events"])
        all_subjects.extend(parsed["subjects"])
        all_objects.extend(parsed["objects"])
        total_parse_errors += parsed["parse_errors"]

    elapsed = time.time() - start_time
    print(f"\nParsing completed in {elapsed:.1f}s")

    # --- Build DataFrames ---
    print(f"\nBuilding DataFrames...")
    events_df, subjects_df, objects_df = build_dataframes({
        "events": all_events,
        "subjects": all_subjects,
        "objects": all_objects,
    })

    # Free the raw lists to save memory
    del all_events, all_subjects, all_objects

    # --- Print summaries ---
    print(f"\n{'='*60}")
    print(f"PARSING SUMMARY")
    print(f"{'='*60}")

    print(f"\nEvents: {len(events_df):,} rows")
    print(f"  Memory: {events_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    if len(events_df) > 0:
        print(f"  Time range: {events_df['timestamp'].min()} to {events_df['timestamp'].max()}")
        print(f"  Unique subjects: {events_df['subject_uuid'].nunique():,}")
        print(f"  Unique objects:  {events_df['predicate_object_uuid'].nunique():,}")
        print(f"\n  Event type distribution:")
        for etype, count in events_df["type"].value_counts().items():
            pct = 100 * count / len(events_df)
            print(f"    {etype:30s} {count:>10,}  ({pct:5.1f}%)")

    print(f"\nSubjects: {len(subjects_df):,} rows")
    if len(subjects_df) > 0:
        print(f"  Types: {subjects_df['type'].value_counts().to_dict()}")

    print(f"\nObjects: {len(objects_df):,} rows")
    if len(objects_df) > 0:
        print(f"  Types: {objects_df['object_type'].value_counts().to_dict()}")

    # --- Save to parquet ---
    print(f"\nSaving to: {processed_dir}")

    events_df.to_parquet(processed_dir / "events.parquet", index=False)
    print(f"  events.parquet: {len(events_df):,} rows")

    subjects_df.to_parquet(processed_dir / "subjects.parquet", index=False)
    print(f"  subjects.parquet: {len(subjects_df):,} rows")

    objects_df.to_parquet(processed_dir / "objects.parquet", index=False)
    print(f"  objects.parquet: {len(objects_df):,} rows")

    # Save summary
    summary = {
        "shards_parsed": [s.name for s in selected_shards],
        "total_events": len(events_df),
        "total_subjects": len(subjects_df),
        "total_objects": len(objects_df),
        "parse_errors": total_parse_errors,
        "event_types": events_df["type"].value_counts().to_dict() if len(events_df) > 0 else {},
        "object_types": objects_df["object_type"].value_counts().to_dict() if len(objects_df) > 0 else {},
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(processed_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  summary.json")

    print(f"\n✓ Done. Total time: {elapsed:.1f}s")
    print(f"  Next step: python -m src.data.darpa_tc_parser --load-only")


if __name__ == "__main__":
    main()
