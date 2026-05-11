"""
Download, extract, and verify DARPA TC E3 data.

This script handles the .json.tar.gz archives from the DARPA TC Google Drive,
extracts the JSON shards, and verifies the data is parseable by reading a
sample and printing record type / event type statistics.

Usage:
    # Extract tar archive and verify (after manual download from Google Drive)
    python -m src.data.download_darpa_tc --extract --verify

    # Verify only (files already extracted)
    python -m src.data.download_darpa_tc --verify

    # Verify with more records (default: 50,000)
    python -m src.data.download_darpa_tc --verify --max-records 100000

Data source (manual download):
    https://drive.google.com/drive/folders/1QlbUFWAGq3Hpl8wVdzOdIoZLFxkII4EK
    Navigate: Engagement3 > data > theia
    Download: ta1-theia-e3-official-1r.json.tar.gz (1.09 GB)

Reference:
    CDM18 schema: https://github.com/darpa-i2o/Transparent-Computing
"""

import argparse
import json
import os
import sys
import tarfile
from collections import Counter
from pathlib import Path


# ============================================================
# Data directory layout
# ============================================================
# data/raw/darpa_tc_e3/<dataset>/
#   <archive>.json.tar.gz   (downloaded archive)
#   <archive>.json          (extracted shard 0)
#   <archive>.json.1        (extracted shard 1)
# ============================================================

# CDM18 record type short names
RECORD_TYPE_SHORT = {
    "com.bbn.tc.schema.avro.cdm18.Event": "Event",
    "com.bbn.tc.schema.avro.cdm18.Subject": "Subject",
    "com.bbn.tc.schema.avro.cdm18.FileObject": "FileObject",
    "com.bbn.tc.schema.avro.cdm18.NetFlowObject": "NetFlowObject",
    "com.bbn.tc.schema.avro.cdm18.SrcSinkObject": "SrcSinkObject",
    "com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject": "UnnamedPipeObject",
    "com.bbn.tc.schema.avro.cdm18.MemoryObject": "MemoryObject",
    "com.bbn.tc.schema.avro.cdm18.Principal": "Principal",
    "com.bbn.tc.schema.avro.cdm18.Host": "Host",
    "com.bbn.tc.schema.avro.cdm18.TimeMarker": "TimeMarker",
    "com.bbn.tc.schema.avro.cdm18.StartMarker": "StartMarker",
    "com.bbn.tc.schema.avro.cdm18.UnitDependency": "UnitDependency",
    "com.bbn.tc.schema.avro.cdm18.EndMarker": "EndMarker",
}

DATASET_ARCHIVES = {
    "theia": "ta1-theia-e3-official-1r.json.tar.gz",
    "trace": "ta1-trace-e3-official-1.json.tar.gz",
    "trace-1": "ta1-trace-e3-official.json.tar.gz",
    "cadets": "ta1-cadets-e3-official-2.json.tar.gz",
}

def get_data_dir(dataset: str) -> Path:
    """Return the project data directory for DARPA TC E3 dataset."""
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data" / "raw" / "darpa_tc_e3" / dataset
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def extract_tar(data_dir: Path, dataset: str) -> list[Path]:
    """Extract .json.tar.gz archive into individual JSON shard files."""
    tar_files = sorted(data_dir.glob("*.json.tar.gz"))
    if not tar_files:
        print(f"ERROR: No .json.tar.gz files found in {data_dir}")
        print(f"\nDownload instructions:")
        print(f"  1. Go to: https://drive.google.com/drive/folders/1QlbUFWAGq3Hpl8wVdzOdIoZLFxkII4EK")
        print(f"  2. Navigate: Engagement3 > data > {dataset}")
        print(f"  3. Download: {DATASET_ARCHIVES.get(dataset, 'the appropriate archive')}")
        print(f"  4. Place in: {data_dir}")
        sys.exit(1)

    extracted = []
    for tar_path in tar_files:
        print(f"\nExtracting: {tar_path.name}")
        size_gb = tar_path.stat().st_size / (1024**3)
        print(f"  Archive size: {size_gb:.2f} GB")

        topic_name = tar_path.name.replace(".json.tar.gz", "")
        existing_shards = sorted(data_dir.glob(f"{topic_name}.json*"))
        existing_shards = [f for f in existing_shards if ".tar.gz" not in f.name]

        if existing_shards:
            print(f"  Already extracted: {len(existing_shards)} shard(s) found")
            extracted.extend(existing_shards)
            continue

        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                members = tar.getmembers()
                print(f"  Archive contains {len(members)} file(s)")
                tar.extractall(path=data_dir)
                for m in members:
                    extracted_path = data_dir / m.name
                    if extracted_path.exists():
                        size_gb = extracted_path.stat().st_size / (1024**3)
                        print(f"  Extracted: {m.name} ({size_gb:.2f} GB)")
                        extracted.append(extracted_path)
        except Exception as e:
            print(f"  ERROR extracting: {e}")
            sys.exit(1)

    return sorted(extracted)


def find_json_shards(data_dir: Path) -> list[Path]:
    """Find all extracted JSON shard files (not tar archives)."""
    shards = []
    for f in sorted(data_dir.iterdir()):
        if f.is_file() and ".json" in f.name and ".tar.gz" not in f.name:
            shards.append(f)
    return sorted(shards)


def unwrap_avro_union(value):
    if value is None:
        return None
    if isinstance(value, dict) and len(value) == 1:
        inner_key = next(iter(value))
        return value[inner_key]
    return value


def verify_shard(filepath: Path, max_records: int = 50000) -> dict:
    print(f"\nVerifying: {filepath.name}")
    size_gb = filepath.stat().st_size / (1024**3)
    print(f"  File size: {size_gb:.2f} GB")

    type_counts = Counter()
    event_type_counts = Counter()
    sample_event = None
    sample_subject = None
    sample_file_object = None
    sample_netflow = None
    total_records = 0
    parse_errors = 0

    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                if parse_errors <= 3:
                    print(f"  WARNING: Parse error on line {line_num}")
                continue

            total_records += 1

            datum = record.get("datum", {})
            if not datum:
                continue

            record_type = next(iter(datum))
            record_data = datum[record_type]
            short_type = RECORD_TYPE_SHORT.get(record_type, record_type.split(".")[-1])

            type_counts[short_type] += 1

            if short_type == "Event" and isinstance(record_data, dict):
                event_type = record_data.get("type", "UNKNOWN")
                event_type_counts[event_type] += 1
                if sample_event is None:
                    sample_event = record_data

            if short_type == "Subject" and sample_subject is None:
                sample_subject = record_data
            if short_type == "FileObject" and sample_file_object is None:
                sample_file_object = record_data
            if short_type == "NetFlowObject" and sample_netflow is None:
                sample_netflow = record_data

            if max_records > 0 and total_records >= max_records:
                print(f"  (Stopped after {max_records:,} records for verification)")
                break

    result = {
        "total_records": total_records,
        "parse_errors": parse_errors,
        "record_type_counts": dict(type_counts),
        "event_type_counts": dict(event_type_counts),
        "sample_event": sample_event,
        "sample_subject": sample_subject,
        "sample_file_object": sample_file_object,
        "sample_netflow": sample_netflow,
    }

    print(f"  Records read: {total_records:,}")
    if parse_errors:
        print(f"  Parse errors: {parse_errors}")

    print(f"\n  Record types:")
    for rtype, count in type_counts.most_common():
        pct = 100 * count / total_records if total_records > 0 else 0
        print(f"    {rtype:20s} {count:>10,}  ({pct:5.1f}%)")

    if event_type_counts:
        print(f"\n  Event subtypes ({len(event_type_counts)} unique):")
        for etype, count in event_type_counts.most_common():
            print(f"    {etype:30s} {count:>10,}")

    return result


def print_sample_records(result: dict):
    print(f"\n{'='*60}")
    print("SAMPLE RECORDS (for verifying field structure)")
    print(f"{'='*60}")

    if result.get("sample_event"):
        ev = result["sample_event"]
        print(f"\n--- Sample Event ---")
        print(f"  uuid:            {ev.get('uuid')}")
        print(f"  type:            {ev.get('type')}")
        print(f"  timestampNanos:  {ev.get('timestampNanos')}")
        print(f"  subject:         {unwrap_avro_union(ev.get('subject'))}")
        print(f"  predicateObject: {unwrap_avro_union(ev.get('predicateObject'))}")
        print(f"  predicateObject2:{unwrap_avro_union(ev.get('predicateObject2'))}")
        print(f"  name:            {unwrap_avro_union(ev.get('name'))}")
        print(f"  size:            {unwrap_avro_union(ev.get('size'))}")

    if result.get("sample_subject"):
        sub = result["sample_subject"]
        print(f"\n--- Sample Subject ---")
        print(f"  uuid:            {sub.get('uuid')}")
        print(f"  type:            {sub.get('type')}")
        print(f"  cid (pid):       {sub.get('cid')}")
        print(f"  parentSubject:   {unwrap_avro_union(sub.get('parentSubject'))}")
        print(f"  cmdLine:         {unwrap_avro_union(sub.get('cmdLine'))}")

    if result.get("sample_file_object"):
        fo = result["sample_file_object"]
        print(f"\n--- Sample FileObject ---")
        print(f"  uuid:            {fo.get('uuid')}")

    if result.get("sample_netflow"):
        nf = result["sample_netflow"]
        print(f"\n--- Sample NetFlowObject ---")
        print(f"  uuid:            {nf.get('uuid')}")


def main():
    parser = argparse.ArgumentParser(description="Extract and verify DARPA TC E3 data")
    parser.add_argument("--extract", action="store_true", help="Extract .json.tar.gz archives")
    parser.add_argument("--verify", action="store_true", help="Verify extracted JSON shard files")
    parser.add_argument("--max-records", type=int, default=50000, help="Max records to read (0=all)")
    parser.add_argument("--shard", type=int, default=0, help="Which shard to verify")
    parser.add_argument("--dataset", type=str, default="theia", choices=["theia", "trace", "cadets", "trace-1"], help="Which dataset to process")
    args = parser.parse_args()

    if not args.extract and not args.verify:
        print("Specify --extract, --verify, or both.")
        parser.print_help()
        sys.exit(1)

    data_dir = get_data_dir(args.dataset)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Data directory: {data_dir}")

    if args.extract:
        extracted = extract_tar(data_dir, args.dataset)
        print(f"\nExtracted {len(extracted)} shard file(s)")

    if args.verify:
        shards = find_json_shards(data_dir)
        if not shards:
            print(f"\nNo extracted JSON files found in {data_dir}")
            print(f"Run: python -m src.data.download_darpa_tc --extract --dataset {args.dataset}")
            sys.exit(1)

        print(f"\nFound {len(shards)} JSON shard(s):")
        for i, s in enumerate(shards):
            size_gb = s.stat().st_size / (1024**3)
            marker = " <-- verifying" if i == args.shard else ""
            print(f"  [{i}] {s.name}: {size_gb:.2f} GB{marker}")

        if args.shard >= len(shards):
            print(f"ERROR: shard {args.shard} out of range (0-{len(shards)-1})")
            sys.exit(1)

        target_shard = shards[args.shard]
        print(f"\n{'='*60}")
        print(f"VERIFYING SHARD {args.shard}: {target_shard.name}")
        print(f"{'='*60}")

        result = verify_shard(target_shard, max_records=args.max_records)
        print_sample_records(result)

        print(f"\n{'='*60}")
        print(f"VERIFICATION COMPLETE")
        print(f"{'='*60}")
        total_events = sum(result.get("event_type_counts", {}).values())
        print(f"  Records sampled:    {result['total_records']:,}")
        print(f"  Events found:       {total_events:,}")
        print(f"\n  ✓ Data pipeline verified for {args.dataset.upper()}.")


if __name__ == "__main__":
    main()
