"""
Inspect network connections in the DARPA TC datasets.

This script extracts all unique remote IPs that processes communicated with,
and shows which processes connected to which IPs. It highlights IPs that
are known attack IPs from the ground truth.

Usage:
    python -m src.data.inspect_ips --dataset trace
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from src.data.ground_truth import load_ground_truth

def main():
    parser = argparse.ArgumentParser(description="Inspect IPs in dataset")
    parser.add_argument("--dataset", type=str, default="trace", choices=["theia", "trace", "trace-1"])
    parser.add_argument("--top", type=int, default=50, help="Number of top IPs to show")
    args = parser.parse_args()

    processed_dir = Path(f"data/processed/darpa_tc_e3/{args.dataset}")
    if not processed_dir.exists():
        print(f"Error: {processed_dir} does not exist. Run parsing first.")
        return

    # Load ground truth
    gt = load_ground_truth(args.dataset)
    known_attack_ips = gt.attack_ips

    print(f"Loading data for {args.dataset}...")
    
    # Load just the columns we need to save memory
    events_df = pd.read_parquet(
        processed_dir / "events.parquet", 
        columns=["subject_uuid", "predicate_object_uuid", "type"]
    )
    subjects_df = pd.read_parquet(
        processed_dir / "subjects.parquet",
        columns=["uuid", "process_path", "cmd_line"]
    )
    
    # Check what columns exist in objects_df
    import pyarrow.parquet as pq
    obj_schema = pq.read_schema(processed_dir / "objects.parquet")
    obj_cols = ["uuid", "object_type"]
    if "remote_address" in obj_schema.names:
        obj_cols.append("remote_address")
    if "remote_port" in obj_schema.names:
        obj_cols.append("remote_port")
        
    objects_df = pd.read_parquet(processed_dir / "objects.parquet", columns=obj_cols)

    print(f"  Events: {len(events_df):,}")
    print(f"  Subjects: {len(subjects_df):,}")
    print(f"  Objects: {len(objects_df):,}")

    if "remote_address" not in objects_df.columns:
        print("No 'remote_address' column found in objects.parquet!")
        return

    # Filter to only NETFLOW objects
    netflow_objs = objects_df[objects_df["object_type"] == "NETFLOW"].copy()
    netflow_objs = netflow_objs[netflow_objs["remote_address"].notna()]
    netflow_objs = netflow_objs[netflow_objs["remote_address"] != ""]
    print(f"\nFound {len(netflow_objs):,} NETFLOW objects with remote_address.")

    # Find events that touch these netflows
    netflow_events = events_df[events_df["predicate_object_uuid"].isin(netflow_objs["uuid"])]
    print(f"Found {len(netflow_events):,} events interacting with these netflows.")

    # Join to get subject names and IP addresses
    sub_map = {}
    for _, row in subjects_df.iterrows():
        path = str(row.get("process_path", ""))
        cmd = str(row.get("cmd_line", ""))
        if cmd and cmd != "nan":
            name = cmd
        elif path and path != "nan":
            name = path
        else:
            name = "unknown"
        sub_map[row["uuid"]] = name
        
    ip_map = {}
    for _, row in netflow_objs.iterrows():
        ip = str(row["remote_address"])
        port = str(row.get("remote_port", ""))
        if port and port != "nan":
            ip_map[row["uuid"]] = f"{ip}:{port}"
        else:
            ip_map[row["uuid"]] = ip
            
    pure_ip_map = {row["uuid"]: str(row["remote_address"]) for _, row in netflow_objs.iterrows()}

    # Tally up
    ip_counts = {}
    ip_to_subjects = {}
    
    for _, row in netflow_events.iterrows():
        obj_uuid = row["predicate_object_uuid"]
        sub_uuid = row["subject_uuid"]
        
        ip = pure_ip_map.get(obj_uuid)
        full_addr = ip_map.get(obj_uuid)
        sub_name = sub_map.get(sub_uuid, "unknown")
        
        if not ip:
            continue
            
        ip_counts[ip] = ip_counts.get(ip, 0) + 1
        
        if ip not in ip_to_subjects:
            ip_to_subjects[ip] = set()
        ip_to_subjects[ip].add(sub_name)

    # Sort by frequency
    sorted_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*80)
    print("KNOWN ATTACK IPs FOUND IN DATA:")
    print("="*80)
    found_attack_ips = False
    for ip, count in sorted_ips:
        if ip in known_attack_ips:
            found_attack_ips = True
            print(f"[ATTACK] {ip:<15} : {count:>10,} events")
            subs = list(ip_to_subjects[ip])
            for s in subs[:5]:
                s_short = s if len(s) < 60 else s[:57] + "..."
                print(f"    ↳ {s_short}")
                
    if not found_attack_ips:
        print("None of the hardcoded attack IPs were found in this dataset's netflows.")
        
if __name__ == "__main__":
    main()
