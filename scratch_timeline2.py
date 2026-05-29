import pandas as pd
from pathlib import Path
from src.data.ground_truth import load_ground_truth

gt = load_ground_truth("theia")
data_dir = Path("data/processed/darpa_tc_e3/theia")

subjects = pd.read_parquet(data_dir / "subjects.parquet")
objects = pd.read_parquet(data_dir / "objects.parquet")

ioc_to_uuids = {}

# Entry process
entry_mask = subjects["process_path"].fillna("").str.contains("firefox", case=False)
ioc_to_uuids["Process: firefox (Entry)"] = set(subjects[entry_mask]["uuid"])

# Processes
for p in gt.malicious_process_basenames:
    mask = subjects["process_path"].fillna("").str.contains(p, case=False)
    ioc_to_uuids[f"Process: {p}"] = set(subjects[mask]["uuid"])

# Files
for f in gt.malicious_file_substrings:
    mask = objects["filename"].fillna("").str.contains(f, case=False)
    ioc_to_uuids[f"File: {f}"] = set(objects[mask]["uuid"])

# IPs
for ip in gt.attack_ips:
    mask = (objects["remote_address"] == ip) | (objects["local_address"] == ip)
    ioc_to_uuids[f"IP: {ip}"] = set(objects[mask]["uuid"])

for k, v in ioc_to_uuids.items():
    print(f"{k}: {len(v)} UUIDs found in dictionaries")

first_seen = {}
for sid in range(10):
    try:
        df = pd.read_parquet(data_dir / f"labeled/labeled_shard{sid}.parquet", 
                             columns=["timestamp_nanos", "subject_uuid", "predicate_object_uuid"])
        
        for name, uuids in ioc_to_uuids.items():
            if not uuids: continue
            mask = df["subject_uuid"].isin(uuids) | df["predicate_object_uuid"].isin(uuids)
            if mask.any():
                min_ts = df[mask]["timestamp_nanos"].min()
                if name not in first_seen or min_ts < first_seen[name]:
                    first_seen[name] = min_ts
    except Exception as e:
        print(f"Error reading shard {sid}: {e}")

print(first_seen)
