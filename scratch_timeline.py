import pandas as pd
from pathlib import Path
from src.data.ground_truth import load_ground_truth

gt = load_ground_truth("theia")
data_dir = Path("data/processed/darpa_tc_e3/theia")

print("Loading data to compute timeline...")
subjects = pd.read_parquet(data_dir / "subjects.parquet")
objects = pd.read_parquet(data_dir / "objects.parquet")

# Find UUIDs for the IoCs
ioc_to_uuids = {}

# Entry process
entry_uuids = subjects[subjects["process_path"].fillna("").str.contains("firefox", case=False)]["uuid"].tolist()
ioc_to_uuids["firefox (Entry)"] = set(entry_uuids)

# IPs
for ip in gt.attack_ips:
    ip_uuids = objects[(objects["remote_address"] == ip) | (objects["local_address"] == ip)]["uuid"].tolist()
    ioc_to_uuids[f"IP: {ip}"] = set(ip_uuids)

# Files
for f in gt.malicious_file_substrings:
    file_uuids = objects[objects["filename"].fillna("").str.contains(f, case=False)]["uuid"].tolist()
    ioc_to_uuids[f"File: {f}"] = set(file_uuids)

# Processes
for p in gt.malicious_process_basenames:
    proc_uuids = subjects[subjects["process_path"].fillna("").str.contains(p, case=False)]["uuid"].tolist()
    ioc_to_uuids[f"Process: {p}"] = set(proc_uuids)

# Now scan all shards to find the *first* timestamp each UUID is seen
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
        pass

# Sort and print timeline
import datetime
timeline = []
for name, ts in first_seen.items():
    dt = datetime.datetime.fromtimestamp(ts / 1e9)
    timeline.append((ts, dt, name))

timeline.sort()

if not timeline:
    print("No events found.")
else:
    print("\nTHEIA ATTACK TIMELINE:")
    print("-" * 60)
    base_ts = timeline[0][0]
    for ts, dt, name in timeline:
        delta_sec = (ts - base_ts) / 1e9
        hours = int(delta_sec // 3600)
        minutes = int((delta_sec % 3600) // 60)
        seconds = int(delta_sec % 60)
        
        time_str = f"T+{hours:02d}:{minutes:02d}:{seconds:02d}"
        print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}]  {time_str}  --  {name}")
        
