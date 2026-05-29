import pandas as pd
from pathlib import Path
import numpy as np
import datetime

data_dir = Path("data/processed/darpa_tc_e3/theia")

print("Loading subjects to map UUIDs to basenames...")
subjects = pd.read_parquet(data_dir / "subjects.parquet", columns=["uuid", "process_path"])
def get_basename(path):
    if not path or pd.isna(path): return "unknown"
    return path.rstrip("/").rsplit("/", 1)[-1].lower()

subjects["basename"] = subjects["process_path"].apply(get_basename)
uuid_to_basename = dict(zip(subjects["uuid"], subjects["basename"]))

print("Scanning shards for attack events...")
first_seen = {}

for sid in range(10):
    try:
        df = pd.read_parquet(data_dir / f"labeled/labeled_shard{sid}.parquet", 
                             columns=["timestamp_nanos", "subject_uuid", "label_broad"])
        
        attacks = df[df["label_broad"] == 1]
        if not attacks.empty:
            for uuid, ts in zip(attacks["subject_uuid"], attacks["timestamp_nanos"]):
                basename = uuid_to_basename.get(uuid, "unknown")
                if basename not in first_seen or ts < first_seen[basename]:
                    first_seen[basename] = ts
    except Exception as e:
        print(f"Error reading shard {sid}: {e}")

# Sort and print timeline
timeline = []
for name, ts in first_seen.items():
    dt = datetime.datetime.fromtimestamp(ts / 1e9)
    timeline.append((ts, dt, name))

timeline.sort()

if not timeline:
    print("No attack events found.")
else:
    print("\nTHEIA EXPERIMENT 3 ATTACK TIMELINE:")
    print("-" * 60)
    base_ts = timeline[0][0]
    for ts, dt, name in timeline:
        delta_sec = (ts - base_ts) / 1e9
        hours = int(delta_sec // 3600)
        minutes = int((delta_sec % 3600) // 60)
        seconds = int(delta_sec % 60)
        
        time_str = f"T+{hours:02d}:{minutes:02d}:{seconds:02d}"
        print(f"[{dt.strftime('%Y-%m-%d %H:%M:%S')}]  {time_str}  --  Process: {name}")

