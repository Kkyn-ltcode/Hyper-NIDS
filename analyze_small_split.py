import pandas as pd
import numpy as np
import os

print("--- Analysis of Process Paths in Small Split ---")

base_dir = "data/processed/darpa_tc_e3/theia"

# Load subjects
try:
    subjects = pd.read_parquet(f"{base_dir}/subjects.parquet")
    uuid_to_path = dict(zip(subjects["uuid"], subjects["process_path"]))
    print(f"Loaded {len(subjects)} subjects")
except Exception as e:
    print(f"Error loading subjects: {e}")
    exit(1)

# Training shards: 0-7
train_attack_paths = set()
for i in range(8):
    try:
        df = pd.read_parquet(f"{base_dir}/labeled/labeled_shard{i}.parquet")
        if "label_crossprocess" in df.columns:
            attack = df[df["label_crossprocess"] == 1]["subject_uuid"].unique()
            paths = {uuid_to_path.get(u, "UNKNOWN") for u in attack}
            train_attack_paths.update(paths)
    except Exception:
        pass

print(f"Malicious process paths in TRAINING (shards 0-7):")
for p in train_attack_paths:
    print(f"  - {p}")

# Test shards: 8, 9
test_attack_paths = set()
for i in [8, 9]:
    try:
        df = pd.read_parquet(f"{base_dir}/labeled/labeled_shard{i}.parquet")
        if "label_crossprocess" in df.columns:
            attack_uuids = df[df["label_crossprocess"] == 1]["subject_uuid"].unique()
            paths = {uuid_to_path.get(u, "UNKNOWN") for u in attack_uuids}
            test_attack_paths.update(paths)
            print(f"Shard {i}: {len(attack_uuids)} malicious subject UUIDs")
            for u in attack_uuids:
                p = uuid_to_path.get(u, "UNKNOWN")
                print(f"  - UUID {u} -> Path: {p}")
    except Exception as e:
        print(f"Error reading shard {i}: {e}")

print(f"\nMalicious process paths in TEST (shards 8-9):")
for p in test_attack_paths:
    print(f"  - {p}")

overlap = test_attack_paths.intersection(train_attack_paths)
print(f"\nOverlap: {len(overlap)} of {len(test_attack_paths)} test paths were seen as malicious in training.")
