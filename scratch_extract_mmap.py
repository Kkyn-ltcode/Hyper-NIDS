import pandas as pd
import numpy as np
from pathlib import Path

# Need to find an EVENT_MMAP with a known malicious label and predicateObject2 not null
labeled_dir = Path("data/processed/darpa_tc_e3/theia/labeled")

def find_mmap_example():
    for sid in range(12, 16): # Theia test set shards have attacks
        parquet_path = labeled_dir / f"labeled_shard{sid}.parquet"
        if not parquet_path.exists():
            continue
            
        print(f"Reading {parquet_path}...")
        df = pd.read_parquet(
            parquet_path,
            columns=[
                "type", "label_broad", "label_crossprocess",
                "subject_uuid", "predicate_object_uuid", "predicate_object2_uuid",
                "timestamp_nanos"
            ]
        )
        
        # Filter for EVENT_MMAP
        mmap_df = df[df["type"] == "EVENT_MMAP"]
        
        # Malicious ones
        malicious = mmap_df[(mmap_df["label_crossprocess"] == 1) & (mmap_df["predicate_object2_uuid"].notnull()) & (mmap_df["predicate_object2_uuid"] != "00000000-0000-0000-0000-000000000000")]
        
        if not malicious.empty:
            print(f"Found {len(malicious)} malicious 3-entity EVENT_MMAP interactions in shard {sid}!")
            ex = malicious.iloc[0]
            print("\nExample Details:")
            print(f"Event Type: {ex['type']}")
            print(f"Subject UUID: {ex['subject_uuid']}")
            print(f"Object UUID: {ex['predicate_object_uuid']}")
            print(f"Object2 UUID: {ex['predicate_object2_uuid']}")
            print(f"Timestamp: {ex['timestamp_nanos']}")
            print(f"Label Broad: {ex['label_broad']}")
            print(f"Label Crossprocess: {ex['label_crossprocess']}")
            
            # Let's also find a benign one to compare
            benign = mmap_df[(mmap_df["label_broad"] == 0) & (mmap_df["predicate_object2_uuid"].notnull()) & (mmap_df["predicate_object2_uuid"] != "00000000-0000-0000-0000-000000000000")]
            if not benign.empty:
                b_ex = benign.iloc[0]
                print("\nBenign Example Details:")
                print(f"Event Type: {b_ex['type']}")
                print(f"Subject UUID: {b_ex['subject_uuid']}")
                print(f"Object UUID: {b_ex['predicate_object_uuid']}")
                print(f"Object2 UUID: {b_ex['predicate_object2_uuid']}")
                print(f"Timestamp: {b_ex['timestamp_nanos']}")
                print(f"Label Broad: {b_ex['label_broad']}")
            
            return

    print("No examples found.")

if __name__ == "__main__":
    find_mmap_example()
