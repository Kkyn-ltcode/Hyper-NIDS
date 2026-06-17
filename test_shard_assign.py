import pandas as pd
from pathlib import Path

def assign_shards(data_root):
    data_root = Path(data_root)
    VAL_START = 1523246400000000000
    TEST_START = 1523332800000000000
    
    labeled_dir = data_root / "labeled"
    if not labeled_dir.exists():
        print("No labeled dir")
        return [], [], []
        
    train_shards, val_shards, test_shards = set(), set(), set()
    
    for p in sorted(labeled_dir.glob("labeled_shard*.parquet")):
        sid = int(p.stem.replace("labeled_shard", ""))
        df = pd.read_parquet(p, columns=["timestamp_nanos"])
        t_min = df["timestamp_nanos"].min()
        t_max = df["timestamp_nanos"].max()
        
        # A shard belongs to a split if it has ANY events in that time window
        if t_min < VAL_START:
            train_shards.add(sid)
        if (t_max >= VAL_START) and (t_min < TEST_START):
            val_shards.add(sid)
        if t_max >= TEST_START:
            test_shards.add(sid)
            
    return sorted(list(train_shards)), sorted(list(val_shards)), sorted(list(test_shards))

print(assign_shards("/Users/nguyen/Documents/Work/NIDS/HyperMamba-NIDS/data/processed/darpa_tc_e3/theia"))
