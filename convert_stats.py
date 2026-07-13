import numpy as np
import pickle
import time
from src.features.feature_extractor import GlobalStats
from pathlib import Path

path = Path("data/processed/darpa_tc_e3/theia/features/global_stats.npz")
out_path = Path("data/processed/darpa_tc_e3/theia/features/global_stats.pkl")

if path.exists() and not out_path.exists():
    print(f"Loading {path}...")
    t0 = time.time()
    data = np.load(path, allow_pickle=True)
    global_stats = GlobalStats(
        total_events=int(data["total_events"]),
        type_counts=data["type_counts"].item(),
        subject_first_ts=data["subject_first_ts"].item(),
        object_first_ts=data["object_first_ts"].item(),
    )
    print(f"Loaded in {time.time()-t0:.1f}s")
    
    print(f"Saving to {out_path}...")
    t0 = time.time()
    with open(out_path, 'wb') as f:
        pickle.dump(global_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved in {time.time()-t0:.1f}s")

    print("Testing fast load...")
    t0 = time.time()
    with open(out_path, 'rb') as f:
        gs = pickle.load(f)
    print(f"Fast load took: {time.time()-t0:.1f}s")
