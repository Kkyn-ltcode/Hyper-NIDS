"""
Diagnostic script to investigate OOM at shard 109.
Run from the HyperMamba-NIDS root:
    python src/pipeline/diagnose_shard109.py
"""
import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd

DATA_ROOT = Path("data/processed/darpa_tc_e3/trace")
LABELED_DIR = DATA_ROOT / "labeled"
FEATURES_DIR = DATA_ROOT / "features"

def _extract_idx(f):
    return int(f.name.replace("labeled_shard", "").replace(".parquet", ""))

shard_files = sorted(LABELED_DIR.glob("labeled_shard*.parquet"), key=_extract_idx)

print("=" * 70)
print("DIAGNOSTIC: Investigating OOM at Shard 109")
print("=" * 70)

# 1. Shard sizes
print("\n--- SHARD SIZES (rows) ---")
sizes = []
import pyarrow.parquet as pq
for f in shard_files:
    meta = pq.read_metadata(f)
    idx = _extract_idx(f)
    sizes.append((idx, meta.num_rows))

# Print a summary table: top 20 largest shards
sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)
print(f"{'Shard':>8} {'Rows':>12}")
print("-" * 22)
for idx, n in sizes_sorted[:20]:
    marker = " <-- THIS ONE" if idx == 109 else ""
    print(f"{idx:>8} {n:>12,}{marker}")

# Find shard 109 specifically
shard109_rows = dict(sizes).get(109, None)
print(f"\nShard 109 rows: {shard109_rows:,}" if shard109_rows else "\nShard 109 not found!")

# 2. Estimate subject_carry size at shard 109
print("\n--- SUBJECT CARRY GROWTH ---")
print("Simulating sequential carry accumulation (lightweight)...")
cumulative_subjects = set()
carry_sizes = []
for i, f in enumerate(shard_files):
    idx = _extract_idx(f)
    df = pd.read_parquet(f, columns=["subject_uuid"])
    unique_subs = set(df["subject_uuid"].dropna().unique())
    cumulative_subjects.update(unique_subs)
    carry_sizes.append((idx, len(cumulative_subjects)))
    del df
    gc.collect()
    
    if idx % 20 == 0 or idx == 109 or idx == 108 or idx == 110:
        print(f"  After shard {idx:>3}: {len(cumulative_subjects):>10,} unique subjects in carry")

print(f"\n--- CARRY SIZE AT SHARD 109 ---")
for idx, size in carry_sizes:
    if idx == 108:
        print(f"  Subjects in carry when processing shard 109: {size:,}")
        est_bytes = size * 100
        print(f"  Estimated carry dict memory: {est_bytes / 1e9:.2f} GB")
    if idx == 109:
        print(f"  Subjects after shard 109: {size:,}")

# 3. Check global_stats size
print("\n--- GLOBAL STATS SIZE ---")
global_stats_path = FEATURES_DIR / "global_stats.npz"
if global_stats_path.exists():
    data = np.load(global_stats_path, allow_pickle=True)
    n_types = len(data["type_counts"].item())
    n_sub_first = len(data["subject_first_ts"].item())
    n_obj_first = len(data["object_first_ts"].item())
    print(f"  type_counts:      {n_types:,} entries")
    print(f"  subject_first_ts: {n_sub_first:,} entries (~{n_sub_first * 100 / 1e9:.2f} GB as dict)")
    print(f"  object_first_ts:  {n_obj_first:,} entries (~{n_obj_first * 100 / 1e9:.2f} GB as dict)")
    
    est_series = (n_sub_first + n_obj_first) * 50
    print(f"  Estimated total global_stats as pd.Series: {est_series / 1e9:.2f} GB")
    del data
else:
    print("  global_stats.npz not found!")

# 4. Check which shard .npz files already exist
print("\n--- EXISTING FEATURE FILES ---")
existing = sorted(FEATURES_DIR.glob("thyne_shard*.npz"), key=lambda f: int(f.name.replace("thyne_shard", "").replace(".npz", "")))
if existing:
    last_idx = int(existing[-1].name.replace("thyne_shard", "").replace(".npz", ""))
    print(f"  Total feature files: {len(existing)}")
    print(f"  Last completed shard: {last_idx}")
else:
    print("  No feature files found!")

# 5. Memory estimate for processing shard 109
print("\n--- MEMORY ESTIMATE FOR SHARD 109 ---")
if shard109_rows:
    df_mem = shard109_rows * 20 * 8
    print(f"  DataFrame load:  ~{df_mem / 1e9:.2f} GB")
    feat_mem = shard109_rows * 35 * 4
    print(f"  Feature matrix:  ~{feat_mem / 1e9:.2f} GB")
    print(f"  Intermediates:   ~{df_mem * 3 / 1e9:.2f} GB (estimate)")
    
    objects_path = DATA_ROOT / "objects.parquet"
    if objects_path.exists():
        obj_meta = pq.read_metadata(objects_path)
        print(f"  objects.parquet:  {obj_meta.num_rows:,} rows")

print("\n" + "=" * 70)
print("DONE. Please share this output!")
print("=" * 70)
