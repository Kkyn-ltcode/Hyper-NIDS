"""
Diagnostic: scan ALL shards to find where positive labels are,
what timestamp ranges they cover, and how many events are in each.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timezone

DATA_ROOT = Path("data/processed/darpa_tc_e3/theia")
labeled_dir = DATA_ROOT / "labeled"
features_dir = DATA_ROOT / "features_norm"

# Find all shards
shard_files = sorted(labeled_dir.glob("labeled_shard*.parquet"),
                     key=lambda f: int(f.stem.replace("labeled_shard", "")))
feature_files = sorted(features_dir.glob("thyne_shard*.npz"),
                       key=lambda f: int(f.stem.replace("thyne_shard", "")))

print(f"Found {len(shard_files)} labeled shards, {len(feature_files)} feature shards")
print()

print(f"{'Shard':>7} | {'Events':>10} | {'Time Start':>19} | {'Time End':>19} | "
      f"{'broad+':>8} | {'xproc+':>8} | {'narrow+':>8} | {'ioc+':>8}")
print("-" * 120)

summary = []
for f in shard_files:
    sid = int(f.stem.replace("labeled_shard", ""))
    
    # Check which label columns exist
    schema = pq.read_schema(f)
    col_names = schema.names
    
    cols_to_read = ["timestamp_nanos"]
    for lbl in ["label_broad", "label_crossprocess", "label_narrow", "label_ioc"]:
        if lbl in col_names:
            cols_to_read.append(lbl)
    
    df = pd.read_parquet(f, columns=cols_to_read)
    n = len(df)
    
    ts = df["timestamp_nanos"].values.astype(np.int64)
    ts_min = int(ts.min())
    ts_max = int(ts.max())
    
    # Handle epoch-0 timestamps gracefully
    try:
        dt_min = datetime.fromtimestamp(ts_min / 1e9, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    except (OSError, ValueError):
        dt_min = f"epoch({ts_min})"
    try:
        dt_max = datetime.fromtimestamp(ts_max / 1e9, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
    except (OSError, ValueError):
        dt_max = f"epoch({ts_max})"
    
    broad_pos = int(df["label_broad"].sum()) if "label_broad" in df.columns else -1
    xproc_pos = int(df["label_crossprocess"].sum()) if "label_crossprocess" in df.columns else -1
    narrow_pos = int(df["label_narrow"].sum()) if "label_narrow" in df.columns else -1
    ioc_pos = int(df["label_ioc"].sum()) if "label_ioc" in df.columns else -1
    
    print(f"{sid:>7} | {n:>10,} | {dt_min:>19} | {dt_max:>19} | "
          f"{broad_pos:>8,} | {xproc_pos:>8,} | {narrow_pos:>8,} | {ioc_pos:>8,}")
    
    summary.append({
        "shard": sid, "events": n,
        "ts_min": ts_min, "ts_max": ts_max,
        "dt_min": dt_min, "dt_max": dt_max,
        "broad+": broad_pos, "xproc+": xproc_pos,
        "narrow+": narrow_pos, "ioc+": ioc_pos,
    })
    del df

print()
print("=" * 80)
print("SHARDS WITH CROSSPROCESS POSITIVES (for val/test split decisions):")
print("=" * 80)
for s in summary:
    if s["xproc+"] > 0:
        print(f"  Shard {s['shard']:>3}: {s['xproc+']:>8,} xproc+ events  "
              f"({s['dt_min']} to {s['dt_max']})")

print()
print("SHARDS WITH ZERO CROSSPROCESS POSITIVES (benign-only):")
for s in summary:
    if s["xproc+"] == 0:
        print(f"  Shard {s['shard']:>3}: {s['events']:>10,} events  "
              f"({s['dt_min']} to {s['dt_max']})")
