#!/usr/bin/env python3
"""
Comprehensive Dataset Analysis Script for HyperMamba-NIDS

Usage:
    python src/analysis/analyze_datasets.py --data_root data/processed/darpa_tc_e3

This script will scan the specified data root for 'theia', 'trace', and 'cadets' datasets
and output a detailed summary of:
1. Shard distributions (events, timestamps, feature counts)
2. Label distributions (malicious event counts)
3. Entity vocabulary sizes
4. Gaps in timestamps (campaign boundaries)
5. Object and Subject metadata distributions
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def analyze_dataset(dataset_name: str, dataset_path: Path):
    print("=" * 100)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 100)

    if not dataset_path.exists():
        print(f"  [!] Path does not exist: {dataset_path}")
        return

    # --- Summary ---
    summary_path = dataset_path / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\n--- INGEST SUMMARY ---")
        for k, v in summary.items():
            print(f"  {k:<20}: {v}")
    else:
        print("  [!] summary.json not found")

    # Directories
    labeled_dir = dataset_path / "labeled"
    features_dir = dataset_path / "features_norm"
    if not features_dir.exists():
        features_dir = dataset_path / "features"

    # Find shards
    labeled_shards = sorted([
        int(f.stem.replace("labeled_shard", ""))
        for f in labeled_dir.glob("labeled_shard*.parquet")
    ]) if labeled_dir.exists() else []
    
    feature_shards = sorted([
        int(f.stem.replace("thyne_shard", ""))
        for f in features_dir.glob("thyne_shard*.npz")
    ]) if features_dir.exists() else []

    print(f"Found {len(labeled_shards)} labeled shards, {len(feature_shards)} feature shards.")
    if not labeled_shards:
        print("  [!] No labeled shards found. Skipping detailed shard analysis.")
        return

    # --- Shard-level Analysis ---
    print("\n--- SHARD SUMMARY ---")
    print(f"{'Shard':>6} | {'Events':>12} | {'Time Start':>22} | {'Time End':>22} | "
          f"{'broad+':>9} | {'xproc+':>9} | {'narrow+':>9}")
    print("-" * 105)

    shard_info = {}
    total_events = 0
    total_xproc = 0

    for sid in labeled_shards:
        l_path = labeled_dir / f"labeled_shard{sid}.parquet"
        
        # Read schema efficiently
        all_cols = pq.read_schema(l_path).names
        load_cols = ["timestamp_nanos"]
        for lc in ["label_broad", "label_crossprocess", "label_narrow"]:
            if lc in all_cols:
                load_cols.append(lc)

        # Read specific columns
        df = pd.read_parquet(l_path, columns=load_cols)
        n = len(df)
        total_events += n
        
        ts_min = pd.Timestamp(df["timestamp_nanos"].min(), unit="ns")
        ts_max = pd.Timestamp(df["timestamp_nanos"].max(), unit="ns")

        counts = {}
        for lc in ["label_broad", "label_crossprocess", "label_narrow"]:
            if lc in df.columns:
                c = int((df[lc] == 1).sum())
                counts[lc] = c
                if lc == "label_crossprocess":
                    total_xproc += c
            else:
                counts[lc] = -1

        shard_info[sid] = {
            "n_events": n,
            "ts_min": ts_min,
            "ts_max": ts_max,
            **counts,
        }

        print(f"{sid:>6} | {n:>12,} | {str(ts_min):>22} | {str(ts_max):>22} | "
              f"{counts.get('label_broad', -1):>9,} | {counts.get('label_crossprocess', -1):>9,} | "
              f"{counts.get('label_narrow', -1):>9,}")
        del df

    print(f"\nTotal Events: {total_events:,}")
    print(f"Total Crossprocess+ Events: {total_xproc:,}")

    # --- Campaign / Time Gap Detection ---
    print("\n--- TIME GAPS (CAMPAIGN BOUNDARIES) ---")
    gaps_found = 0
    for i in range(1, len(labeled_shards)):
        prev_end = shard_info[labeled_shards[i - 1]]["ts_max"]
        curr_start = shard_info[labeled_shards[i]]["ts_min"]
        gap = curr_start - prev_end
        gap_hours = gap.total_seconds() / 3600
        if gap_hours > 6:
            print(f"  Gap of {gap_hours:.1f} hours between Shard {labeled_shards[i-1]} and {labeled_shards[i]}")
            gaps_found += 1
    if gaps_found == 0:
        print("  No major time gaps (> 6 hours) found between consecutive shards.")

    # --- Entities ---
    print("\n--- ENTITY VOCABULARY & GRAPH ---")
    vocab_path = dataset_path / "graph" / "entity_vocab.npz"
    if vocab_path.exists():
        vocab = np.load(vocab_path, allow_pickle=True)
        print(f"  Total Entities: {vocab['num_entities']}")
        if 'node_types' in vocab:
             nt = pd.Series(vocab['node_types']).value_counts()
             print("  Entity Types Breakdown:")
             for t, c in nt.items():
                 print(f"    {t:<15}: {c:>10,}")
    else:
        print("  [!] entity_vocab.npz not found.")

    incidence_path = dataset_path / "graph" / "incidence.npz"
    if incidence_path.exists():
        from scipy import sparse
        try:
            H = sparse.load_npz(incidence_path)
            print(f"\n  Incidence Matrix (H):")
            print(f"    Shape:      {H.shape[0]:,} nodes x {H.shape[1]:,} hyperedges")
            print(f"    Non-zeros:  {H.nnz:,}")
            density = H.nnz / max(1, (H.shape[0] * H.shape[1]))
            print(f"    Density:    {density:.2e}")
        except Exception as e:
            print(f"  [!] Could not load incidence.npz: {e}")

    # --- Metadata Files ---
    print("\n--- METADATA FILES ---")
    subj_path = dataset_path / "subjects.parquet"
    if subj_path.exists():
        subj_df = pd.read_parquet(subj_path)
        print(f"  Subjects: {len(subj_df):,} total")
        if "process_path" in subj_df.columns:
            print("  Top 5 Process Paths:")
            for p, c in subj_df["process_path"].value_counts().head(5).items():
                print(f"    {c:>8,}x | {p}")
        del subj_df
    else:
        print("  [!] subjects.parquet not found.")

    obj_path = dataset_path / "objects.parquet"
    if obj_path.exists():
        obj_df = pd.read_parquet(obj_path)
        print(f"\n  Objects: {len(obj_df):,} total")
        if "object_type" in obj_df.columns:
            print("  Object Types:")
            for t, c in obj_df["object_type"].value_counts().items():
                print(f"    {t:<15}: {c:>10,}")
        del obj_df
    else:
        print("  [!] objects.parquet not found.")

    print("\n")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Dataset Analysis")
    parser.add_argument("--data_root", type=str, default="data/processed/darpa_tc_e3",
                        help="Root directory containing datasets (e3)")
    parser.add_argument("--datasets", type=str, nargs="+", default=["theia", "trace", "cadets"],
                        help="Specific datasets to analyze")
    args = parser.parse_args()

    root = Path(args.data_root)
    if not root.exists():
        print(f"Error: Data root {root} does not exist.")
        return

    for ds in args.datasets:
        analyze_dataset(ds, root / ds)

if __name__ == "__main__":
    main()
