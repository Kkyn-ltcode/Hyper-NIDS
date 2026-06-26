#!/usr/bin/env python3
"""
Diagnostic script for HyperMamba cross-campaign analysis.
Run this on the machine that has ALL 25 shards.

Usage:
    python diagnostic_cross_campaign.py --data_root /path/to/data/processed/darpa_tc_e3/theia

Output: prints everything to stdout. Pipe to a file if needed:
    python diagnostic_cross_campaign.py --data_root /path/to/theia > diagnostic_output.txt
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to data/processed/darpa_tc_e3/theia")
    args = parser.parse_args()
    root = Path(args.data_root)

    labeled_dir = root / "labeled"
    features_dir = root / "features"
    if not features_dir.exists():
        features_dir = root / "features_norm"

    # ── 1. Discover available shards ──────────────────────────────
    labeled_shards = sorted([
        int(f.stem.replace("labeled_shard", ""))
        for f in labeled_dir.glob("labeled_shard*.parquet")
    ])
    feature_shards = sorted([
        int(f.stem.replace("thyne_shard", ""))
        for f in features_dir.glob("thyne_shard*.npz")
    ]) if features_dir.exists() else []

    print(f"Found {len(labeled_shards)} labeled shards, {len(feature_shards)} feature shards")
    print(f"Labeled: {labeled_shards}")
    print(f"Feature: {feature_shards}")
    print()

    # ── 2. Per-shard summary ──────────────────────────────────────
    print("=" * 120)
    print(f"{'Shard':>6} | {'Events':>10} | {'Time Start':>21} | {'Time End':>21} | "
          f"{'broad+':>9} | {'xproc+':>9} | {'narrow+':>9} | {'ioc+':>9}")
    print("-" * 120)

    shard_info = {}
    per_shard_malicious_subjects = {}

    for sid in labeled_shards:
        path = labeled_dir / f"labeled_shard{sid}.parquet"
        all_cols = pq.read_schema(path).names

        load_cols = ["timestamp_nanos", "subject_uuid"]
        for lc in ["label_broad", "label_crossprocess", "label_narrow", "label_ioc"]:
            if lc in all_cols:
                load_cols.append(lc)

        df = pd.read_parquet(path, columns=load_cols)
        n = len(df)
        ts_min = pd.Timestamp(df["timestamp_nanos"].min(), unit="ns")
        ts_max = pd.Timestamp(df["timestamp_nanos"].max(), unit="ns")

        counts = {}
        for lc in ["label_broad", "label_crossprocess", "label_narrow", "label_ioc"]:
            if lc in df.columns:
                counts[lc] = int((df[lc] == 1).sum())
            else:
                counts[lc] = -1

        if "label_crossprocess" in df.columns:
            mal_mask = df["label_crossprocess"] == 1
            per_shard_malicious_subjects[sid] = set(df.loc[mal_mask, "subject_uuid"].unique())
        else:
            per_shard_malicious_subjects[sid] = set()

        shard_info[sid] = {
            "n_events": n,
            "ts_min": str(ts_min),
            "ts_max": str(ts_max),
            **counts,
        }

        print(f"{sid:>6} | {n:>10,} | {str(ts_min):>21} | {str(ts_max):>21} | "
              f"{counts.get('label_broad', -1):>9,} | {counts.get('label_crossprocess', -1):>9,} | "
              f"{counts.get('label_narrow', -1):>9,} | {counts.get('label_ioc', -1):>9,}")

        del df

    print()

    # ── 3. Campaign boundary detection ────────────────────────────
    print("=" * 80)
    print("CAMPAIGN BOUNDARY DETECTION (gaps > 6 hours)")
    print("-" * 80)

    campaigns = []
    current_campaign = [labeled_shards[0]]

    for i in range(1, len(labeled_shards)):
        prev_end = pd.Timestamp(shard_info[labeled_shards[i - 1]]["ts_max"])
        curr_start = pd.Timestamp(shard_info[labeled_shards[i]]["ts_min"])
        gap = curr_start - prev_end
        gap_hours = gap.total_seconds() / 3600

        if gap_hours > 6:
            campaigns.append(current_campaign)
            print(f"  GAP: {gap_hours:.1f}h between shard {labeled_shards[i-1]} and {labeled_shards[i]}")
            current_campaign = [labeled_shards[i]]
        else:
            current_campaign.append(labeled_shards[i])

    campaigns.append(current_campaign)

    print()
    for ci, camp in enumerate(campaigns):
        n_events = sum(shard_info[s]["n_events"] for s in camp)
        n_xproc = sum(max(shard_info[s].get("label_crossprocess", 0), 0) for s in camp)
        print(f"  Campaign {ci}: shards {camp}, {n_events:,} events, {n_xproc:,} crossprocess+")
    print()

    # ── 4. Cross-campaign entity overlap ──────────────────────────
    print("=" * 80)
    print("CROSS-CAMPAIGN MALICIOUS SUBJECT OVERLAP")
    print("-" * 80)

    subj_path = root / "subjects.parquet"
    if subj_path.exists():
        subj_df = pd.read_parquet(subj_path, columns=["uuid", "process_path", "cmd_line"])
        uuid_to_path = dict(zip(subj_df["uuid"], subj_df["process_path"]))
        uuid_to_cmd = dict(zip(subj_df["uuid"], subj_df["cmd_line"]))
    else:
        print("  WARNING: subjects.parquet not found")
        uuid_to_path = {}
        uuid_to_cmd = {}

    for ci, camp in enumerate(campaigns):
        camp_subjs = set()
        for sid in camp:
            camp_subjs.update(per_shard_malicious_subjects.get(sid, set()))

        print(f"\n  Campaign {ci} (shards {camp}):")
        print(f"    Total malicious subject UUIDs: {len(camp_subjs)}")

        path_counts = defaultdict(int)
        for u in camp_subjs:
            p = uuid_to_path.get(u, "UNKNOWN")
            path_counts[p] += 1

        print(f"    Unique process paths: {len(path_counts)}")
        for p, c in sorted(path_counts.items(), key=lambda x: -x[1]):
            print(f"      {c:>5}x  {p}")

    if len(campaigns) >= 2:
        print(f"\n  --- Pairwise Campaign Overlap ---")
        for i in range(len(campaigns)):
            for j in range(i + 1, len(campaigns)):
                ci_subjs = set()
                for sid in campaigns[i]:
                    ci_subjs.update(per_shard_malicious_subjects.get(sid, set()))
                cj_subjs = set()
                for sid in campaigns[j]:
                    cj_subjs.update(per_shard_malicious_subjects.get(sid, set()))

                uuid_overlap = ci_subjs & cj_subjs
                ci_paths = {uuid_to_path.get(u, "UNKNOWN") for u in ci_subjs}
                cj_paths = {uuid_to_path.get(u, "UNKNOWN") for u in cj_subjs}
                path_overlap = ci_paths & cj_paths

                print(f"\n  Campaign {i} vs Campaign {j}:")
                print(f"    UUID overlap:         {len(uuid_overlap)} / {len(ci_subjs)} vs {len(cj_subjs)}")
                print(f"    Process path overlap:  {len(path_overlap)} / {len(ci_paths)} vs {len(cj_paths)}")
                print(f"    Shared paths: {path_overlap}")
                print(f"    C{i}-only paths: {ci_paths - cj_paths}")
                print(f"    C{j}-only paths: {cj_paths - ci_paths}")

    # ── 5. Malicious event details per campaign ───────────────────
    print()
    print("=" * 80)
    print("MALICIOUS EVENT DETAILS PER CAMPAIGN")
    print("-" * 80)

    for ci, camp in enumerate(campaigns):
        print(f"\n  Campaign {ci}:")
        event_type_counts = defaultdict(int)
        obj_uuid_set = set()
        obj2_uuid_set = set()

        for sid in camp:
            path = labeled_dir / f"labeled_shard{sid}.parquet"
            all_cols = pq.read_schema(path).names
            load_cols = ["type", "predicate_object_uuid", "label_crossprocess"]
            if "predicate_object2_uuid" in all_cols:
                load_cols.append("predicate_object2_uuid")

            df = pd.read_parquet(path, columns=load_cols)
            if "label_crossprocess" not in df.columns:
                continue
            mal = df[df["label_crossprocess"] == 1]

            for et, c in mal["type"].value_counts().items():
                event_type_counts[et] += c

            obj_uuid_set.update(mal["predicate_object_uuid"].unique())
            if "predicate_object2_uuid" in mal.columns:
                obj2_uuid_set.update(mal["predicate_object2_uuid"].unique())

            del df, mal

        print("    Event type distribution (crossprocess+ events):")
        for et, c in sorted(event_type_counts.items(), key=lambda x: -x[1]):
            print(f"      {c:>10,}  {et}")

        # Look up object types
        obj_path = root / "objects.parquet"
        if obj_path.exists():
            obj_df = pd.read_parquet(obj_path, columns=["uuid", "object_type"])
            uuid_to_otype = dict(zip(obj_df["uuid"], obj_df["object_type"]))

            otype_counts = defaultdict(int)
            for u in obj_uuid_set:
                otype_counts[uuid_to_otype.get(u, "UNKNOWN")] += 1
            print(f"    Unique predicate_objects: {len(obj_uuid_set)}")
            print(f"    Object type breakdown:")
            for ot, c in sorted(otype_counts.items(), key=lambda x: -x[1]):
                print(f"      {c:>6}  {ot}")

    # ── 6. Feature availability ───────────────────────────────────
    print()
    print("=" * 80)
    print("FEATURE FILE INFO")
    print("-" * 80)

    if features_dir.exists():
        feat_names_path = features_dir / "feature_names.txt"
        if not feat_names_path.exists():
            feat_names_path = root / "features" / "feature_names.txt"
        if feat_names_path.exists():
            names = feat_names_path.read_text().strip().split("\n")
            print(f"  Feature names ({len(names)}):")
            for n in names:
                print(f"    {n}")

        if feature_shards:
            d = np.load(features_dir / f"thyne_shard{feature_shards[0]}.npz")
            print(f"\n  Sample shard shape: {d['X'].shape}")
            print(f"  Keys: {list(d.keys())}")
    print()

    # ── 7. Labeled shard columns ──────────────────────────────────
    print("=" * 80)
    print("LABELED SHARD COLUMNS")
    print("-" * 80)
    if labeled_shards:
        path = labeled_dir / f"labeled_shard{labeled_shards[0]}.parquet"
        cols = pq.read_schema(path).names
        print(f"  Columns ({len(cols)}): {cols}")
    print()

    # ── 8. Entity metadata ────────────────────────────────────────
    print("=" * 80)
    print("ENTITY METADATA SUMMARY")
    print("-" * 80)

    if subj_path.exists():
        subj_full = pd.read_parquet(subj_path)
        print(f"  Subjects: {len(subj_full)} rows, columns: {list(subj_full.columns)}")
        print(f"  Process paths (top 30):")
        for p, c in subj_full["process_path"].value_counts().head(30).items():
            print(f"    {c:>6}x  {p}")
        print(f"  cmd_line non-N/A: {(subj_full['cmd_line'] != 'N/A').sum()} / {len(subj_full)}")

    obj_path = root / "objects.parquet"
    if obj_path.exists():
        obj_full = pd.read_parquet(obj_path)
        print(f"\n  Objects: {len(obj_full)} rows, columns: {list(obj_full.columns)}")
        print(f"  Object types: {dict(obj_full['object_type'].value_counts())}")
        print(f"  Files with paths: {obj_full['filename'].notna().sum()} / {len(obj_full)}")
        print(f"  Netflows with ports: {obj_full['local_port'].notna().sum()} / {len(obj_full)}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()