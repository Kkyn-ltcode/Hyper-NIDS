"""
Populate narrow and IoC labels in existing labeled shards.

Reads each labeled shard, computes narrow + IoC labels using
ground truth, and overwrites the Parquet with updated columns.

Usage:
    python -m src.pipeline.relabel --dataset theia
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ground_truth import (
    load_ground_truth,
    label_narrow_events,
    label_ioc_events,
    label_crossprocess_events,
)


DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3"
)


def main():
    parser = argparse.ArgumentParser(
        description="Add narrow/IoC labels to labeled shards")
    parser.add_argument("--dataset", default="theia",
                        choices=["theia", "trace", "trace-1"])
    args = parser.parse_args()

    labeled_dir = DATA_ROOT / args.dataset / "labeled"
    shard_files = sorted(labeled_dir.glob("labeled_shard*.parquet"))

    gt = load_ground_truth(args.dataset)

    print("=" * 60)
    print(f"RELABEL: {args.dataset.upper()}")
    print("=" * 60)
    print(f"  Shards: {len(shard_files)}")
    print(f"  IoC IPs: {len(gt.attack_ips)}")
    print(f"  Malicious files: {gt.malicious_file_substrings}")

    total_narrow = 0
    total_ioc = 0
    total_xproc = 0
    total_events = 0

    # Only load the columns needed for ground truth evaluation to save memory
    import pyarrow.parquet as pq
    sub_schema = pq.read_schema(DATA_ROOT / args.dataset / "subjects.parquet").names
    obj_schema = pq.read_schema(DATA_ROOT / args.dataset / "objects.parquet").names

    sub_cols = [c for c in ["uuid", "process_path", "cmd_line", "parent_uuid"] if c in sub_schema]
    obj_cols = [c for c in ["uuid", "filename", "remote_address", "local_address"] if c in obj_schema]

    subjects_df = pd.read_parquet(DATA_ROOT / args.dataset / "subjects.parquet", columns=sub_cols)
    objects_df  = pd.read_parquet(DATA_ROOT / args.dataset / "objects.parquet", columns=obj_cols)
    print(f"  Global subjects: {len(subjects_df):,} (loaded {len(sub_cols)} cols)")
    print(f"  Global objects:  {len(objects_df):,} (loaded {len(obj_cols)} cols)")

    # Pre-compute attack UUID sets ONCE (not 3x per shard)
    from src.data.ground_truth import (
        build_attack_subject_uuids,
        build_attack_object_uuids,
        build_child_only_subject_uuids,
    )
    print(f"\n  Pre-computing attack UUID sets...")
    t_pre = time.time()
    attack_sub_uuids = build_attack_subject_uuids(subjects_df, gt)
    attack_obj_uuids = build_attack_object_uuids(objects_df, gt)
    child_sub_uuids = build_child_only_subject_uuids(subjects_df, gt)
    print(f"    Attack subjects: {len(attack_sub_uuids):,}")
    print(f"    Attack objects:  {len(attack_obj_uuids):,}")
    print(f"    Child subjects:  {len(child_sub_uuids):,}")
    print(f"    Time: {time.time()-t_pre:.1f}s")

    # Free the massive dataframes before we start processing shards
    del subjects_df, objects_df
    gc.collect()

    for fi, f in enumerate(shard_files):
        shard_name = f.stem
        shard_idx = int(shard_name.replace("labeled_shard", ""))
        print(f"\n  ── {shard_name} ({fi+1}/{len(shard_files)}) ──")

        # Check if shard already has all required label columns (resume support)
        import pyarrow.parquet as pq
        existing_cols = set(pq.read_schema(f).names)
        required_cols = {"label_narrow", "label_ioc", "label_crossprocess"}
        if required_cols.issubset(existing_cols):
            # Quick read to report stats without reprocessing
            df_check = pd.read_parquet(
                f, columns=["label_broad", "label_narrow", "label_ioc",
                             "label_crossprocess"])
            n = len(df_check)
            n_narrow = int(df_check["label_narrow"].sum())
            n_ioc = int(df_check["label_ioc"].sum())
            n_xproc = int(df_check["label_crossprocess"].sum())
            total_narrow += n_narrow
            total_ioc += n_ioc
            total_xproc += n_xproc
            total_events += n
            print(f"    SKIPPED (already relabeled, {n:,} events)")
            del df_check
            gc.collect()
            continue

        t0 = time.time()

        # Load labeled events
        events_df = pd.read_parquet(f)
        n = len(events_df)

        # --- Narrow labels (attack subject + IoC object) ---
        is_attack_sub = events_df["subject_uuid"].isin(attack_sub_uuids)
        touches_ioc_obj = events_df["predicate_object_uuid"].isin(attack_obj_uuids)
        if "predicate_object2_uuid" in events_df.columns:
            touches_ioc_obj = touches_ioc_obj | \
                events_df["predicate_object2_uuid"].isin(attack_obj_uuids)
        narrow = (is_attack_sub & touches_ioc_obj).astype(np.int8)
        events_df["label_narrow"] = narrow.values

        # --- IoC labels (any event touching IoC object) ---
        ioc = touches_ioc_obj.astype(np.int8)
        events_df["label_ioc"] = ioc.values

        # --- Cross-process labels (child processes only) ---
        xproc = events_df["subject_uuid"].isin(child_sub_uuids).astype(np.int8)
        events_df["label_crossprocess"] = xproc.values

        # Overwrite parquet (atomic write via tmp file)
        tmp_path = f.with_suffix(".tmp.parquet")
        events_df.to_parquet(tmp_path, index=False)
        tmp_path.replace(f)

        n_narrow = int(narrow.sum())
        n_ioc = int(ioc.sum())
        n_xproc = int(xproc.sum())
        total_narrow += n_narrow
        total_ioc += n_ioc
        total_xproc += n_xproc
        total_events += n

        print(f"    Broad:  {int(events_df['label_broad'].sum()):,} "
              f"({100*events_df['label_broad'].sum()/max(n, 1):.1f}%)")
        print(f"    XProc:  {n_xproc:,} ({100*n_xproc/max(n, 1):.3f}%)")
        print(f"    Narrow: {n_narrow:,} ({100*n_narrow/max(n, 1):.3f}%)")
        print(f"    IoC:    {n_ioc:,} ({100*n_ioc/max(n, 1):.4f}%)")
        print(f"    Time:   {time.time()-t0:.1f}s")

        del events_df, narrow, ioc, xproc
        gc.collect()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total events: {total_events:,}")
    print(f"  XProc:        {total_xproc:,} "
          f"({100*total_xproc/max(total_events, 1):.3f}%)")
    print(f"  Narrow:       {total_narrow:,} "
          f"({100*total_narrow/max(total_events, 1):.3f}%)")
    print(f"  IoC:          {total_ioc:,} "
          f"({100*total_ioc/max(total_events, 1):.4f}%)")


if __name__ == "__main__":
    main()
