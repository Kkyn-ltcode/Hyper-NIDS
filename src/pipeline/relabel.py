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
                        choices=["theia", "trace"])
    args = parser.parse_args()

    labeled_dir = DATA_ROOT / args.dataset / "labeled"
    shard_dir = DATA_ROOT / args.dataset / "shards"
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

    for f in shard_files:
        shard_name = f.stem
        shard_idx = int(shard_name.replace("labeled_shard", ""))
        print(f"\n  ── {shard_name} ──")
        t0 = time.time()

        # Load labeled events
        events_df = pd.read_parquet(f)
        n = len(events_df)

        # Load subjects/objects from raw shards (needed for label functions)
        subjects_df = pd.read_parquet(
            shard_dir / f"subjects_shard{shard_idx}.parquet")
        objects_df = pd.read_parquet(
            shard_dir / f"objects_shard{shard_idx}.parquet")

        # Compute narrow labels
        narrow = label_narrow_events(events_df, subjects_df, objects_df, gt)
        events_df["label_narrow"] = narrow.values

        # Compute IoC labels
        ioc = label_ioc_events(events_df, subjects_df, objects_df, gt)
        events_df["label_ioc"] = ioc.values

        # Compute cross-process labels
        xproc = label_crossprocess_events(events_df, subjects_df, objects_df, gt)
        events_df["label_crossprocess"] = xproc.values

        # Overwrite parquet
        events_df.to_parquet(f, index=False)

        n_narrow = int(narrow.sum())
        n_ioc = int(ioc.sum())
        n_xproc = int(xproc.sum())
        total_narrow += n_narrow
        total_ioc += n_ioc
        total_xproc += n_xproc
        total_events += n

        print(f"    Broad:  {int(events_df['label_broad'].sum()):,} "
              f"({100*events_df['label_broad'].sum()/n:.1f}%)")
        print(f"    XProc:  {n_xproc:,} ({100*n_xproc/n:.3f}%)")
        print(f"    Narrow: {n_narrow:,} ({100*n_narrow/n:.3f}%)")
        print(f"    IoC:    {n_ioc:,} ({100*n_ioc/n:.4f}%)")
        print(f"    Time:   {time.time()-t0:.1f}s")

        del events_df, subjects_df, objects_df, narrow, ioc, xproc
        gc.collect()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total events: {total_events:,}")
    print(f"  XProc:        {total_xproc:,} "
          f"({100*total_xproc/total_events:.3f}%)")
    print(f"  Narrow:       {total_narrow:,} "
          f"({100*total_narrow/total_events:.3f}%)")
    print(f"  IoC:          {total_ioc:,} "
          f"({100*total_ioc/total_events:.4f}%)")


if __name__ == "__main__":
    main()
