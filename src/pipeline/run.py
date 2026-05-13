#!/usr/bin/env python3
"""
Unified pipeline runner for HyperMamba-NIDS.

Runs all pipeline stages in the correct order:
  1. download   — Download and extract DARPA TC data
  2. ingest     — Parse JSON → Parquet shards + labeling
  3. relabel    — Recompute narrow/IoC/crossprocess labels
  4. features   — Extract per-event feature vectors
  5. normalize  — Z-score normalization with train stats
  6. graph      — Build incidence matrix + entity vocab
  7. sequences  — Build per-subject temporal sequences

Usage:
    # Full pipeline
    python -m src.pipeline.run --dataset theia

    # From a specific stage
    python -m src.pipeline.run --dataset theia --start-from features

    # Single stage
    python -m src.pipeline.run --dataset theia --only normalize --train-shards 0-6
"""

import argparse
import subprocess
import sys
import time

STAGES = [
    "download",
    "parse",
    "ingest",
    "relabel",
    "features",
    "normalize",
    "graph",
    "sequences",
]

STAGE_COMMANDS = {
    "download": lambda args: [
        sys.executable, "-m", "src.data.download_darpa_tc",
        "--extract", "--verify",
        "--dataset", args.dataset,
    ],
    "parse": lambda args: [
        sys.executable, "-m", "src.data.darpa_tc_parser",
        "--shards", "all",
        "--dataset", args.dataset,
    ],
    "ingest": lambda args: [
        sys.executable, "-m", "src.pipeline.batch_ingest",
        "--dataset", args.dataset,
    ],
    "relabel": lambda args: [
        sys.executable, "-m", "src.pipeline.relabel",
        "--dataset", args.dataset,
    ],
    "features": lambda args: [
        sys.executable, "-m", "src.pipeline.batch_features",
        "--dataset", args.dataset,
    ],
    "normalize": lambda args: [
        sys.executable, "-m", "src.pipeline.normalize",
        "--dataset", args.dataset,
        "--train-shards", args.train_shards or (
            "0-6" if args.dataset == "theia" else "0-4"),
    ],
    "graph": lambda args: [
        sys.executable, "-m", "src.pipeline.build_graph",
        "--dataset", args.dataset,
    ],
    "sequences": lambda args: [
        sys.executable, "-m", "src.pipeline.build_sequences",
        "--dataset", args.dataset,
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description="Run HyperMamba-NIDS pipeline stages")
    parser.add_argument("--dataset", required=True,
                        choices=["theia", "trace"],
                        help="Dataset to process")
    parser.add_argument("--start-from", default=None,
                        choices=STAGES,
                        help="Start from this stage (default: first)")
    parser.add_argument("--only", default=None,
                        choices=STAGES,
                        help="Run only this single stage")
    parser.add_argument("--train-shards", default=None,
                        help="Train shard range for normalize. "
                             "Default: 0-6 for theia, 0-4 for trace")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    # Determine which stages to run
    if args.only:
        stages = [args.only]
    elif args.start_from:
        start_idx = STAGES.index(args.start_from)
        stages = STAGES[start_idx:]
    else:
        stages = STAGES

    print("=" * 60)
    print(f"PIPELINE: {args.dataset.upper()}")
    print(f"Stages: {' → '.join(stages)}")
    print("=" * 60)

    t_total = time.time()
    for i, stage in enumerate(stages, 1):
        cmd = STAGE_COMMANDS[stage](args)
        print(f"\n{'='*60}")
        print(f"[{i}/{len(stages)}] {stage.upper()}")
        print(f"  cmd: {' '.join(cmd)}")
        print(f"{'='*60}")

        if args.dry_run:
            continue

        t0 = time.time()
        result = subprocess.run(cmd, cwd=".")
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\n  ✗ Stage '{stage}' FAILED (exit {result.returncode})")
            print(f"    after {elapsed:.1f}s")
            sys.exit(1)

        print(f"\n  ✓ Stage '{stage}' completed in {elapsed:.1f}s")

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {total_time:.1f}s total")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
