"""Build process name vocabulary from subjects.parquet.

Maps subject UUIDs to process name indices for cross-campaign generalization.
The vocabulary is shared across all splits and campaigns.

Usage:
    python -m src.data.process_vocab --data_root data/processed/darpa_tc_e3/theia
"""

import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


def build_process_vocab(data_root: str | Path, top_k: int = 50) -> dict:
    """Build process name vocabulary from subjects.parquet.
    
    Returns dict with:
        uuid_to_idx: {subject_uuid: process_name_index}
        idx_to_name: list of process names (index 0 = unknown, 1..top_k = top names, top_k+1 = other)
        num_classes: total number of process name classes
    """
    data_root = Path(data_root)
    subj_path = data_root / "subjects.parquet"
    
    if not subj_path.exists():
        raise FileNotFoundError(f"subjects.parquet not found at {subj_path}")
    
    subj = pd.read_parquet(subj_path, columns=["uuid", "process_path"])
    
    # Count process paths
    path_counts = Counter(subj["process_path"].dropna().values)
    top_paths = [p for p, _ in path_counts.most_common(top_k)]
    
    # Build index: 0 = unknown/padding, 1..top_k = top paths, top_k+1 = other
    path_to_idx = {p: i + 1 for i, p in enumerate(top_paths)}
    other_idx = len(top_paths) + 1
    unknown_idx = 0
    
    idx_to_name = ["<unknown>"] + top_paths + ["<other>"]
    num_classes = len(idx_to_name)
    
    # Map each subject UUID to its process name index
    uuid_to_idx = {}
    for uuid, path in zip(subj["uuid"], subj["process_path"]):
        if pd.isna(path):
            uuid_to_idx[str(uuid)] = unknown_idx
        else:
            uuid_to_idx[str(uuid)] = path_to_idx.get(path, other_idx)
    
    print(f"Process vocabulary: {num_classes} classes ({top_k} top paths + unknown + other)")
    print(f"Mapped {len(uuid_to_idx)} subject UUIDs")
    print(f"\nTop 10 process paths:")
    for i, (p, c) in enumerate(path_counts.most_common(10)):
        print(f"  idx={i+1:>3}  {c:>8,}x  {p}")
    
    return {
        "uuid_to_idx": uuid_to_idx,
        "idx_to_name": idx_to_name,
        "num_classes": num_classes,
    }


def main():
    parser = argparse.ArgumentParser(description="Build process name vocabulary")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to dataset root (e.g., data/processed/darpa_tc_e3/theia)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Number of top process paths to keep (default: 50)")
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    vocab = build_process_vocab(data_root, top_k=args.top_k)
    
    # Save vocabulary
    out_path = data_root / "graph" / "process_vocab.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        uuid_to_idx_keys=np.array(list(vocab["uuid_to_idx"].keys())),
        uuid_to_idx_vals=np.array(list(vocab["uuid_to_idx"].values()), dtype=np.int64),
        idx_to_name=np.array(vocab["idx_to_name"]),
        num_classes=np.array(vocab["num_classes"]),
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
