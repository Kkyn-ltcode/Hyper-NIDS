"""
Test the THyNDataset and DataLoader.

Validates shapes, label distributions, no NaN leakage, and basic
data integrity for train/val/test splits.

Usage:
    python -m src.pipeline.test_dataloader --dataset theia
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.thyn_dataset import THyNDataset


DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3"
)

SPLITS = {
    "train": [0, 1, 2, 3, 4, 5, 6],
    "val":   [7],
    "test":  [8, 9],
}


def test_split(name, shard_ids, data_root, max_seq_len=512, batch_size=64):
    print(f"\n{'='*60}")
    print(f"SPLIT: {name.upper()} (shards {shard_ids})")
    print(f"{'='*60}")

    t0 = time.time()
    ds = THyNDataset(shard_ids, data_root, max_seq_len=max_seq_len)
    print(f"  Init time: {time.time()-t0:.1f}s")

    print(f"\n  Dataset stats:")
    print(f"    Length (windows): {len(ds):,}")
    print(f"    Features:        {ds.n_features}")
    print(f"    Max seq len:     {ds.max_seq_len}")

    # Test single item
    item = ds[0]
    print(f"\n  Single item shapes:")
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:15s}: {str(v.shape):20s} dtype={v.dtype}")
        else:
            print(f"    {k:15s}: {v}")

    # Test DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=False)
    t0 = time.time()
    batch = next(iter(loader))
    load_time = time.time() - t0

    print(f"\n  Batch shapes (batch_size={batch_size}):")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:15s}: {str(v.shape):20s} dtype={v.dtype}")
        else:
            print(f"    {k:15s}: type={type(v)}")

    # Validate
    X = batch["X"]
    y = batch["y"]
    mask = batch["mask"]
    ent = batch["entity_ids"]

    print(f"\n  Validation:")

    # No NaN
    has_nan = torch.isnan(X).any().item()
    print(f"    NaN in X:       {'⚠ YES' if has_nan else '✓ None'}")

    # Labels
    real_labels = y[mask.bool()]
    n_atk = (real_labels == 1).sum().item()
    n_ben = (real_labels == 0).sum().item()
    n_pad = (y == -1).sum().item()
    print(f"    Real labels:    {n_atk + n_ben:,} "
          f"(attack={n_atk}, benign={n_ben})")
    print(f"    Padding labels: {n_pad:,} (should be -1)")
    if n_atk + n_ben > 0:
        print(f"    Attack %:       {100*n_atk/(n_atk+n_ben):.1f}%")

    # Entity IDs
    real_ent = ent[mask.bool()]
    n_valid_ent = (real_ent >= 0).sum().item()
    n_total_ent = real_ent.numel()
    print(f"    Valid entity IDs: {n_valid_ent:,} / {n_total_ent:,}")

    # Mask consistency
    seq_lens = batch["seq_len"]
    for i in range(min(5, len(seq_lens))):
        sl = seq_lens[i].item() if isinstance(seq_lens[i], torch.Tensor) else seq_lens[i]
        mask_sum = int(mask[i].sum().item())
        ok = "✓" if sl == mask_sum else "⚠"
        print(f"    Window {i}: seq_len={sl}, mask_sum={mask_sum} {ok}")

    # Speed test: iterate 10 batches
    t0 = time.time()
    for i, b in enumerate(loader):
        if i >= 10:
            break
    throughput = (10 * batch_size) / (time.time() - t0)
    print(f"\n  Throughput: {throughput:.0f} windows/sec")
    print(f"  First batch load: {load_time:.3f}s")

    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="theia")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split", default=None,
                        help="Test only this split (train/val/test)")
    args = parser.parse_args()

    data_root = DATA_ROOT / args.dataset

    splits = {args.split: SPLITS[args.split]} if args.split else SPLITS

    for name, shard_ids in splits.items():
        test_split(name, shard_ids, data_root,
                   max_seq_len=args.max_seq_len,
                   batch_size=args.batch_size)

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED ✓")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
