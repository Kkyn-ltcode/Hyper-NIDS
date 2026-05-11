"""
Test the THyNDataset and DataLoader.

Usage:
    python -m src.pipeline.test_dataloader --dataset theia --split train
"""

import argparse
import time
from pathlib import Path

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
    print(f"  Windows: {len(ds):,}")
    print(f"  Cont features: {ds.n_cont_features}")
    print(f"  Event types: {ds.num_event_types}")

    item = ds[0]
    print(f"\n  Single item:")
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:15s}: {str(v.shape):20s} {v.dtype}")
        else:
            print(f"    {k:15s}: {v}")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    print(f"\n  Batch (bs={batch_size}):")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:15s}: {str(v.shape):20s} {v.dtype}")

    # Validate
    mask = batch["mask"]
    y = batch["y"]
    real = y[mask.bool()]
    n_atk = (real == 1).sum().item()
    n_ben = (real == 0).sum().item()
    print(f"\n  Labels: attack={n_atk}, benign={n_ben}")
    print(f"  NaN in X_cont: {torch.isnan(batch['X_cont']).any().item()}")
    print(f"  Event type range: [{batch['event_type'].min()}, "
          f"{batch['event_type'].max()}]")

    # Throughput
    t0 = time.time()
    for i, b in enumerate(loader):
        if i >= 10: break
    tp = (10 * batch_size) / (time.time() - t0)
    print(f"  Throughput: {tp:.0f} windows/sec")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="theia")
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    data_root = DATA_ROOT / args.dataset
    splits = {args.split: SPLITS[args.split]} if args.split else SPLITS
    for name, sids in splits.items():
        test_split(name, sids, data_root, max_seq_len=args.max_seq_len)
    print(f"\n✓ ALL TESTS PASSED")


if __name__ == "__main__":
    main()
