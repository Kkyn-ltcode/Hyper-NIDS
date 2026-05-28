"""
Training script for HyperMamba Minimal Prototype.

Uses Truncated Backpropagation Through Time (TBPTT):
  - Events processed in strict chronological order (batch_size=1, shuffle=False)
  - Entity states persist across chunks within an epoch
  - States detached at chunk boundaries to bound memory
  - States reset at epoch boundaries

Usage:
    python -m src.pipeline.train_proto --dataset theia --label_type l1
    python -m src.pipeline.train_proto --dataset trace --label_type l1
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.data.chrono_dataset import ChronoDataset
from src.model.hypermamba_proto import HyperMambaProto

DATA_ROOT = Path("data/processed/darpa_tc_e3")

# Dataset-specific shard configuration
SHARD_CONFIG = {
    "theia": {"train": list(range(7)), "val": [7], "test": [8, 9]},
    "trace": {"train": list(range(5)), "val": [5], "test": [6]},
}


def compute_metrics(all_logits, all_labels):
    logits_t = torch.tensor(all_logits).clamp(-50, 50)
    probs = torch.sigmoid(logits_t).numpy()
    labels = np.array(all_labels)

    valid = labels >= 0
    probs, labels = probs[valid], labels[valid]

    m = {}
    try:
        m["auprc"] = float(average_precision_score(labels, probs))
    except ValueError:
        m["auprc"] = 0.0
    try:
        prec, rec, thr = precision_recall_curve(labels, probs)
        f1_all = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        m["best_f1"] = float(f1_all[np.argmax(f1_all)])
    except ValueError:
        m["best_f1"] = 0.0
    return m


def train_epoch(model, loader, optimizer, device, pos_weight, grad_clip=1.0):
    model.train()
    model.reset_bank()

    total_loss = 0.0
    n_chunks = 0
    nan_chunks = 0
    t0 = time.time()
    events_processed = 0

    pw_t = torch.tensor([pos_weight], device=device)

    for i, batch in enumerate(loader):
        X_c = batch["X_cont"].to(device).clamp(-20, 20)
        et = batch["event_type"].to(device)
        y = batch["y"].to(device).float()
        ent = batch["entity_ids"].to(device)

        optimizer.zero_grad()

        logits = model(X_c, et, ent)

        # Clamp logits before loss to prevent overflow
        logits = logits.clamp(-50, 50)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pw_t)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_chunks += 1
            model.detach_bank()
            continue

        loss.backward()

        # Check for NaN gradients
        has_nan = any(
            p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in model.parameters()
        )
        if has_nan:
            nan_chunks += 1
            optimizer.zero_grad()
            model.detach_bank()
            continue

        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Detach bank after each chunk (TBPTT boundary)
        model.detach_bank()

        total_loss += loss.item()
        n_chunks += 1
        events_processed += X_c.size(1)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            throughput = events_processed / elapsed
            avg_loss = total_loss / max(n_chunks, 1)
            print(f"    Chunk {i+1}/{len(loader)}: "
                  f"loss={avg_loss:.4f}, {throughput:.0f} events/s"
                  + (f", {nan_chunks} NaN skipped" if nan_chunks else ""))

    elapsed = time.time() - t0
    avg_loss = total_loss / max(n_chunks, 1)
    throughput = events_processed / elapsed if elapsed > 0 else 0

    if nan_chunks > 0:
        print(f"    ⚠ {nan_chunks}/{i+1} chunks had NaN loss/gradients")
    print(f"    Epoch done: {events_processed:,} events in {elapsed:.1f}s "
          f"({throughput:.0f} events/s)")

    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    model.reset_bank()

    all_logits = []
    all_labels = []

    for batch in loader:
        X_c = batch["X_cont"].to(device).clamp(-20, 20)
        et = batch["event_type"].to(device)
        y = batch["y"].to(device)
        ent = batch["entity_ids"].to(device)

        logits = model(X_c, et, ent)

        # Detach bank during eval too
        model.detach_bank()

        all_logits.extend(logits.squeeze(0).cpu().tolist())
        all_labels.extend(y.squeeze(0).cpu().tolist())

    return compute_metrics(all_logits, all_labels)


def main():
    parser = argparse.ArgumentParser(
        description="Train HyperMamba Prototype (Cross-Entity State Propagation)")
    parser.add_argument("--dataset", default="theia", choices=["theia", "trace"])
    parser.add_argument("--label_type", default="l1", choices=["broad", "l1"])
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_pos_weight", type=float, default=30.0,
                        help="Cap pos_weight to prevent gradient explosion")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_root = DATA_ROOT / args.dataset
    shards = SHARD_CONFIG[args.dataset]

    print("=" * 60)
    print(f"  HYPERMAMBA PROTOTYPE — {args.dataset.upper()}")
    print("=" * 60)
    print(f"  Device:      {device}")
    print(f"  Label type:  {args.label_type}")
    print(f"  Chunk size:  {args.chunk_size}")
    print(f"  d_model:     {args.d_model}")
    print(f"  Train shards: {shards['train']}")
    print(f"  Val shards:   {shards['val']}")

    # --- Data ---
    print(f"\nLoading training data...")
    train_ds = ChronoDataset(
        shards["train"], data_root,
        chunk_size=args.chunk_size, label_type=args.label_type)

    # Always evaluate on broad labels (even when training on L1*)
    eval_label = "broad" if args.label_type == "l1" else args.label_type
    print(f"\nLoading validation data (labels={eval_label})...")
    val_ds = ChronoDataset(
        shards["val"], data_root,
        chunk_size=args.chunk_size, label_type=eval_label)

    # Strict chronological: batch_size=1, shuffle=False, num_workers=0
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # --- Model ---
    model = HyperMambaProto(
        num_entities=train_ds.num_entities,
        n_cont_features=train_ds.n_cont_features,
        num_event_types=train_ds.num_event_types,
        d_model=args.d_model,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Compute and cap pos_weight
    n_pos = int((train_ds.y == 1).sum())
    n_neg = int((train_ds.y == 0).sum())
    raw_pw = n_neg / max(n_pos, 1)
    pos_weight = min(raw_pw, args.max_pos_weight)
    print(f"  pos_weight: {pos_weight:.1f} (raw={raw_pw:.1f}, cap={args.max_pos_weight})")

    # --- Training ---
    save_dir = Path("checkpoints") / f"proto_{args.dataset}_{args.label_type}"
    save_dir.mkdir(parents=True, exist_ok=True)

    best_auprc = 0.0
    best_epoch = 0
    patience = 5
    no_improve = 0

    print(f"\nStarting training ({args.epochs} epochs)...")

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        train_loss = train_epoch(
            model, train_loader, optimizer, device, pos_weight)

        val_metrics = evaluate(model, val_loader, device)
        auprc = val_metrics["auprc"]
        f1 = val_metrics["best_f1"]

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val AUPRC:  {auprc:.4f}  |  Val F1: {f1:.4f}")

        if auprc > best_auprc:
            best_auprc = auprc
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_auprc": auprc,
                "val_f1": f1,
            }, save_dir / "best.pt")
            print(f"  ✓ New best! AUPRC={auprc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch} (best={best_epoch})")
                break

    print(f"\n{'='*60}")
    print(f"  DONE — Best AUPRC: {best_auprc:.4f} (epoch {best_epoch})")
    print(f"  Checkpoint: {save_dir / 'best.pt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
