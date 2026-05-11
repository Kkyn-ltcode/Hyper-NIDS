"""
THyN Training Script — Single & Multi-GPU.

Supports DistributedDataParallel for multi-GPU training.

Single GPU:
    python -m src.pipeline.train --config configs/thyn_v0.yaml

Multi-GPU (e.g., 4 GPUs):
    torchrun --nproc_per_node=4 -m src.pipeline.train --config configs/thyn_v0.yaml

Quick smoke test:
    python -m src.pipeline.train --config configs/thyn_v0.yaml --quick
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
)
import yaml

from src.data.thyn_dataset import THyNDataset
from src.model.thyn import THyN


DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3"
)


# ── DDP helpers ──────────────────────────────────────────────

def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_main():
    return get_rank() == 0


def setup_distributed():
    """Initialize DDP if launched via torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return None


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def log(msg):
    """Print only on rank 0."""
    if is_main():
        print(msg)


# ── Training logic ───────────────────────────────────────────

def masked_bce_loss(logits, y, mask, pos_weight):
    """Compute BCE loss only on non-padded positions."""
    real = mask.bool()
    logits_real = logits[real]
    y_real = y[real].float()
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits_real, y_real,
        pos_weight=torch.tensor([pos_weight], device=logits.device),
    )
    return loss


def compute_metrics(all_logits, all_labels):
    """Compute AUPRC, AUROC, F1."""
    probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    labels = np.array(all_labels)

    metrics = {}
    try:
        metrics["auprc"] = average_precision_score(labels, probs)
    except ValueError:
        metrics["auprc"] = 0.0
    try:
        metrics["auroc"] = roc_auc_score(labels, probs)
    except ValueError:
        metrics["auroc"] = 0.0

    preds = (probs > 0.5).astype(int)
    metrics["f1"] = f1_score(labels, preds, zero_division=0)
    metrics["n_attack"] = int(labels.sum())
    metrics["n_benign"] = int((labels == 0).sum())
    metrics["pred_attack"] = int(preds.sum())

    return metrics


def train_epoch(model, loader, optimizer, pos_weight, grad_clip, device,
                log_every=100):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for i, batch in enumerate(loader):
        X = batch["X"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        ent = batch["entity_ids"].to(device, non_blocking=True)

        logits = model(X, entity_ids=ent, mask=mask)
        loss = masked_bce_loss(logits, y, mask, pos_weight)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if log_every and (i + 1) % log_every == 0:
            avg = total_loss / n_batches
            log(f"    batch {i+1}: loss={avg:.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, pos_weight, device, max_batches=None):
    """Evaluate model, return loss and metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits = []
    all_labels = []

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break

        X = batch["X"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        ent = batch["entity_ids"].to(device, non_blocking=True)

        # Use base model (unwrap DDP) to avoid sync overhead in eval
        m = model.module if isinstance(model, DDP) else model
        logits = m(X, entity_ids=ent, mask=mask)
        loss = masked_bce_loss(logits, y, mask, pos_weight)

        total_loss += loss.item()
        n_batches += 1

        real = mask.bool()
        all_logits.extend(logits[real].cpu().tolist())
        all_labels.extend(y[real].cpu().tolist())

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = avg_loss

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train THyN")
    parser.add_argument("--config", default="configs/thyn_v0.yaml")
    parser.add_argument("--dataset", default="theia")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 epoch, small subset")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # DDP setup
    ddp_device = setup_distributed()

    if args.device:
        device = torch.device(args.device)
    elif ddp_device is not None:
        device = ddp_device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    world_size = get_world_size()
    data_root = DATA_ROOT / args.dataset
    save_dir = Path("checkpoints") / "thyn_v0"
    if is_main():
        save_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("TRAIN THyN v0")
    log("=" * 60)
    log(f"  Device:     {device}")
    log(f"  World size: {world_size}")
    log(f"  Config:     {args.config}")
    log(f"  Quick:      {args.quick}")

    # ============================================================
    # Data
    # ============================================================
    log(f"\n[1/4] Loading data...")

    max_seq_len = cfg["data"]["max_seq_len"]
    batch_size = cfg["training"]["batch_size"]

    train_ds = THyNDataset(
        cfg["data"]["train_shards"], data_root,
        max_seq_len=max_seq_len,
        stride=cfg["data"].get("stride", max_seq_len),
    )
    val_ds = THyNDataset(
        cfg["data"]["val_shards"], data_root,
        max_seq_len=max_seq_len,
    )

    # Samplers: DistributedSampler for DDP, None for single GPU
    train_sampler = (DistributedSampler(train_ds, shuffle=True)
                     if is_distributed() else None)
    val_sampler = (DistributedSampler(val_ds, shuffle=False)
                   if is_distributed() else None)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
    )

    n_features = train_ds.n_features
    log(f"  Train: {len(train_ds):,} windows, "
        f"{len(train_ds.X):,} events")
    log(f"  Val:   {len(val_ds):,} windows, "
        f"{len(val_ds.X):,} events")
    log(f"  Features: {n_features}")

    # ============================================================
    # Model
    # ============================================================
    log(f"\n[2/4] Building model...")

    model = THyN(
        n_features=n_features,
        d_model=cfg["model"]["d_model"],
        d_hidden=cfg["model"]["d_hidden"],
        n_layers=cfg["model"]["n_layers"],
        dropout=cfg["model"]["dropout"],
        use_conv=cfg["model"]["use_conv"],
        encoder_type=cfg["model"]["encoder_type"],
    ).to(device)

    # Wrap in DDP if distributed
    if is_distributed():
        model = DDP(model, device_ids=[device.index],
                    find_unused_parameters=False)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Architecture: {cfg['model']['encoder_type'].upper()}")
    log(f"  Parameters: {n_params:,}")
    log(f"  HG Conv: {cfg['model']['use_conv']}")
    if is_distributed():
        log(f"  DDP: {world_size} GPUs")

    # Scale LR linearly with world size
    base_lr = cfg["training"]["lr"]
    effective_lr = base_lr * world_size
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=is_main())

    pos_weight = cfg["training"]["pos_weight"]
    grad_clip = cfg["training"]["grad_clip"]
    epochs = 1 if args.quick else cfg["training"]["epochs"]
    patience = cfg["training"]["patience"]
    log_every = cfg["training"]["log_every"]

    log(f"\n  Effective LR: {effective_lr} "
        f"(base={base_lr} × {world_size} GPUs)")

    # ============================================================
    # Training
    # ============================================================
    log(f"\n[3/4] Training...")
    log(f"  Epochs: {epochs}, Batch: {batch_size}×{world_size}="
        f"{batch_size*world_size}, PosW: {pos_weight}")

    best_auprc = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        log(f"\n  ── Epoch {epoch}/{epochs} ──")

        # Set epoch for DistributedSampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, pos_weight, grad_clip,
            device, log_every=log_every)
        train_time = time.time() - t0

        # Validate (only rank 0 computes metrics for simplicity)
        t1 = time.time()
        val_max_batches = 50 if args.quick else None
        if is_main():
            val_metrics = evaluate(
                model, val_loader, pos_weight, device,
                max_batches=val_max_batches)
        val_time = time.time() - t1

        if is_main():
            scheduler.step(val_metrics["auprc"])

            log(f"  Train: loss={train_loss:.4f} ({train_time:.0f}s)")
            log(f"  Val:   loss={val_metrics['loss']:.4f}, "
                f"AUPRC={val_metrics['auprc']:.4f}, "
                f"AUROC={val_metrics['auroc']:.4f}, "
                f"F1={val_metrics['f1']:.4f} ({val_time:.0f}s)")
            log(f"         attack={val_metrics['n_attack']:,}, "
                f"benign={val_metrics['n_benign']:,}, "
                f"pred_atk={val_metrics['pred_attack']:,}")

            # Early stopping
            if val_metrics["auprc"] > best_auprc:
                best_auprc = val_metrics["auprc"]
                best_epoch = epoch
                no_improve = 0
                raw_model = model.module if isinstance(model, DDP) else model
                torch.save({
                    "epoch": epoch,
                    "model_state": raw_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": cfg,
                }, save_dir / "best.pt")
                log(f"  ✓ New best! Saved to {save_dir}/best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    log(f"  Early stopping at epoch {epoch} "
                        f"(best={best_epoch}, AUPRC={best_auprc:.4f})")
                    break

        # Broadcast early stopping decision to all ranks
        if is_distributed():
            stop_tensor = torch.tensor([no_improve >= patience],
                                       dtype=torch.bool, device=device)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item():
                break

    # ============================================================
    # Final evaluation
    # ============================================================
    if is_main():
        log(f"\n[4/4] Final evaluation on best model (epoch {best_epoch})...")

        raw_model = model.module if isinstance(model, DDP) else model
        ckpt = torch.load(save_dir / "best.pt", map_location=device)
        raw_model.load_state_dict(ckpt["model_state"])

        val_metrics = evaluate(model, val_loader, pos_weight, device)

        log(f"\n{'='*60}")
        log("RESULTS (best model)")
        log(f"{'='*60}")
        log(f"  Epoch:  {best_epoch}")
        log(f"  AUPRC:  {val_metrics['auprc']:.4f}")
        log(f"  AUROC:  {val_metrics['auroc']:.4f}")
        log(f"  F1:     {val_metrics['f1']:.4f}")
        log(f"  Loss:   {val_metrics['loss']:.4f}")

        results = {
            "best_epoch": best_epoch,
            "val_metrics": val_metrics,
            "config": cfg,
            "n_params": n_params,
        }
        torch.save(results, save_dir / "results.pt")
        log(f"\n  Results saved to {save_dir}/results.pt")

    cleanup_distributed()


if __name__ == "__main__":
    main()
