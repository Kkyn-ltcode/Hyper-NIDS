"""
Training script for GRU-TGN Baseline.

Reuses the same data pipeline (ChronoDataset), training loop structure,
and evaluation metrics as train_full.py for a fair comparison.

Usage:
    python -m src.baselines.train_gru_tgn --label_type crossprocess
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.data.chrono_dataset import ChronoDataset
from src.baselines.gru_tgn import GRUTGNBaseline

DATA_ROOT = Path("data/processed/darpa_tc_e3")

SHARD_CONFIG = {
    "theia": {
        "train": list(range(11)),       # Shards 0-10 (April 3-5)
        "val":   [11],                  # Shard 11 (April 9)
        "test":  list(range(12, 25))    # Shards 12-24 (April 10-13)
    },
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

    chunk_losses = []
    all_train_logits = []
    all_train_labels = []

    pw_t = torch.tensor([pos_weight], device=device)

    for i, batch in enumerate(loader):
        X_c = torch.nan_to_num(batch["X_cont"].to(device), nan=0.0).clamp(-20, 20)
        et = batch["event_type"].to(device)
        y = batch["y"].to(device).float()
        ent = batch["entity_ids"].to(device)
        ts = batch["timestamp"].to(device)

        optimizer.zero_grad()

        logits = model(X_c, et, ent, ts)
        logits = logits.clamp(-50, 50)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pw_t)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_chunks += 1
            model.detach_bank()
            continue

        all_train_logits.extend(logits.detach().squeeze(0).cpu().tolist())
        all_train_labels.extend(y.squeeze(0).cpu().tolist())

        loss.backward()

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

        model.detach_bank()

        total_loss += loss.item()
        chunk_losses.append(loss.item())
        n_chunks += 1
        events_processed += X_c.size(1)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            throughput = events_processed / elapsed
            avg_loss = total_loss / max(n_chunks, 1)
            logging.info(f"    Chunk {i+1}/{len(loader)}: "
                  f"loss={avg_loss:.4f}, {throughput:.0f} events/s"
                  + (f", {nan_chunks} NaN skipped" if nan_chunks else ""))

    elapsed = time.time() - t0
    avg_loss = total_loss / max(n_chunks, 1)
    throughput = events_processed / elapsed if elapsed > 0 else 0

    if nan_chunks > 0:
        logging.info(f"    ⚠ {nan_chunks}/{i+1} chunks had NaN loss/gradients")
    logging.info(f"    Epoch done: {events_processed:,} events in {elapsed:.1f}s "
          f"({throughput:.0f} events/s)")

    train_metrics = compute_metrics(all_train_logits, all_train_labels)
    return avg_loss, chunk_losses, train_metrics


@torch.no_grad()
def warmup_bank(model, loaders, device):
    """Process data shards in eval mode to warm up entity states."""
    model.eval()
    model.reset_bank()
    
    total_events = 0
    for loader in loaders:
        for batch in loader:
            X_c = torch.nan_to_num(batch["X_cont"].to(device), nan=0.0).clamp(-20, 20)
            et = batch["event_type"].to(device)
            ent = batch["entity_ids"].to(device)
            ts = batch["timestamp"].to(device)
            
            _ = model(X_c, et, ent, ts)
            model.detach_bank()
            total_events += X_c.size(1)
    
    logging.info(f"  Bank warmup complete: {total_events:,} events processed")


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on a dataset."""
    model.eval()

    all_logits = []
    all_labels = []
    total_loss = 0.0
    n_chunks = 0

    for batch in loader:
        X_c = torch.nan_to_num(batch["X_cont"].to(device), nan=0.0).clamp(-20, 20)
        et = batch["event_type"].to(device)
        y = batch["y"].to(device).float()
        ent = batch["entity_ids"].to(device)
        ts = batch["timestamp"].to(device)

        logits = model(X_c, et, ent, ts)
        
        logits_clamp = logits.clamp(-50, 50)
        loss = nn.functional.binary_cross_entropy_with_logits(logits_clamp, y)
            
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            n_chunks += 1

        model.detach_bank()

        all_logits.extend(logits.squeeze(0).cpu().tolist())
        all_labels.extend(y.squeeze(0).cpu().tolist())

    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / max(n_chunks, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train GRU-TGN Baseline")
    parser.add_argument("--dataset", default="theia", choices=["theia", "trace"])
    parser.add_argument("--label_type", type=str, default="crossprocess")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_pos_weight", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # --- Reproducibility ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # --- Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("runs") / f"gru_tgn_{args.dataset}_{args.label_type}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"GRU-TGN Baseline | args={vars(args)}")
    logging.info(f"Device: {device}")
    logging.info(f"Log dir: {log_dir}")

    # --- Data ---
    shards = SHARD_CONFIG[args.dataset]
    data_dir = DATA_ROOT / args.dataset

    logging.info("Loading training shards...")
    train_ds = ChronoDataset(
        shard_ids=shards["train"],
        data_root=data_dir,
        chunk_size=args.chunk_size,
        label_type=args.label_type,
    )

    logging.info("Loading validation shards...")
    val_ds = ChronoDataset(
        shard_ids=shards["val"],
        data_root=data_dir,
        chunk_size=args.chunk_size,
        label_type=args.label_type,
        t0_nanos=train_ds.t0_nanos,
    )

    logging.info("Loading test shards...")
    test_ds = ChronoDataset(
        shard_ids=shards["test"],
        data_root=data_dir,
        chunk_size=args.chunk_size,
        label_type=args.label_type,
        t0_nanos=train_ds.t0_nanos,
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # --- Model ---
    model = GRUTGNBaseline(
        num_entities=train_ds.num_entities,
        n_cont_features=train_ds.n_cont_features,
        num_event_types=train_ds.num_event_types,
        d_model=args.d_model,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"  Model parameters: {n_params:,}")
    logging.info(f"  n_cont_features: {train_ds.n_cont_features}")
    logging.info(f"  num_event_types: {train_ds.num_event_types}")
    logging.info(f"  num_entities: {train_ds.num_entities}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)

    # pos_weight
    n_pos = int((train_ds.y == 1).sum())
    n_neg = int((train_ds.y == 0).sum())
    raw_ratio = n_neg / max(n_pos, 1)
    pos_weight = min(raw_ratio, args.max_pos_weight)
    logging.info(f"  pos_weight: {pos_weight:.1f} (raw ratio {raw_ratio:.1f}, capped at {args.max_pos_weight})")

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    # --- Training ---
    best_auprc = 0.0
    best_epoch = 0
    patience = 5
    no_improve = 0

    logging.info(f"\nStarting GRU-TGN training ({args.epochs} epochs)...")

    for epoch in range(1, args.epochs + 1):
        logging.info(f"\n--- Epoch {epoch}/{args.epochs} ---")

        model.reset_bank()

        train_loss, epoch_chunk_losses, train_metrics = train_epoch(
            model, train_loader, optimizer, device, pos_weight)

        val_metrics = evaluate(model, val_loader, device)

        scheduler.step()

        val_loss = val_metrics["loss"]
        auprc = val_metrics["auprc"]
        f1 = val_metrics["best_f1"]
        t_auprc = train_metrics["auprc"]
        t_f1 = train_metrics["best_f1"]

        logging.info(f"  train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_auprc={t_auprc:.4f}, train_f1={t_f1:.4f}, "
              f"val_auprc={auprc:.4f}, val_f1={f1:.4f}")

        if auprc > best_auprc:
            best_auprc = auprc
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_auprc": auprc,
            }, log_dir / "best_model.pt")
            logging.info(f"  ★ New best val AUPRC: {auprc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info(f"  Early stopping: no improvement for {patience} epochs")
                break

    # --- Test ---
    logging.info(f"\n--- Test Evaluation (best epoch {best_epoch}) ---")

    ckpt = torch.load(log_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # Warmup bank
    warmup_bank(model, [train_loader, val_loader], device)

    test_metrics = evaluate(model, test_loader, device)
    logging.info(f"  test_loss={test_metrics['loss']:.4f}, "
          f"test_auprc={test_metrics['auprc']:.4f}, test_f1={test_metrics['best_f1']:.4f}")

    # Final summary
    logging.info(f"\n{'='*60}")
    logging.info(f"GRU-TGN BASELINE RESULTS ({args.dataset} / {args.label_type})")
    logging.info(f"  Best epoch: {best_epoch}")
    logging.info(f"  Val  AUPRC: {best_auprc:.4f}")
    logging.info(f"  Test AUPRC: {test_metrics['auprc']:.4f}")
    logging.info(f"  Test F1:    {test_metrics['best_f1']:.4f}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
