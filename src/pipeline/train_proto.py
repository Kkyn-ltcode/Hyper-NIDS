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
import logging
import shutil
import time
from datetime import datetime
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

    chunk_losses = []

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

    return avg_loss, chunk_losses


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
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_pos_weight", type=float, default=30.0,
                        help="Cap pos_weight to prevent gradient explosion")
    parser.add_argument("--no_state", action="store_true",
                        help="Ablation: disable cross-entity state propagation")
    parser.add_argument("--bank_decay", type=float, default=0.95,
                        help="Decay factor applied to bank after each chunk")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Create a timestamped run directory (will be renamed with results at end)
    state_tag = "state" if not args.no_state else "nostate"
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_base = Path("checkpoints") / "proto_runs"
    save_dir = run_base / f"{args.dataset}_{args.label_type}_{state_tag}_{run_ts}"
    save_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(save_dir / "train.log"),
            logging.StreamHandler()
        ]
    )

    data_root = DATA_ROOT / args.dataset
    shards = SHARD_CONFIG[args.dataset]

    use_state = not args.no_state
    state_label = "WITH state propagation" if use_state else "WITHOUT state (ablation)"

    logging.info("=" * 60)
    logging.info(f"  HYPERMAMBA PROTOTYPE — {args.dataset.upper()}")
    logging.info("=" * 60)
    logging.info(f"  Device:      {device}")
    logging.info(f"  Label type:  {args.label_type}")
    logging.info(f"  Chunk size:  {args.chunk_size}")
    logging.info(f"  d_model:     {args.d_model}")
    logging.info(f"  State:       {state_label}")
    logging.info(f"  Bank decay:  {args.bank_decay}")
    logging.info(f"  Train shards: {shards['train']}")
    logging.info(f"  Val shards:   {shards['val']}")

    # --- Data ---
    logging.info(f"\nLoading training data...")
    train_ds = ChronoDataset(
        shards["train"], data_root,
        chunk_size=args.chunk_size, label_type=args.label_type)

    # Always evaluate on broad labels (even when training on L1*)
    eval_label = "broad" if args.label_type == "l1" else args.label_type
    logging.info(f"\nLoading validation data (labels={eval_label})...")
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
        use_state=use_state,
        bank_decay=args.bank_decay,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Compute and cap pos_weight
    n_pos = int((train_ds.y == 1).sum())
    n_neg = int((train_ds.y == 0).sum())
    raw_pw = n_neg / max(n_pos, 1)
    pos_weight = min(raw_pw, args.max_pos_weight)
    logging.info(f"  pos_weight: {pos_weight:.1f} (raw={raw_pw:.1f}, cap={args.max_pos_weight})")

    # --- Training ---

    best_auprc = 0.0
    best_epoch = 0
    patience = 5
    no_improve = 0

    history = {
        'train_loss': [],
        'val_auprc': [],
        'val_f1': [],
        'chunk_loss': []
    }

    logging.info(f"\nStarting training ({args.epochs} epochs)...")

    for epoch in range(1, args.epochs + 1):
        logging.info(f"\n--- Epoch {epoch}/{args.epochs} ---")

        train_loss, epoch_chunk_losses = train_epoch(
            model, train_loader, optimizer, device, pos_weight)

        val_metrics = evaluate(model, val_loader, device)
        auprc = val_metrics["auprc"]
        f1 = val_metrics["best_f1"]

        history['train_loss'].append(train_loss)
        history['chunk_loss'].extend(epoch_chunk_losses)
        history['val_auprc'].append(auprc)
        history['val_f1'].append(f1)

        logging.info(f"  Train Loss: {train_loss:.4f}")
        logging.info(f"  Val AUPRC:  {auprc:.4f}  |  Val F1: {f1:.4f}")

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
            logging.info(f"  ✓ New best! AUPRC={auprc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info(f"  Early stop at epoch {epoch} (best={best_epoch})")
                break

    logging.info(f"\n{'='*60}")
    logging.info(f"  DONE — Best AUPRC: {best_auprc:.4f} (epoch {best_epoch})")
    logging.info(f"  Checkpoint: {save_dir / 'best.pt'}")
    logging.info(f"{'='*60}")

    # Save training history
    torch.save(history, save_dir / "history.pt")

    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 10))
        
        # Plot chunk losses
        plt.subplot(3, 1, 1)
        plt.plot(history['chunk_loss'], alpha=0.6, label='Chunk Loss')
        plt.xlabel('Chunk')
        plt.ylabel('Loss')
        plt.title('Training Loss per Chunk')
        plt.legend()
        plt.grid(True)
        
        # Plot epoch losses
        plt.subplot(3, 1, 2)
        epochs_range = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs_range, history['train_loss'], marker='o', color='red', label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss per Epoch')
        if epochs_range:
            plt.xticks(epochs_range)
        plt.legend()
        plt.grid(True)
        
        # Plot validation metrics
        plt.subplot(3, 1, 3)
        val_epochs_range = range(1, len(history['val_auprc']) + 1)
        plt.plot(val_epochs_range, history['val_auprc'], marker='s', color='green', label='Val AUPRC')
        plt.plot(val_epochs_range, history['val_f1'], marker='^', color='purple', label='Val F1')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        if val_epochs_range:
            plt.xticks(val_epochs_range)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = save_dir / "loss_plot.png"
        plt.savefig(plot_path)
        logging.info(f"Loss plot saved to {plot_path}")
    except ImportError:
        logging.warning("matplotlib is not installed. Skipping loss plot generation.")

    # Rename folder with results
    best_f1 = max(history['val_f1']) if history['val_f1'] else 0.0
    actual_epochs = len(history['train_loss'])
    final_name = (
        f"{args.dataset}_{args.label_type}_{state_tag}"
        f"_auprc{best_auprc:.4f}_f1{best_f1:.4f}"
        f"_chunk{args.chunk_size}_ep{actual_epochs}"
    )
    final_dir = run_base / final_name
    if final_dir.exists():
        # Append timestamp to avoid collision
        final_dir = run_base / f"{final_name}_{run_ts}"
    try:
        save_dir.rename(final_dir)
        logging.info(f"  Run saved to: {final_dir}")
    except OSError:
        # Fallback: if rename fails (cross-device), copy instead
        shutil.copytree(save_dir, final_dir)
        shutil.rmtree(save_dir)
        logging.info(f"  Run saved to: {final_dir}")


if __name__ == "__main__":
    main()
