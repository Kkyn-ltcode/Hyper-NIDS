"""
Training script for HyperMamba Full Architecture.

Uses Truncated Backpropagation Through Time (TBPTT):
  - Events processed in strict chronological order (batch_size=1, shuffle=False)
  - Entity states persist across chunks within an epoch
  - States detached at chunk boundaries to bound memory
  - States reset at epoch boundaries

Usage:
    # Dual validation: L1* for early stopping, broad logged alongside
    python -m src.pipeline.train_full --dataset theia --label_type l1 --dual_val

    # Standard: single val label type
    python -m src.pipeline.train_full --dataset theia --label_type broad
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
from src.model.hypermamba_full import HyperMambaFull

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

        # Clamp logits before loss to prevent overflow
        logits = logits.clamp(-50, 50)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pw_t)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_chunks += 1
            model.detach_bank()
            continue

        # Collect train predictions (detached, no grad impact)
        all_train_logits.extend(logits.detach().squeeze(0).cpu().tolist())
        all_train_labels.extend(y.squeeze(0).cpu().tolist())

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

    train_metrics = compute_metrics(all_train_logits, all_train_labels)
    return avg_loss, chunk_losses, train_metrics


@torch.no_grad()
def warmup_bank(model, loaders, device):
    """Process data shards in eval mode to warm up entity states.
    
    In real deployment, the IDS runs continuously — entity states are never
    'cold'. Before test evaluation, we must simulate this by processing
    the training + val shards in forward-pass-only mode to build up the
    entity state bank. No gradients, no metric collection — just state
    accumulation.
    """
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
    """Evaluate model on a dataset. Always uses pos_weight=1.0 (unweighted BCE)
    since val/test may have different class distributions than training."""
    model.eval()
    # DO NOT reset the bank here! We want to carry the warm states
    # from the end of the training shards into the validation shards.

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

        # Detach bank during eval too
        model.detach_bank()

        all_logits.extend(logits.squeeze(0).cpu().tolist())
        all_labels.extend(y.squeeze(0).cpu().tolist())

    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / max(n_chunks, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train HyperMamba Full (SSM-Driven Taint Propagation)")
    parser.add_argument("--dataset", default="theia", choices=["theia", "trace"])
    parser.add_argument("--label_type", type=str, default="crossprocess", choices=["broad", "crossprocess", "l1"])
    parser.add_argument("--train_label_type", type=str, default=None, help="Override label type for training")
    parser.add_argument("--test_label_type", type=str, default=None, help="Override label type for testing")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--finetune_from", type=str, default=None, help="Path to checkpoint to fine-tune from")
    parser.add_argument("--freeze_body", action="store_true", help="Freeze everything except the classifier head")
    
    # Ablation flags
    parser.add_argument("--no_state", action="store_true", help="Ablation: Disable entity state bank (event features only)")
    parser.add_argument("--no_cross_entity", action="store_true", help="Ablation: Disable cross-entity propagation (self-state only)")
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_pos_weight", type=float, default=30.0,
                        help="Cap pos_weight to prevent gradient explosion")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
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

    # Determine label strategy:
    train_lbl = args.train_label_type if args.train_label_type else args.label_type
    test_lbl = args.test_label_type if args.test_label_type else args.label_type
    val_label_primary = train_lbl

    # Create a timestamped run directory (will be renamed with results at end)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_base = Path("ckpts") / "full_runs"
    save_dir = run_base / f"{args.dataset}_{train_lbl}_{run_ts}"
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



    logging.info("=" * 60)
    logging.info(f"  HYPERMAMBA FULL — {args.dataset.upper()}")
    logging.info("=" * 60)
    logging.info(f"  Device:       {device}")
    logging.info(f"  Train labels: {train_lbl}")
    logging.info(f"  Test labels:  {test_lbl}")
    logging.info(f"  Seed:         {args.seed}")
    logging.info(f"  Train shards: {shards['train']}")
    logging.info(f"  Val shards:   {shards['val']}")

    # --- Data ---
    logging.info(f"\nLoading training data (labels={train_lbl})...")
    train_ds = ChronoDataset(
        shards["train"], data_root,
        chunk_size=args.chunk_size, label_type=train_lbl)

    # All datasets must share the same timestamp reference (t0_nanos)
    # so that dt = t_curr - last_seen is valid across train→val→test boundaries
    t0 = train_ds.t0_nanos
    logging.info(f"  Global t0_nanos: {t0} (all splits share this reference)")

    logging.info(f"Loading validation data (labels={train_lbl})...")
    val_ds = ChronoDataset(
        shards["val"], data_root,
        chunk_size=args.chunk_size, label_type=train_lbl, t0_nanos=t0)

    logging.info(f"Loading testing data (labels={test_lbl})...")
    test_ds = ChronoDataset(
        shards["test"], data_root,
        chunk_size=args.chunk_size, label_type=test_lbl, t0_nanos=t0)

    # Strict chronological: batch_size=1, shuffle=False, num_workers=0
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # --- Model ---
    model = HyperMambaFull(
        num_entities=train_ds.num_entities,
        n_cont_features=train_ds.n_cont_features,
        num_event_types=train_ds.num_event_types,
        d_model=args.d_model,
        use_state=not args.no_state,
        cross_entity=not args.no_cross_entity
    ).to(device)

    if args.finetune_from:
        logging.info(f"Loading checkpoint from {args.finetune_from}")
        ckpt = torch.load(args.finetune_from, map_location=device)
        
        # Handle both raw state_dict and our checkpoint dict wrapper
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        
        # strict=False allows loading even if the classifier head architecture has changed
        # (e.g. going from 4*d_model to 2*d_model input)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logging.info(f"  Missing keys (expected if classifier changed): {len(missing)}")
        if unexpected:
            logging.info(f"  Unexpected keys: {len(unexpected)}")
        
        if args.freeze_body:
            logging.info("Freezing all parameters except the classifier head...")
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"\nTrainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    # Compute pos_weight for training loss — cap to prevent gradient explosion
    n_pos = int((train_ds.y == 1).sum())
    n_neg = int((train_ds.y == 0).sum())
    raw_ratio = n_neg / max(n_pos, 1)
    pos_weight = min(raw_ratio, args.max_pos_weight)
    logging.info(f"  pos_weight: {pos_weight:.1f} (raw ratio {raw_ratio:.1f}, capped at {args.max_pos_weight})")

    # --- Training ---
    best_auprc = 0.0
    best_epoch = 0
    patience = 5
    no_improve = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_auprc': [],
        'train_f1': [],
        'val_auprc': [],
        'val_f1': [],
        'chunk_loss': [],
    }

    # Learning rate scheduler — cosine decay to eta_min
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    logging.info(f"\nStarting training ({args.epochs} epochs)...")

    for epoch in range(1, args.epochs + 1):
        logging.info(f"\n--- Epoch {epoch}/{args.epochs} ---")

        # CRITICAL: Reset the bank at the start of each epoch!
        # Otherwise, epoch 2 starts training on shard 0 using the future
        # entity states left over from the end of the validation set (shard 7).
        # This creates a massive data leak and completely breaks learning.
        model.reset_bank()

        train_loss, epoch_chunk_losses, train_metrics = train_epoch(
            model, train_loader, optimizer, device, pos_weight)

        # Validation (used for early stopping) — bank carries warm state from training
        val_metrics = evaluate(model, val_loader, device)

        # Step LR scheduler
        scheduler.step()
        val_loss = val_metrics["loss"]
        auprc = val_metrics["auprc"]
        f1 = val_metrics["best_f1"]
        t_auprc = train_metrics["auprc"]
        t_f1 = train_metrics["best_f1"]

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['chunk_loss'].extend(epoch_chunk_losses)
        history['train_auprc'].append(t_auprc)
        history['train_f1'].append(t_f1)
        history['val_auprc'].append(auprc)
        history['val_f1'].append(f1)

        logging.info(f"  Train Loss:  {train_loss:.4f}  |  Val({val_label_primary}) Loss: {val_loss:.4f}")
        logging.info(f"  Train AUPRC: {t_auprc:.4f}  |  Train F1: {t_f1:.4f}")
        logging.info(f"  Val({val_label_primary}) AUPRC: {auprc:.4f}  |  Val({val_label_primary}) F1: {f1:.4f}")

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
            logging.info(f"  ✓ New best! AUPRC={auprc:.4f} (early stopping on {val_label_primary})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info(f"  Early stop at epoch {epoch} (best={best_epoch})")
                break

    logging.info(f"\n{'='*60}")
    logging.info(f"  DONE — Best Val({val_label_primary}) AUPRC: {best_auprc:.4f} (epoch {best_epoch})")
    logging.info(f"  Checkpoint: {save_dir / 'best.pt'}")
    logging.info(f"{'='*60}")

    logging.info(f"\nEvaluating on Test Set (labels={test_lbl}) with best model...")
    checkpoint = torch.load(save_dir / 'best.pt')
    model.load_state_dict(checkpoint["model_state"])
    # state_dict() includes bank buffers, but they were accumulated during
    # training using mid-epoch weights. We re-warm the bank using the final
    # best weights in clean eval mode for a stronger, more consistent initialization.
    logging.info(f"  Warming up bank states with best weights...")
    warmup_bank(model, [train_loader, val_loader], device)
    
    test_metrics = evaluate(model, test_loader, device)
    test_loss = test_metrics["loss"]
    test_auprc = test_metrics["auprc"]
    test_f1 = test_metrics["best_f1"]
    
    logging.info(f"  Test Loss:  {test_loss:.4f}")
    logging.info(f"  Test AUPRC: {test_auprc:.4f}")
    logging.info(f"  Test F1:    {test_f1:.4f}")
    logging.info(f"{'='*60}")

    history["test_loss"] = test_loss
    history["test_auprc"] = test_auprc
    history["test_f1"] = test_f1

    # Save training history
    torch.save(history, save_dir / "history.pt")

    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 10))
        ep_range = range(1, len(history['train_loss']) + 1)
        
        # Plot 1: Chunk losses
        plt.subplot(2, 2, 1)
        plt.plot(history['chunk_loss'], alpha=0.5, linewidth=0.5, label='Chunk Loss')
        # Add epoch boundaries as vertical lines
        chunks_per_epoch = len(history['chunk_loss']) // max(len(history['train_loss']), 1)
        for e in range(1, len(history['train_loss'])):
            plt.axvline(x=e * chunks_per_epoch, color='gray', linestyle='--', alpha=0.3)
        plt.xlabel('Chunk')
        plt.ylabel('Loss')
        plt.title('Training Loss per Chunk')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Train vs Val Epoch Loss
        plt.subplot(2, 2, 2)
        plt.plot(ep_range, history['train_loss'], marker='o', color='blue', label='Train Loss')
        plt.plot(ep_range, history['val_loss'], marker='s', color='red', label='Val Loss')
        if 'test_loss' in history:
            plt.plot(best_epoch, history['test_loss'], marker='*', markersize=15, color='darkred', label='Test Loss (Best Epoch)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss: Train vs Validation')
        plt.xticks(ep_range)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Train vs Val AUPRC
        plt.subplot(2, 2, 3)
        plt.plot(ep_range, history['train_auprc'], marker='o', color='blue', label='Train AUPRC')
        plt.plot(ep_range, history['val_auprc'], marker='s', color='green', label=f'Val({val_label_primary}) AUPRC')
        if 'test_auprc' in history:
            plt.plot(best_epoch, history['test_auprc'], marker='*', markersize=15, color='darkgreen', label='Test AUPRC')
        plt.xlabel('Epoch')
        plt.ylabel('AUPRC')
        plt.title('AUPRC: Train vs Validation')
        plt.xticks(ep_range)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Train vs Val F1
        plt.subplot(2, 2, 4)
        plt.plot(ep_range, history['train_f1'], marker='o', color='blue', label='Train F1')
        plt.plot(ep_range, history['val_f1'], marker='s', color='green', label=f'Val({val_label_primary}) F1')
        if 'test_f1' in history:
            plt.plot(best_epoch, history['test_f1'], marker='*', markersize=15, color='darkgreen', label='Test F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1')
        plt.title('F1: Train vs Validation')
        plt.xticks(ep_range)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = save_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logging.info(f"Training curves saved to {plot_path}")
    except ImportError:
        logging.warning("matplotlib is not installed. Skipping loss plot generation.")

    # Rename folder with results
    best_f1 = max(history['val_f1']) if history['val_f1'] else 0.0
    actual_epochs = len(history['train_loss'])
    final_name = (
        f"{args.dataset}_train-{train_lbl}_test-{test_lbl}"
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
