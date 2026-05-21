"""
THyN Training Script — Single & Multi-GPU with correct DDP and best‑threshold.
"""

import argparse
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
import yaml

from src.data.thyn_dataset import THyNDataset
from src.model.thyn import THyN

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "darpa_tc_e3"

# ---------- DDP helpers ----------
def is_distributed():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_distributed() else 0

def get_world_size():
    return dist.get_world_size() if is_distributed() else 1

def is_main():
    return get_rank() == 0

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return None

def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()

def log(msg):
    if is_main():
        print(msg)

# ---------- Loss and metrics ----------
def masked_bce_loss(logits, y, mask, pos_weight_t):
    real = mask.bool()
    logits_real = logits[real]
    y_real = y[real].float()
    if logits_real.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits_real = logits_real.clamp(-50, 50)   # prevent overflow
    loss = nn.functional.binary_cross_entropy_with_logits(logits_real, y_real, pos_weight=pos_weight_t)
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return loss

def compute_metrics(all_logits, all_labels, fixed_threshold=0.5):
    """Return dict with AUPRC, AUROC, F1@fixed, best F1 & threshold."""
    probs = torch.sigmoid(torch.tensor(all_logits).clamp(-50, 50)).numpy()
    labels = np.array(all_labels)
    valid = labels >= 0
    probs = probs[valid]
    labels = labels[valid]
    # Replace any NaN
    nan_count = int(np.isnan(probs).sum())
    if nan_count > 0:
        probs = np.nan_to_num(probs, nan=0.5)
    m = {"nan_count": nan_count, "fixed_threshold": fixed_threshold}
    try:
        m["auprc"] = float(average_precision_score(labels, probs))
    except ValueError:
        m["auprc"] = 0.0
    try:
        m["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        m["auroc"] = 0.0
    preds = (probs > fixed_threshold).astype(int)
    m["f1"] = float(f1_score(labels, preds, zero_division=0))
    # Best threshold F1
    try:
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        best_idx = np.argmax(f1_scores)
        m["best_f1"] = float(f1_scores[best_idx])
        m["best_f1_threshold"] = float(thresholds[best_idx])
    except ValueError:
        m["best_f1"] = 0.0
        m["best_f1_threshold"] = 0.5
    m["n_attack"] = int(labels.sum())
    m["n_benign"] = int((labels == 0).sum())
    m["pred_attack"] = int(preds.sum())
    return m

# ---------- Training ----------
def train_epoch(model, loader, optimizer, pw_t, grad_clip, device, log_every=100):
    model.train()
    total_loss = 0.0
    n_batches = 0
    total_events = 0
    t0 = time.time()
    nan_batches = 0
    for i, batch in enumerate(loader):
        X_c = batch["X_cont"].to(device, non_blocking=True).clamp(-100, 100)
        et = batch["event_type"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        ent = batch["entity_ids"].to(device, non_blocking=True)

        logits = model(X_c, et, entity_ids=ent, mask=mask)
        loss = masked_bce_loss(logits, y, mask, pw_t)

        optimizer.zero_grad()
        loss.backward()

        # Skip NaN gradients
        has_nan_grad = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                           for p in model.parameters())
        if has_nan_grad:
            nan_batches += 1
            optimizer.zero_grad()
            continue

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        total_events += int(mask.sum().item())

        if log_every and (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            throughput = total_events / elapsed
            mem = f", GPU={torch.cuda.max_memory_allocated(device)/1e9:.1f}GB" if torch.cuda.is_available() else ""
            log(f"    batch {i+1}: loss={total_loss/n_batches:.4f}, {throughput:.0f} ev/s{mem}")

    elapsed = time.time() - t0
    throughput = total_events / elapsed if elapsed > 0 else 0
    if nan_batches:
        log(f"    ⚠ {nan_batches} batches skipped (NaN gradients)")
    return total_loss / max(n_batches, 1), throughput

# ---------- Distributed evaluation (gathers all logits/labels to rank0) ----------
@torch.no_grad()
def evaluate(model, loader, pw_t, device, world_size, max_batches=None, fixed_threshold=0.5):
    model.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    total_events = 0
    t0 = time.time()

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        X_c = batch["X_cont"].to(device, non_blocking=True).clamp(-100, 100)
        et = batch["event_type"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        ent = batch["entity_ids"].to(device, non_blocking=True)

        m = model.module if isinstance(model, DDP) else model
        logits = m(X_c, et, entity_ids=ent, mask=mask)
        loss = masked_bce_loss(logits, y, mask, pw_t)

        total_loss += loss.item()
        n_batches += 1
        real = mask.bool()
        all_logits.extend(logits[real].cpu().tolist())
        all_labels.extend(y[real].cpu().tolist())
        total_events += int(real.sum().item())

    # Convert to tensors for gathering
    logits_tensor = torch.tensor(all_logits, device=device)
    labels_tensor = torch.tensor(all_labels, device=device)

    if is_distributed():
        # Gather from all ranks
        gathered_logits = [torch.zeros_like(logits_tensor) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_logits, logits_tensor)
        dist.all_gather(gathered_labels, labels_tensor)
        if is_main():
            all_logits = torch.cat(gathered_logits).cpu().tolist()
            all_labels = torch.cat(gathered_labels).cpu().tolist()
        else:
            all_logits, all_labels = [], []
    else:
        all_logits = all_logits
        all_labels = all_labels

    elapsed = time.time() - t0
    metrics = compute_metrics(all_logits if is_main() else [], all_labels if is_main() else [], fixed_threshold)
    if is_main():
        metrics["loss"] = total_loss / max(n_batches, 1)
        metrics["throughput"] = total_events / elapsed if elapsed > 0 else 0
    return metrics

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/thyn_v0.yaml")
    parser.add_argument("--dataset", default="theia")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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
    save_dir = Path("checkpoints") / cfg.get("name", "thyn_v0")
    if is_main():
        save_dir.mkdir(parents=True, exist_ok=True)

    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]

    log("="*60)
    log(f"TRAIN: {cfg.get('name', 'THyN v0').upper()}")
    log(f"  Device: {device}, World size: {world_size}")
    log(f"  Model: {mcfg['model_type']}, Encoder: {mcfg['encoder_type']}")
    log(f"  Labels: {dcfg.get('label_type', 'broad')}, Dataset: {args.dataset}")

    # ---------- Data ----------
    log("\n[1/4] Loading data...")
    label_type = dcfg.get("label_type", "broad")
    train_ds = THyNDataset(dcfg["train_shards"], data_root,
                           max_seq_len=dcfg["max_seq_len"],
                           stride=dcfg.get("stride", dcfg["max_seq_len"]),
                           label_type=label_type)
    val_ds = THyNDataset(dcfg["val_shards"], data_root,
                         max_seq_len=dcfg["max_seq_len"],
                         label_type=label_type)
    test_ds = THyNDataset(dcfg["test_shards"], data_root,
                          max_seq_len=dcfg["max_seq_len"],
                          label_type=label_type)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed() else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if is_distributed() else None

    bs = tcfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, sampler=val_sampler,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, sampler=test_sampler,
                             num_workers=2, pin_memory=True)

    log(f"  Train: {len(train_ds):,} windows, {len(train_ds.X_cont):,} events")
    log(f"  Val:   {len(val_ds):,} windows, {len(val_ds.X_cont):,} events")
    log(f"  Test:  {len(test_ds):,} windows, {len(test_ds.X_cont):,} events")
    log(f"  Cont features: {train_ds.n_cont_features}, Event types: {train_ds.num_event_types}")

    # ---------- Model ----------
    log("\n[2/4] Building model...")
    model = THyN(
        n_cont_features=train_ds.n_cont_features,
        num_event_types=train_ds.num_event_types,
        d_type=mcfg.get("d_type", 16),
        d_model=mcfg["d_model"],
        d_hidden=mcfg["d_hidden"],
        n_layers=mcfg["n_layers"],
        dropout=mcfg["dropout"],
        model_type=mcfg["model_type"],
        encoder_type=mcfg["encoder_type"],
    ).to(device)
    if is_distributed():
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
    log(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer & scheduler
    effective_lr = tcfg["lr"] * world_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # pos_weight handling
    pw_cfg = tcfg.get("pos_weight", 1.0)
    if pw_cfg == "auto":
        n_pos = int((train_ds.y == 1).sum())
        n_neg = int((train_ds.y == 0).sum())
        pw_val = n_neg / max(n_pos, 1)
        log(f"  Auto pos_weight: {pw_val:.1f} (neg={n_neg:,} / pos={n_pos:,})")
    else:
        pw_val = float(pw_cfg)
        log(f"  Config pos_weight: {pw_val:.1f}")
    pw_t = torch.tensor([pw_val], device=device)

    epochs = 1 if args.quick else tcfg["epochs"]
    patience = tcfg["patience"]
    log_every = tcfg["log_every"]
    grad_clip = tcfg["grad_clip"]

    # ---------- Training ----------
    log(f"\n[3/4] Training ({epochs} epochs, batch={bs}×{world_size}={bs*world_size})...")
    best_auprc = -1.0
    best_epoch = 0
    no_improve = 0
    best_threshold = 0.5

    for epoch in range(1, epochs+1):
        log(f"\n  ── Epoch {epoch}/{epochs} ──")
        if train_sampler:
            train_sampler.set_epoch(epoch)

        train_loss, train_tp = train_epoch(model, train_loader, optimizer, pw_t, grad_clip, device, log_every)

        # Validation (all ranks evaluate their subset, metrics gathered on rank0)
        val_mb = 50 if args.quick else None
        vm = evaluate(model, val_loader, pw_t, device, world_size, max_batches=val_mb)

        if is_main():
            scheduler.step(vm["auprc"])
            log(f"  Train: loss={train_loss:.4f}, {train_tp:.0f} ev/s"
                f"{', GPU peak='+str(torch.cuda.max_memory_allocated(device)/1e9)+'GB' if torch.cuda.is_available() else ''}")
            log(f"  Val:   loss={vm['loss']:.4f}, AUPRC={vm['auprc']:.4f}, AUROC={vm['auroc']:.4f}, "
                f"F1@{vm['fixed_threshold']:.3f}={vm['f1']:.4f}, BestF1={vm['best_f1']:.4f}@{vm['best_f1_threshold']:.3f}, "
                f"{vm['throughput']:.0f} ev/s, NaN_logits={vm['nan_count']}")
            if vm["auprc"] > best_auprc:
                best_auprc = vm["auprc"]
                best_epoch = epoch
                best_threshold = vm["best_f1_threshold"]
                no_improve = 0
                raw = model.module if isinstance(model, DDP) else model
                torch.save({
                    "epoch": epoch,
                    "model_state": raw.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics": vm,
                    "best_threshold": best_threshold,
                    "config": cfg,
                }, save_dir / "best.pt")
                log(f"  ✓ New best! AUPRC={best_auprc:.4f}, best_threshold={best_threshold:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    log(f"  Early stop (best epoch {best_epoch})")
                    break

        # Broadcast early stop to all ranks
        if is_distributed():
            stop = torch.tensor([no_improve >= patience], dtype=torch.bool, device=device)
            dist.broadcast(stop, src=0)
            if stop.item():
                break

    # ---------- Final evaluation on test set ----------
    if is_main():
        log("\n[4/4] Final evaluation...")
        # Load best checkpoint
        raw = model.module if isinstance(model, DDP) else model
        ckpt = torch.load(save_dir / "best.pt", map_location=device)
        raw.load_state_dict(ckpt["model_state"])
        best_threshold = ckpt.get("best_threshold", 0.5)
        log(f"  Using best threshold from validation: {best_threshold:.4f}")

        # Evaluate on validation again (full, no max_batches)
        vm_final = evaluate(model, val_loader, pw_t, device, world_size, max_batches=None, fixed_threshold=best_threshold)
        # Evaluate on test
        tm = evaluate(model, test_loader, pw_t, device, world_size, max_batches=None, fixed_threshold=best_threshold)

        log(f"\n{'='*60}")
        log(f"RESULTS — {cfg.get('name', 'THyN v0')}")
        log(f"  Best epoch: {best_epoch}")
        log(f"  --- Validation (using best_threshold={best_threshold:.4f}) ---")
        log(f"  AUPRC: {vm_final['auprc']:.4f}, AUROC: {vm_final['auroc']:.4f}, F1: {vm_final['f1']:.4f}")
        log(f"  --- Test (threshold={best_threshold:.4f}) ---")
        log(f"  AUPRC: {tm['auprc']:.4f}, AUROC: {tm['auroc']:.4f}, F1: {tm['f1']:.4f}")
        log(f"  Test Best F1 (oracle): {tm['best_f1']:.4f} @ threshold={tm['best_f1_threshold']:.4f}")
        log(f"  Throughput: {tm['throughput']:.0f} ev/s")

        torch.save({
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "val_metrics": vm_final,
            "test_metrics": tm,
            "config": cfg,
        }, save_dir / "results.pt")

    cleanup_distributed()

if __name__ == "__main__":
    main()