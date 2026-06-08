import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator

from src.data.chrono_dataset import ChronoDataset

DATA_ROOT = Path("data/processed/darpa_tc_e3")

SHARD_CONFIG = {
    "theia": {"train": list(range(7)), "val": [7], "test": [8, 9]},
    "trace": {"train": list(range(5)), "val": [5], "test": [6]},
}

class TGNBaseline(nn.Module):
    """
    Fair KAIROS-style baseline: TGN Memory (GRU) + MLP Classifier.
    Processes ChronoDataset batches, directly swapping HyperMamba components
    for PyG standard TGN components.
    """
    def __init__(self, num_entities, msg_dim, d_model=128):
        super().__init__()
        self.d_model = d_model
        self.dummy_node = num_entities
        
        # TGN Memory uses GRU under the hood. Add +1 to num_nodes for the dummy sink.
        self.memory = TGNMemory(
            num_nodes=num_entities + 1,
            raw_msg_dim=msg_dim,
            memory_dim=d_model,
            time_dim=d_model,
            message_module=IdentityMessage(msg_dim, d_model, d_model),
            aggregator_module=LastAggregator()
        )
        
        # Classifier Head: [src_state, dst_state, msg] -> scalar
        # Since KAIROS operates on edges (src->dst), we ignore obj2.
        classifier_in_dim = d_model * 2 + msg_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        
    def reset_bank(self):
        self.memory.reset_state()
        
    def detach_bank(self):
        self.memory.detach()
        
    def forward(self, src, dst, t, msg):
        """
        src: (C,)
        dst: (C,)
        t: (C,)
        msg: (C, msg_dim)
        """
        # Route missing entities (-1) to the dummy sink node
        safe_src = torch.where(src >= 0, src, self.dummy_node)
        safe_dst = torch.where(dst >= 0, dst, self.dummy_node)
        
        # We need the memory states *before* the update to predict the current event
        n_id = torch.cat([safe_src, safe_dst]).unique()
        z, last_update = self.memory(n_id)
        
        # Create a mapping from global node ID to the index in z
        assoc = torch.empty(self.memory.memory.size(0), dtype=torch.long, device=z.device)
        assoc[n_id] = torch.arange(n_id.size(0), device=z.device)
        
        z_src = z[assoc[safe_src]]
        z_dst = z[assoc[safe_dst]]
        
        # Zero out invalid states so they don't corrupt the classifier
        z_src = z_src * (src >= 0).unsqueeze(-1)
        z_dst = z_dst * (dst >= 0).unsqueeze(-1)
        
        # Predict
        cls_input = torch.cat([z_src, z_dst, msg], dim=-1)
        logits = self.classifier(cls_input).squeeze(-1)
        
        # Update memory *after* prediction
        self.memory.update_state(safe_src, safe_dst, t, msg)
        
        return logits


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


def train_epoch(model, loader, optimizer, device, pos_weight, num_event_types, grad_clip=1.0):
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
        # Extract features
        X_c = torch.nan_to_num(batch["X_cont"].to(device), nan=0.0).clamp(-20, 20)
        et = batch["event_type"].to(device)
        et_onehot = torch.nn.functional.one_hot(et, num_classes=num_event_types).float()
        
        # Combine into msg
        msg = torch.cat([X_c, et_onehot], dim=-1)
        
        # KAIROS focuses on src->dst edges (ignores hyperedge obj2)
        src = batch["entity_ids"][:, 0].to(device)
        dst = batch["entity_ids"][:, 1].to(device)
        t = batch["timestamp"].to(device).long()
        y = batch["y"].to(device).float()

        optimizer.zero_grad()

        # Forward + update
        logits = model(src, dst, t, msg)
        logits = logits.clamp(-50, 50)
        
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=pw_t)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_chunks += 1
            model.detach_bank()
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Detach TBPTT chunk
        model.detach_bank()

        ls_val = loss.item()
        total_loss += ls_val
        n_chunks += 1
        chunk_losses.append(ls_val)

        events_processed += len(y)
        all_train_logits.extend(logits.detach().cpu().numpy())
        all_train_labels.extend(y.cpu().numpy())

        if (i + 1) % 50 == 0:
            speed = events_processed / (time.time() - t0)
            logging.info(f"  Chunk {i+1} | Loss: {ls_val:.4f} | Speed: {speed:.0f} ev/s")

    avg_loss = total_loss / max(1, n_chunks)
    metrics = compute_metrics(all_train_logits, all_train_labels)

    if nan_chunks > 0:
        logging.warning(f"  WARNING: Skipped {nan_chunks} chunks due to NaN loss")

    return avg_loss, chunk_losses, metrics


@torch.no_grad()
def evaluate(model, loader, device, num_event_types):
    model.eval()
    total_loss = 0.0
    n_chunks = 0

    all_logits = []
    all_labels = []

    for batch in loader:
        X_c = torch.nan_to_num(batch["X_cont"].to(device), nan=0.0).clamp(-20, 20)
        et = batch["event_type"].to(device)
        et_onehot = torch.nn.functional.one_hot(et, num_classes=num_event_types).float()
        msg = torch.cat([X_c, et_onehot], dim=-1)
        
        src = batch["entity_ids"][:, 0].to(device)
        dst = batch["entity_ids"][:, 1].to(device)
        t = batch["timestamp"].to(device).long()
        y = batch["y"].to(device).float()

        logits = model(src, dst, t, msg)
        logits = logits.clamp(-50, 50)
        
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

        model.detach_bank()

        total_loss += loss.item()
        n_chunks += 1
        all_logits.extend(logits.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / max(1, n_chunks)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="TGN Baseline (Fair KAIROS)")
    parser.add_argument("--dataset", default="theia", choices=["theia", "trace"])
    parser.add_argument("--label_type", default="crossprocess", help="Target label column")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Chunk size for TBPTT")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_pos_weight", type=float, default=30.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models_dir = Path(f"models/fair_kairos/{args.dataset}_{args.label_type}")
    models_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(models_dir / "train.log"),
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"--- Starting Fair KAIROS Baseline ---")
    logging.info(f"Args: {args}")

    # --- Data ---
    shards = SHARD_CONFIG[args.dataset]
    data_root = DATA_ROOT / args.dataset
    ds_kwargs = dict(
        data_root=data_root,
        label_type=args.label_type,
        chunk_size=args.batch_size
    )

    train_ds = ChronoDataset(shard_ids=shards["train"], **ds_kwargs)
    val_ds = ChronoDataset(shard_ids=shards["val"], t0_nanos=train_ds.t0_nanos, **ds_kwargs)
    test_ds = ChronoDataset(shard_ids=shards["test"], t0_nanos=train_ds.t0_nanos, **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=None, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=None, num_workers=0)

    # --- Model ---
    msg_dim = train_ds.n_cont_features + train_ds.num_event_types
    model = TGNBaseline(
        num_entities=train_ds.num_entities,
        msg_dim=msg_dim,
        d_model=args.d_model
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    n_pos = int((train_ds.y == 1).sum())
    n_neg = int((train_ds.y == 0).sum())
    raw_ratio = n_neg / max(n_pos, 1)
    pos_weight = min(raw_ratio, args.max_pos_weight)
    logging.info(f"  pos_weight: {pos_weight:.1f} (raw ratio {raw_ratio:.1f})")

    best_auprc = 0.0
    best_epoch = 0

    # --- Training Loop ---
    for epoch in range(1, args.epochs + 1):
        logging.info(f"\n--- Epoch {epoch}/{args.epochs} ---")
        model.reset_bank()

        train_loss, _, train_metrics = train_epoch(
            model, train_loader, optimizer, device, pos_weight, train_ds.num_event_types)

        val_metrics = evaluate(model, val_loader, device, train_ds.num_event_types)
        scheduler.step()

        logging.info(f"Train | Loss: {train_loss:.4f} | AUPRC: {train_metrics['auprc']:.4f} | F1: {train_metrics['best_f1']:.4f}")
        logging.info(f"Val   | Loss: {val_metrics['loss']:.4f} | AUPRC: {val_metrics['auprc']:.4f} | F1: {val_metrics['best_f1']:.4f}")

        if val_metrics["auprc"] > best_auprc:
            best_auprc = val_metrics["auprc"]
            best_epoch = epoch
            torch.save(model.state_dict(), models_dir / "best_model.pt")
            logging.info("  -> Saved new best model")

    # --- Test Evaluation ---
    logging.info(f"\n--- Testing (Best Model from Epoch {best_epoch}) ---")
    model.load_state_dict(torch.load(models_dir / "best_model.pt", map_location=device))
    model.reset_bank()
    
    # Warm up memory on train+val (required for TGN)
    logging.info("Warming up memory on train+val shards...")
    evaluate(model, train_loader, device, train_ds.num_event_types)
    evaluate(model, val_loader, device, train_ds.num_event_types)
    
    # Test evaluation
    test_metrics = evaluate(model, test_loader, device, train_ds.num_event_types)
    logging.info(f"Test  | Loss: {test_metrics['loss']:.4f} | AUPRC: {test_metrics['auprc']:.4f} | F1: {test_metrics['best_f1']:.4f}")
    
    print("\nTraining Complete!")
    print(f"Val AUPRC:  {best_auprc:.4f}")
    print(f"Test AUPRC: {test_metrics['auprc']:.4f}")

if __name__ == "__main__":
    main()
