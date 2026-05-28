import argparse
import time
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
import numpy as np

from src.data.chrono_dataset import ChronoDataset
from src.model.hypermamba_proto import HyperMambaProto

DATA_ROOT = Path("data/processed/darpa_tc_e3")

def compute_metrics(all_logits, all_labels):
    logits_t = torch.tensor(all_logits).clamp(-50, 50)
    probs = torch.sigmoid(logits_t).numpy()
    labels = np.array(all_labels)
    
    valid = labels >= 0
    probs = probs[valid]
    labels = labels[valid]
    
    m = {}
    try:
        m["auprc"] = float(average_precision_score(labels, probs))
    except ValueError:
        m["auprc"] = 0.0
        
    try:
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        f1_all = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        best_idx = np.argmax(f1_all)
        m["best_f1"] = float(f1_all[best_idx])
    except ValueError:
        m["best_f1"] = 0.0
        
    return m

def train_epoch(model, loader, optimizer, device, pos_weight):
    model.train()
    model.reset_bank()
    
    total_loss = 0.0
    t0 = time.time()
    events_processed = 0
    
    for i, batch in enumerate(loader):
        X_c = batch["X_cont"].to(device)
        et = batch["event_type"].to(device)
        y = batch["y"].to(device).float()
        ent = batch["entity_ids"].to(device)
        
        optimizer.zero_grad()
        
        logits = model(X_c, et, ent)
        
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, y, pos_weight=torch.tensor(pos_weight, device=device)
        )
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        events_processed += X_c.size(1)
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            throughput = events_processed / elapsed
            print(f"    Chunk {i+1}/{len(loader)}: loss={loss.item():.4f}, {throughput:.0f} events/s")
            
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    model.reset_bank()
    
    all_logits = []
    all_labels = []
    
    print("  Running validation chronologically...")
    for batch in loader:
        X_c = batch["X_cont"].to(device)
        et = batch["event_type"].to(device)
        y = batch["y"].to(device)
        ent = batch["entity_ids"].to(device)
        
        logits = model(X_c, et, ent)
        
        all_logits.extend(logits.cpu().tolist()[0])
        all_labels.extend(y.cpu().tolist()[0])
        
    return compute_metrics(all_logits, all_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="theia")
    parser.add_argument("--label_type", default="l1")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = DATA_ROOT / args.dataset
    
    # Using existing shards from THyN config
    with open("configs/theia_l1_thyn.yaml") as f:
        cfg = yaml.safe_load(f)
        
    print(f"Loading ChronoDataset for {args.dataset} (label={args.label_type})...")
    train_ds = ChronoDataset(cfg["data"]["train_shards"], data_root, chunk_size=args.chunk_size, label_type=args.label_type)
    val_ds = ChronoDataset(cfg["data"]["val_shards"], data_root, chunk_size=args.chunk_size, label_type="broad") # always eval on broad
    
    # Must use batch_size=1, shuffle=False for chronological TBPTT
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    model = HyperMambaProto(
        num_entities=train_ds.num_entities,
        n_cont_features=train_ds.n_cont_features,
        num_event_types=train_ds.num_event_types,
        d_model=128
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    n_pos = int((train_ds.y == 1).sum())
    n_neg = int((train_ds.y == 0).sum())
    pos_weight = n_neg / max(n_pos, 1)
    print(f"Auto pos_weight: {pos_weight:.1f}")
    
    # Ensure checkpoints directory exists
    Path("checkpoints").mkdir(exist_ok=True)
    
    best_auprc = 0.0
    
    print("\nStarting Training (Chronological TBPTT)")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device, pos_weight)
        
        val_metrics = evaluate(model, val_loader, device)
        auprc = val_metrics["auprc"]
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val AUPRC:  {auprc:.4f}  |  Val F1: {val_metrics['best_f1']:.4f}")
        
        if auprc > best_auprc:
            best_auprc = auprc
            print(f"  --> New best! Saved prototype checkpoint.")
            torch.save(model.state_dict(), f"checkpoints/proto_{args.dataset}_{args.label_type}.pt")

if __name__ == "__main__":
    main()
