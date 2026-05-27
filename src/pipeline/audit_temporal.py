import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.thyn_dataset import THyNDataset
from src.model.thyn import THyN
from src.pipeline.train import compute_metrics, masked_bce_loss

DATA_ROOT = Path("data/processed/darpa_tc_e3")

def reorder_batch(batch, order="normal"):
    """Reorder events within each sequence in the batch."""
    if order == "normal":
        return batch
    
    X_c = batch["X_cont"].clone()
    et = batch["event_type"].clone()
    y = batch["y"].clone()
    mask = batch["mask"].clone()
    ent = batch["entity_ids"].clone()
    
    bs, seq_len = mask.shape
    for i in range(bs):
        # Find number of real events
        real_len = int(mask[i].sum().item())
        if real_len <= 1:
            continue
            
        if order == "reverse":
            idx = torch.arange(real_len - 1, -1, -1)
        elif order == "shuffle":
            idx = torch.randperm(real_len)
        else:
            raise ValueError(f"Unknown order: {order}")
            
        # Apply reordering to real events
        X_c[i, :real_len] = X_c[i, :real_len][idx]
        et[i, :real_len] = et[i, :real_len][idx]
        y[i, :real_len] = y[i, :real_len][idx]
        ent[i, :real_len] = ent[i, :real_len][idx]
        
    batch["X_cont"] = X_c
    batch["event_type"] = et
    batch["y"] = y
    batch["mask"] = mask
    batch["entity_ids"] = ent
    return batch

@torch.no_grad()
def evaluate_ordered(model, loader, device, order="normal"):
    model.eval()
    all_logits = []
    all_labels = []
    
    t0 = time.time()
    for batch in loader:
        batch = reorder_batch(batch, order=order)
        
        X_c = batch["X_cont"].to(device)
        et = batch["event_type"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        ent = batch["entity_ids"].to(device)
        
        X_c = X_c.clamp(-20, 20)
        logits = model(X_c, et, entity_ids=ent, mask=mask)
        
        real = mask.bool()
        all_logits.extend(logits[real].cpu().tolist())
        all_labels.extend(y[real].cpu().tolist())
        
    elapsed = time.time() - t0
    metrics = compute_metrics(all_logits, all_labels)
    return metrics, elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", default="theia")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    data_root = DATA_ROOT / args.dataset
    
    print(f"Loading validation set...")
    val_ds = THyNDataset(
        dcfg["val_shards"], data_root,
        max_seq_len=dcfg["max_seq_len"],
        label_type=dcfg.get("label_type", "broad"),
        verbose=False
    )
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    print(f"Loading model from {args.ckpt}...")
    model = THyN(
        n_cont_features=val_ds.n_cont_features,
        num_event_types=val_ds.num_event_types,
        d_type=mcfg.get("d_type", 16),
        d_model=mcfg["d_model"],
        d_hidden=mcfg["d_hidden"],
        n_layers=mcfg["n_layers"],
        dropout=mcfg["dropout"],
        model_type=mcfg["model_type"],
        encoder_type=mcfg["encoder_type"],
    ).to(device)
    
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    
    print("\n--- Running Temporal Sensitivity Probe ---")
    orders = ["normal", "reverse", "shuffle"]
    results = {}
    
    for order in orders:
        print(f"\nEvaluating order: {order.upper()}")
        metrics, t = evaluate_ordered(model, val_loader, device, order=order)
        results[order] = metrics["auprc"]
        print(f"  AUPRC: {metrics['auprc']:.4f}")
        print(f"  F1:    {metrics['best_f1']:.4f}")
        
    print("\n--- Summary ---")
    normal_auprc = results["normal"]
    if normal_auprc > 0:
        print(f"Reversed Drop:  {normal_auprc - results['reverse']:.4f} (Score: {results['reverse']/normal_auprc:.3f})")
        print(f"Shuffled Drop:  {normal_auprc - results['shuffle']:.4f} (Score: {results['shuffle']/normal_auprc:.3f})")

if __name__ == "__main__":
    main()
