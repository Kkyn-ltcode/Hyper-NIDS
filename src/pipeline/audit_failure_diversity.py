import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from collections import Counter
from scipy.stats import entropy

from src.data.thyn_dataset import THyNDataset
from src.model.thyn import THyN

DATA_ROOT = Path("data/processed/darpa_tc_e3")

@torch.no_grad()
def evaluate_and_collect_failures(model, loader, device):
    model.eval()
    
    false_negatives = []
    
    print("Evaluating and collecting false negatives...")
    for batch_idx, batch in enumerate(loader):
        X_c = batch["X_cont"].to(device)
        et = batch["event_type"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        ent = batch["entity_ids"].to(device)
        
        X_c = X_c.clamp(-20, 20)
        logits = model(X_c, et, entity_ids=ent, mask=mask)
        probs = torch.sigmoid(logits)
        
        preds = (probs > 0.5).long()
        
        # Analyze failures per sequence
        bs, seq_len = mask.shape
        for i in range(bs):
            real_len = int(mask[i].sum().item())
            if real_len == 0:
                continue
                
            labels = y[i, :real_len].cpu().numpy()
            pred = preds[i, :real_len].cpu().numpy()
            event_types = et[i, :real_len].cpu().numpy()
            
            # Find false negatives (True label is 1, pred is 0)
            fn_mask = (labels == 1) & (pred == 0)
            fn_indices = np.where(fn_mask)[0]
            
            for idx in fn_indices:
                false_negatives.append({
                    "event_type": event_types[idx],
                    "norm_position": idx / max(1, real_len - 1),
                    "seq_length": real_len,
                })
                
        if (batch_idx + 1) % 100 == 0:
            print(f"  Processed {batch_idx + 1} batches...")
            
    return false_negatives

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
    
    print(f"Loading validation set for {args.dataset}...")
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
    
    # Collect False Negatives
    fns = evaluate_and_collect_failures(model, val_loader, device)
    
    if not fns:
        print("No false negatives found! Model is perfect (or evaluation failed).")
        return
        
    print(f"\n--- Failure Diversity Analysis (N={len(fns)} False Negatives) ---")
    
    # Cluster 1: Event Type
    et_counts = Counter([f["event_type"] for f in fns])
    et_dist = np.array(list(et_counts.values())) / len(fns)
    et_entropy = entropy(et_dist)
    print(f"\n1. Event Type Entropy: {et_entropy:.3f}")
    print("  Top Event Types missed:")
    for k, v in et_counts.most_common(5):
        print(f"    Type {k}: {v/len(fns)*100:.1f}%")
        
    # Cluster 2: Temporal Position
    pos_counts = Counter([int(f["norm_position"] * 10) for f in fns]) # Deciles
    pos_dist = np.array(list(pos_counts.values())) / len(fns)
    pos_entropy = entropy(pos_dist)
    print(f"\n2. Temporal Position Entropy (Deciles): {pos_entropy:.3f}")
    print("  Position distributions missed:")
    for k, v in sorted(pos_counts.items()):
        print(f"    Decile {k} ({k*10}% - {(k+1)*10}%): {v/len(fns)*100:.1f}%")
        
    # Cluster 3: Sequence Length (Short vs Long)
    len_bins = [10, 50, 100, 200, 500]
    len_counts = Counter([np.digitize(f["seq_length"], len_bins) for f in fns])
    len_dist = np.array(list(len_counts.values())) / len(fns)
    len_entropy = entropy(len_dist)
    print(f"\n3. Sequence Length Entropy: {len_entropy:.3f}")
    print("  Missed by sequence length:")
    labels = ["<10", "10-50", "50-100", "100-200", "200-500", ">500"]
    for k, v in sorted(len_counts.items()):
        print(f"    Length {labels[k]}: {v/len(fns)*100:.1f}%")
        
    print("\nDecision Rule:")
    print("If entropy is LOW, failures are clustered (systematic blind spot).")
    print("If entropy is HIGH, failures are uniformly distributed (model fails randomly).")

if __name__ == "__main__":
    main()
