import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.thyn_dataset import THyNDataset
from src.model.thyn import THyN
from src.pipeline.train import compute_metrics

DATA_ROOT = Path("data/processed/darpa_tc_e3")

@torch.no_grad()
def evaluate_early_warning(model, loader, device, k_values):
    model.eval()
    
    # We will store logits and labels for each K
    results = {k: {"logits": [], "labels": []} for k in k_values}
    
    t0 = time.time()
    for batch in loader:
        X_c = batch["X_cont"].to(device)
        et = batch["event_type"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        ent = batch["entity_ids"].to(device)
        
        X_c = X_c.clamp(-20, 20)
        
        # We only need to run the model once per batch because it's causal.
        # The output at position i only depends on inputs <= i.
        logits = model(X_c, et, entity_ids=ent, mask=mask)
        
        bs, seq_len = mask.shape
        for i in range(bs):
            real_len = int(mask[i].sum().item())
            if real_len == 0:
                continue
                
            for k in k_values:
                # Only consider sequences that actually reach length K?
                # Or consider the first min(K, real_len) events?
                # Early warning usually means: how much of the attack footprint is needed?
                # Let's take the first K events (or the whole sequence if it's shorter than K).
                limit = min(k, real_len)
                
                # We evaluate event-level predictions up to `limit`
                seq_logits = logits[i, :limit].cpu().tolist()
                seq_labels = y[i, :limit].cpu().tolist()
                
                results[k]["logits"].extend(seq_logits)
                results[k]["labels"].extend(seq_labels)
                
    elapsed = time.time() - t0
    
    metrics_per_k = {}
    for k in k_values:
        metrics = compute_metrics(results[k]["logits"], results[k]["labels"])
        metrics_per_k[k] = metrics
        
    return metrics_per_k, elapsed

def main():
    parser = argparse.ArgumentParser(description="Experiment III: Early Warning Test")
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
        label_type="broad", # Always evaluate against broad ground truth
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
    
    print("\n--- Running Experiment III: Early Warning Test ---")
    k_values = [5, 10, 20, 50, 100, 200, 512]
    
    metrics_per_k, elapsed = evaluate_early_warning(model, val_loader, device, k_values)
    
    print(f"\nEvaluation completed in {elapsed:.1f}s.")
    print(f"{'K (Events)':<15} | {'AUPRC':<10} | {'F1':<10}")
    print("-" * 40)
    for k in k_values:
        m = metrics_per_k[k]
        k_label = str(k) if k < 512 else "All (512)"
        print(f"{k_label:<15} | {m['auprc']:.4f}     | {m['best_f1']:.4f}")
        
    print("\nDecision Rule:")
    print("If AUPRC drops heavily at low K, the model relies on seeing the full attack footprint.")
    print("If AUPRC remains high at low K, the model successfully propagates context to detect attacks early.")

if __name__ == "__main__":
    main()
