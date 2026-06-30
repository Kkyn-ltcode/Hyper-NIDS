"""Shard-level diagnostic: evaluate existing best.pt on individual test shards.

Usage:
    python -m src.pipeline.diagnose_shards \
        --checkpoint ckpts/full_runs/theia_full_crossprocess_ablation-full_20260629_094926/best.pt \
        --data_root data/processed/darpa_tc_e3/theia

This evaluates the checkpoint on:
  - Shard 10 alone (Campaign 0 continuation — same campaign as training)
  - Shards 22,23,24 combined (Campaign 2 — cross-campaign)

Prediction: if the process embedding shortcut hypothesis is correct,
shard 10 will show dramatically worse event-level AUPRC than 22-24.
"""
import argparse
import logging
import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.chrono_dataset import ChronoDataset
from src.model.hypermamba_full import HyperMambaFull
from src.pipeline.train_full import evaluate, warmup_bank, SHARD_CONFIG

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--data_root", required=True, type=str)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--label_type", type=str, default="crossprocess")
    parser.add_argument("--regime", type=str, default="A", choices=["A", "B"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    
    # Load train set to get t0 and model dimensions
    train_shards = SHARD_CONFIG["theia"]["full"]["train"]
    val_shards = SHARD_CONFIG["theia"]["full"]["val"]
    
    logging.info("Loading train dataset for t0 reference and model init...")
    train_ds = ChronoDataset(train_shards, data_root, chunk_size=args.chunk_size,
                             label_type=args.label_type)
    t0 = train_ds.t0_nanos
    
    # Initialize model
    model = HyperMambaFull(
        num_entities=train_ds.num_entities,
        n_cont_features=train_ds.n_cont_features,
        num_event_types=train_ds.num_event_types,
        num_process_names=train_ds.num_process_names,
        d_model=128,
    ).to(device)
    
    # Load checkpoint
    logging.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.info(f"  Missing keys: {missing}")
    if unexpected:
        logging.info(f"  Unexpected keys: {unexpected}")
    
    # Define test shard groups
    shard_groups = {
        "Shard 10 (C0 continuation)": [10],
        "Shards 22-24 (C2 cross-campaign)": [22, 23, 24],
        "Full test [10,22,23,24]": [10, 22, 23, 24],
    }
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    val_ds = ChronoDataset(val_shards, data_root, chunk_size=args.chunk_size,
                           label_type=args.label_type, t0_nanos=t0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    
    for name, shards in shard_groups.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"Evaluating: {name}")
        logging.info(f"{'='*60}")
        
        test_ds = ChronoDataset(shards, data_root, chunk_size=args.chunk_size,
                                label_type=args.label_type, t0_nanos=t0)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
        
        n_pos = int((test_ds.y == 1).sum())
        n_neg = int((test_ds.y == 0).sum())
        logging.info(f"  Events: {len(test_ds.y):,} ({n_pos:,} pos / {n_neg:,} neg)")
        
        if args.regime == "B":
            model.reset_bank()
        else:
            warmup_bank(model, [train_loader, val_loader], device)
        
        metrics, preds = evaluate(model, test_loader, device, return_preds=True)
        
        logging.info(f"  --- Event-level ---")
        logging.info(f"  AUPRC: {metrics['auprc']:.4f}  |  AUROC: {metrics.get('auroc', 0):.4f}")
        logging.info(f"  Prec:  {metrics.get('precision', 0):.4f}  |  "
                     f"Rec:   {metrics.get('recall', 0):.4f}  |  "
                     f"F1:    {metrics.get('best_f1', 0):.4f}")
        logging.info(f"  FPR:   {metrics.get('fpr', 0):.6f}")
        
        if "node_auprc" in metrics:
            logging.info(f"  --- Node-level ({metrics.get('node_n_pos', 0):,} pos / "
                         f"{metrics.get('node_n_neg', 0):,} neg) ---")
            logging.info(f"  AUPRC: {metrics['node_auprc']:.4f}  |  "
                         f"AUROC: {metrics.get('node_auroc', 0):.4f}")
            logging.info(f"  Prec:  {metrics.get('node_precision', 0):.4f}  |  "
                         f"Rec:   {metrics.get('node_recall', 0):.4f}  |  "
                         f"F1:    {metrics.get('node_best_f1', 0):.4f}")

if __name__ == "__main__":
    main()
