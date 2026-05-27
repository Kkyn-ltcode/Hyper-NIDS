import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import yaml

from src.data.thyn_dataset import THyNDataset

DATA_ROOT = Path("data/processed/darpa_tc_e3")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/theia_thyn.yaml")
    parser.add_argument("--dataset", default="theia")
    parser.add_argument("--out-dir", default="experiments/audits")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        
    dcfg = cfg["data"]
    data_root = DATA_ROOT / args.dataset
    
    print("Loading training dataset...")
    train_ds = THyNDataset(
        dcfg["train_shards"], data_root,
        max_seq_len=dcfg["max_seq_len"],
        label_type="broad",
        verbose=True
    )
    
    loader = DataLoader(train_ds, batch_size=128, shuffle=False)
    
    print("\nAnalyzing attack event positions...")
    
    # Trackers
    atk_seq_lengths = []
    benign_seq_lengths = []
    
    atk_positions_normalized = []
    atk_event_types = []
    atk_positions_absolute = []
    
    for batch_idx, batch in enumerate(loader):
        mask = batch["mask"].numpy()
        y = batch["y"].numpy()
        et = batch["event_type"].numpy()
        
        bs, seq_len = mask.shape
        
        for i in range(bs):
            real_len = int(mask[i].sum())
            if real_len == 0:
                continue
                
            labels = y[i, :real_len]
            has_attack = (labels == 1).any()
            
            if has_attack:
                atk_seq_lengths.append(real_len)
                
                # Find positions of attack events
                atk_idx = np.where(labels == 1)[0]
                
                atk_positions_absolute.extend(atk_idx.tolist())
                # Normalize position to 0-1
                if real_len > 1:
                    norm_pos = atk_idx / (real_len - 1)
                else:
                    norm_pos = np.zeros_like(atk_idx, dtype=float)
                    
                atk_positions_normalized.extend(norm_pos.tolist())
                atk_event_types.extend(et[i, atk_idx].tolist())
                
            else:
                benign_seq_lengths.append(real_len)
                
        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {batch_idx + 1} batches...")
            
    print(f"\nAnalyzed {len(atk_seq_lengths):,} attack sequences and {len(benign_seq_lengths):,} benign sequences.")
    
    if not atk_positions_normalized:
        print("No attacks found!")
        return
        
    # --- Analytics ---
    atk_pos_norm = np.array(atk_positions_normalized)
    
    p10 = (atk_pos_norm <= 0.10).mean() * 100
    p25 = (atk_pos_norm <= 0.25).mean() * 100
    p50 = (atk_pos_norm <= 0.50).mean() * 100
    
    print("\n--- Attack Event Temporal Distribution ---")
    print(f"  First 10% of sequence: {p10:.1f}% of attacks")
    print(f"  First 25% of sequence: {p25:.1f}% of attacks")
    print(f"  First 50% of sequence: {p50:.1f}% of attacks")
    
    mean_atk_len = np.mean(atk_seq_lengths)
    mean_ben_len = np.mean(benign_seq_lengths)
    print(f"\n--- Sequence Lengths ---")
    print(f"  Mean Attack Sequence Length: {mean_atk_len:.1f} events")
    print(f"  Mean Benign Sequence Length: {mean_ben_len:.1f} events")
    
    # --- Plots ---
    plt.figure(figsize=(10, 6))
    sns.histplot(atk_pos_norm, bins=50, kde=True, color="red")
    plt.title("Distribution of Attack Events (Normalized Position in Sequence)")
    plt.xlabel("Normalized Position (0=Start, 1=End)")
    plt.ylabel("Frequency")
    plt.savefig(out_dir / "attack_position_dist.png")
    plt.close()
    
    print(f"\nSaved plots to {out_dir}")

if __name__ == "__main__":
    main()
