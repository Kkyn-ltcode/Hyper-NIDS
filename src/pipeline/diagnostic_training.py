import argparse
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_log(log_path):
    epochs = []
    train_loss = []
    val_auprc = []
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    current_epoch = -1
    for line in lines:
        # Match [Epoch X]
        ep_match = re.search(r'\[Epoch (\d+)\]', line)
        if ep_match:
            current_epoch = int(ep_match.group(1))
            if current_epoch not in epochs:
                epochs.append(current_epoch)
                
        # Match Train Loss
        tl_match = re.search(r'Train Loss:\s*([0-9.]+)', line)
        if tl_match and current_epoch == epochs[-1] and len(train_loss) < len(epochs):
            train_loss.append(float(tl_match.group(1)))
            
        # Match Val AUPRC
        va_match = re.search(r'Val AUPRC:\s*([0-9.]+)', line)
        if va_match and current_epoch == epochs[-1] and len(val_auprc) < len(epochs):
            val_auprc.append(float(va_match.group(1)))
            
    # Ensure lists are aligned
    min_len = min(len(epochs), len(train_loss), len(val_auprc))
    return epochs[:min_len], train_loss[:min_len], val_auprc[:min_len]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="theia")
    args = parser.parse_args()
    
    ckpt_dir = Path("ckpts")
    models = ["thyn", "baseline_a", "l1_thyn", "l1_baseline_a"]
    
    plt.figure(figsize=(15, 10))
    
    # 1. Plot Train Loss
    plt.subplot(2, 1, 1)
    for m in models:
        log_path = ckpt_dir / f"{args.dataset}_{m}" / "training.log"
        if not log_path.exists():
            print(f"Skipping {m}, no log found at {log_path}")
            continue
            
        eps, loss, _ = parse_log(log_path)
        if not eps:
            continue
            
        plt.plot(eps, loss, marker='o', label=f"{args.dataset}_{m}")
        
    plt.title("Train Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Plot Val AUPRC
    plt.subplot(2, 1, 2)
    for m in models:
        log_path = ckpt_dir / f"{args.dataset}_{m}" / "training.log"
        if not log_path.exists():
            continue
            
        eps, _, auprc = parse_log(log_path)
        if not eps:
            continue
            
        # Mark the best epoch
        best_ep = eps[np.argmax(auprc)]
        best_val = np.max(auprc)
        
        line, = plt.plot(eps, auprc, marker='o', label=f"{args.dataset}_{m}")
        plt.axvline(x=best_ep, color=line.get_color(), linestyle='--', alpha=0.5)
        plt.scatter([best_ep], [best_val], color=line.get_color(), s=100, zorder=5)
        
    plt.title("Validation AUPRC Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("AUPRC")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = f"experiments/audits/{args.dataset}_training_dynamics.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Diagnostic plot saved to {out_path}")

if __name__ == "__main__":
    main()
