import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def run_analysis(preds_file, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading predictions from {preds_file}...")
    data = torch.load(preds_file)
    
    logits = data["logits"]
    labels = data["labels"]
    entity_ids = data["entity_ids"]  # (N, 3)
    timestamps = data["timestamps"]
    
    # Determine best threshold for F1
    prec, rec, thres = precision_recall_curve(labels, logits)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    best_idx = np.argmax(f1)
    best_thresh = thres[best_idx] if best_idx < len(thres) else 0.0
    print(f"Best Threshold: {best_thresh:.4f} (F1: {f1[best_idx]:.4f})")
    
    preds = (logits >= best_thresh).astype(int)
    
    # Tracking
    # For each entity, keep track of the index of its last attack event
    last_attack_idx = {}
    # For each entity, keep track of how many benign events it has participated in since its last attack event
    benign_count_since_attack = defaultdict(int)
    
    results = []
    
    for i in range(len(labels)):
        is_attack = (labels[i] == 1)
        ents = entity_ids[i]
        valid_ents = [e for e in ents if e >= 0]
        
        if is_attack:
            # Find the minimum benign gap among participating entities that have a previous attack
            gaps = []
            time_gaps = []
            for e in valid_ents:
                if e in last_attack_idx:
                    gaps.append(benign_count_since_attack[e])
                    time_gaps.append(timestamps[i] - timestamps[last_attack_idx[e]])
            
            if gaps:
                min_gap = min(gaps)
                min_time_gap = min(time_gaps)
                results.append({
                    "event_idx": i,
                    "benign_gap": min_gap,
                    "time_gap": min_time_gap,
                    "pred_correct": (preds[i] == 1)
                })
            
            # Reset tracking for these entities
            for e in valid_ents:
                last_attack_idx[e] = i
                benign_count_since_attack[e] = 0
        else:
            # Benign event: increment counter for any entity that has seen an attack
            for e in valid_ents:
                if e in last_attack_idx:
                    benign_count_since_attack[e] += 1
                    
    df = pd.DataFrame(results)
    print(f"Analyzed {len(df)} attack events with a history.")
    
    # 1. Dilution Analysis (Benign Gap)
    bins = [-1, 0, 10, 50, 200, 1000, np.inf]
    labels_bins = ['0', '1-10', '11-50', '51-200', '201-1000', '>1000']
    df['gap_bucket'] = pd.cut(df['benign_gap'], bins=bins, labels=labels_bins)
    
    recall_by_gap = df.groupby('gap_bucket')['pred_correct'].mean()
    count_by_gap = df.groupby('gap_bucket')['pred_correct'].count()
    
    print("\nRecall by Benign Event Gap:")
    for b in labels_bins:
        if b in recall_by_gap:
            print(f"  Gap {b:>10}: Recall {recall_by_gap[b]:.4f} (n={count_by_gap[b]})")
            
    # 2. Temporal Analysis (Time Gap)
    t_bins = [-1, 1, 60, 600, 3600, np.inf]
    t_labels = ['<1s', '1s-1m', '1m-10m', '10m-1h', '>1h']
    df['time_bucket'] = pd.cut(df['time_gap'], bins=t_bins, labels=t_labels)
    
    recall_by_time = df.groupby('time_bucket')['pred_correct'].mean()
    count_by_time = df.groupby('time_bucket')['pred_correct'].count()
    
    print("\nRecall by Time Gap:")
    for b in t_labels:
        if b in recall_by_time:
            print(f"  Gap {b:>10}: Recall {recall_by_time[b]:.4f} (n={count_by_time[b]})")
            
    df.to_csv(out_dir / "dilution_results.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True, help="Path to saved predictions dict (.pt)")
    parser.add_argument("--out_dir", default="results/analysis", help="Output directory")
    args = parser.parse_args()
    
    run_analysis(args.preds, args.out_dir)
