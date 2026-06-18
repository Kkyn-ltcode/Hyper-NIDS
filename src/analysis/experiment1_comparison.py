import argparse
import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from src.evaluation.metrics import aggregate_node_scores, evaluate_node_level

def load_ground_truth(gt_dir):
    """
    Load ground truth UUIDs from PIDSMaker ground truth CSVs.
    """
    gt_files = glob.glob(os.path.join(gt_dir, "**", "*.csv"), recursive=True)
    malicious_uuids = set()
    for f in gt_files:
        try:
            df = pd.read_csv(f, header=None)
            # Assuming UUID is the first column
            uuids = df.iloc[:, 0].astype(str).values
            malicious_uuids.update(uuids)
        except Exception as e:
            print(f"Warning: could not read {f}: {e}")
    print(f"Loaded {len(malicious_uuids)} malicious UUIDs from {len(gt_files)} files.")
    return malicious_uuids

def load_hypermamba_scores(preds_path, vocab_path):
    """
    Load HyperMamba preds.pt and map integer entity IDs to UUID strings.
    """
    print(f"Loading HyperMamba predictions from {preds_path}")
    preds = torch.load(preds_path, map_location="cpu", weights_only=False)
    logits = preds["logits"]
    entity_ids = preds["entity_ids"]
    
    print(f"Loading entity vocab from {vocab_path}")
    vocab = np.load(vocab_path, allow_pickle=True)
    uuid_mapping = {i: str(u) for i, u in enumerate(vocab["uuids"])}
    
    print("Aggregating HyperMamba node scores...")
    node_scores = aggregate_node_scores(entity_ids, logits, uuid_mapping)
    return node_scores

def plot_pr_curves(results, out_file):
    plt.figure(figsize=(10, 8))
    
    colors = {
        "HyperMamba": "red",
        "KAIROS": "blue",
        "ThreaTrace": "green",
        "MAGIC": "purple",
        "Flash": "orange"
    }
    
    for system, data in results.items():
        scores = data["scores"]
        y_true = data["y_true"]
        if len(y_true) > 0 and sum(y_true) > 0:
            prec, rec, _ = precision_recall_curve(y_true, scores)
            color = colors.get(system, "gray")
            plt.plot(rec, prec, label=f'{system} (AUPRC={data["metrics"]["auprc"]:.3f})', color=color, linewidth=2)
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('End-to-End Detection PR Curves (THEIA E3)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    print(f"Saved PR curves to {out_file}")

def print_latex_table(results):
    print("\n" + "="*80)
    print("LATEX TABLE FOR EXPERIMENT 1")
    print("="*80)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{l c c c c}")
    print("\\hline")
    print("\\textbf{System} & \\textbf{AUPRC} & \\textbf{Best F1} & \\textbf{Prec@0.1\\% FPR} & \\textbf{Prec@0.01\\% FPR} \\\\")
    print("\\hline")
    
    for system, data in results.items():
        m = data["metrics"]
        print(f"{system} & {m['auprc']:.3f} & {m['best_f1']:.3f} & {m['prec_at_0.1%_fpr']:.3f} & {m['prec_at_0.01%_fpr']:.3f} \\\\")
        
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{End-to-End Node-Level Detection Performance on THEIA E3}")
    print("\\label{tab:exp1}")
    print("\\end{table}")
    print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: End-to-End Comparison")
    parser.add_argument("--hm_preds", type=str, required=True, help="Path to HyperMamba preds.pt")
    parser.add_argument("--vocab", type=str, required=True, help="Path to entity_vocab.npz")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory with Ground Truth CSVs")
    parser.add_argument("--baseline_scores", type=str, nargs='+', help="Paths to baseline parsed scores pkl (format: Name:path.pkl)")
    parser.add_argument("--out_plot", type=str, default="experiment1_pr_curve.png", help="Output plot path")
    args = parser.parse_args()
    
    gt_uuids = load_ground_truth(args.gt_dir)
    
    all_results = {}
    
    # Load HyperMamba
    hm_scores = load_hypermamba_scores(args.hm_preds, args.vocab)
    hm_metrics = evaluate_node_level(hm_scores, gt_uuids)
    all_results["HyperMamba"] = {
        "node_scores": hm_scores,
        "metrics": hm_metrics,
        "scores": np.array([hm_scores[u] for u in hm_scores]),
        "y_true": np.array([1 if u in gt_uuids else 0 for u in hm_scores])
    }
    
    # Load Baselines
    if args.baseline_scores:
        for baseline_arg in args.baseline_scores:
            if ":" not in baseline_arg:
                print(f"Skipping invalid baseline argument: {baseline_arg}")
                continue
            name, path = baseline_arg.split(":", 1)
            if not os.path.exists(path):
                print(f"Baseline file not found: {path}")
                continue
                
            print(f"Loading {name} scores from {path}")
            with open(path, "rb") as f:
                baseline_scores = pickle.load(f)
                
            b_metrics = evaluate_node_level(baseline_scores, gt_uuids)
            all_results[name] = {
                "node_scores": baseline_scores,
                "metrics": b_metrics,
                "scores": np.array([baseline_scores[u] for u in baseline_scores]),
                "y_true": np.array([1 if u in gt_uuids else 0 for u in baseline_scores])
            }
            
    plot_pr_curves(all_results, args.out_plot)
    print_latex_table(all_results)
