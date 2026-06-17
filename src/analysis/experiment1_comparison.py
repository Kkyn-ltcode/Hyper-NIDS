"""
Experiment 1: Dual-Level End-to-End Detection Comparison.

Generates overlaid PR curves and LaTeX tables comparing HyperMamba
against PIDSMaker baselines (KAIROS, ThreaTrace, MAGIC) on DARPA TC E3.

Evaluates at TWO levels:
  1. Event-Level: Apples-to-apples comparison using our timestamp-matched events.
  2. Node-Level: Apples-to-oranges comparison where each system is evaluated
     against its own ground truth (PIDSMaker's narrow GT vs Our broad GT),
     reflecting the fundamental difference in their preprocessing pipelines.

Usage:
    python -m src.analysis.experiment1_comparison \
        --hypermamba_preds results/hypermamba/preds.pt \
        --kairos_dir /path/to/PIDSMaker/kairos/edge_losses/test/model_epoch_0 \
        --pidsmaker_gt /path/to/PIDSMaker/Ground_Truth/orthrus/E3-THEIA \
        --dataset theia \
        --out_dir results/experiment1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.metrics import (
    compute_node_level_metrics,
    compute_event_level_metrics,
    compute_pr_curve,
    compute_precision_at_fpr,
    format_metrics_table,
    format_latex_table,
)
from src.evaluation.node_aggregation import (
    load_entity_vocab,
    aggregate_to_nodes,
    load_pidsmaker_gt,
)
from src.baselines.parse_pidsmaker import (
    load_edge_losses_from_dir,
    aggregate_edge_losses_vectorized,
    match_pidsmaker_edges_to_our_labels,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Plotting
# ============================================================

def plot_pr_curves(
    model_curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out_path: str | Path,
    title: str,
):
    colors = {
        "HyperMamba": "#2563EB",
        "KAIROS": "#DC2626",
        "ThreaTrace": "#16A34A",
        "MAGIC": "#9333EA",
    }
    default_colors = plt.cm.Set2(np.linspace(0, 1, 8))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, (name, (prec, rec, auprc)) in enumerate(model_curves.items()):
        color = colors.get(name, default_colors[i % len(default_colors)])
        ax.plot(rec, prec, color=color, linewidth=2.0,
                label=f"{name} (AUPRC={auprc:.4f})")
    
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Main Comparison
# ============================================================

def run_comparison(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_root = Path(f"data/processed/darpa_tc_e3/{args.dataset}")
    shard_config = {"theia": [8, 9], "trace": [6]}
    test_shards = shard_config[args.dataset]
    
    # ── Load Vocab and GT ──
    id_to_uuid, _ = load_entity_vocab(data_root)
    
    # Load PIDSMaker GT
    if args.pidsmaker_gt:
        pidsmaker_gt_labels = load_pidsmaker_gt(args.pidsmaker_gt)
    else:
        logger.warning("No PIDSMaker GT provided. Node-level evaluation for PIDSMaker will be skipped.")
        pidsmaker_gt_labels = {}
        
    # Load Our GT (broad IoC-based)
    logger.info("Loading HyperMamba broad ground truth...")
    from src.data.ground_truth import load_ground_truth, build_attack_subject_uuids
    subjects_df = pd.read_parquet(data_root / "subjects.parquet")
    our_gt = load_ground_truth(args.dataset)
    our_attack_uuids = build_attack_subject_uuids(subjects_df, our_gt)
    our_gt_labels = {str(u): 1 for u in our_attack_uuids}
    # Note: we don't list all benign UUIDs, the metrics functions assume 0 if not in dict
    
    # Storage for results
    event_scores = {}
    event_labels = {}
    node_scores = {}
    
    # ── HyperMamba ──
    if args.hypermamba_preds:
        logger.info(f"Loading HyperMamba predictions from {args.hypermamba_preds}...")
        data = torch.load(args.hypermamba_preds, map_location="cpu", weights_only=False)
        logits = data["logits"]
        labels = data["labels"]
        entity_ids = data["entity_ids"]
        
        # Sigmoid for scores
        hm_scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        
        event_scores["HyperMamba"] = hm_scores
        event_labels["HyperMamba"] = labels
        
        # Aggregate to nodes
        hm_node_scores = aggregate_to_nodes(
            hm_scores, entity_ids, id_to_uuid, method=args.aggregation
        )
        # To evaluate all test nodes, we must pad the GT dict with 0s for nodes that appeared in test
        # but aren't in our_gt_labels.
        test_node_labels = {u: our_gt_labels.get(u, 0) for u in hm_node_scores.keys()}
        node_scores["HyperMamba"] = (hm_node_scores, test_node_labels)
    
    # ── PIDSMaker Baselines ──
    # Map NID to UUID for PIDSMaker
    if args.nid_to_uuid:
        from src.baselines.parse_pidsmaker import load_nid_to_uuid_from_pickle
        nid_to_uuid = load_nid_to_uuid_from_pickle(args.nid_to_uuid)
    else:
        logger.warning("No nid_to_uuid provided, PIDSMaker node aggregation will be skipped.")
        nid_to_uuid = None
        
    for name, losses_dir in [
        ("KAIROS", args.kairos_dir),
        ("ThreaTrace", args.threatrace_dir),
        ("MAGIC", args.magic_dir),
    ]:
        if not losses_dir:
            continue
            
        logger.info(f"\nProcessing {name}...")
        edge_df = load_edge_losses_from_dir(losses_dir)
        
        # 1. Event-Level Evaluation (timestamp matching)
        matched_scores, matched_labels = match_pidsmaker_edges_to_our_labels(
            edge_df, data_root, test_shards, label_type=args.label_type
        )
        if len(matched_scores) > 0:
            event_scores[name] = matched_scores
            event_labels[name] = matched_labels
            
        # 2. Node-Level Evaluation
        if nid_to_uuid and pidsmaker_gt_labels:
            logger.info(f"Aggregating {name} scores to nodes...")
            pm_node_scores = aggregate_edge_losses_vectorized(
                edge_df, nid_to_uuid, aggregation=args.aggregation, use_dst_node=True
            )
            # Evaluate against ALL mapped nodes. If a node is in pm_node_scores, check if it's in GT
            pm_test_labels = {u: pidsmaker_gt_labels.get(u, 0) for u in pm_node_scores.keys()}
            node_scores[name] = (pm_node_scores, pm_test_labels)

    # ── Compute Event-Level Metrics ──
    logger.info(f"\n{'='*60}\n  EVENT-LEVEL COMPARISON\n{'='*60}")
    event_metrics = {}
    event_curves = {}
    
    for name in event_scores.keys():
        metrics = compute_event_level_metrics(event_scores[name], event_labels[name])
        event_metrics[name] = metrics
        
        prec, rec, _ = precision_recall_curve(event_labels[name], event_scores[name])
        event_curves[name] = (prec, rec, metrics["auprc"])
        
    table_str = format_metrics_table(event_metrics, metrics_keys=["auprc", "best_f1", "precision_at_best", "recall_at_best"])
    logger.info(f"\n{table_str}")
    (out_dir / "event_level_table.txt").write_text(table_str)
    
    latex_str = format_latex_table(event_metrics, caption="Event-level detection on DARPA TC E3.")
    (out_dir / "event_level_table.tex").write_text(latex_str)
    plot_pr_curves(event_curves, out_dir / "pr_curves_event.png", "Event-Level PR Curves")
    
    # ── Compute Node-Level Metrics ──
    logger.info(f"\n{'='*60}\n  NODE-LEVEL COMPARISON (Model-Specific GT)\n{'='*60}")
    node_metrics = {}
    node_curves = {}
    
    for name, (n_scores, n_labels) in node_scores.items():
        metrics = compute_node_level_metrics(n_scores, n_labels)
        node_metrics[name] = metrics
        
        prec, rec, _ = compute_pr_curve(n_scores, n_labels)
        node_curves[name] = (prec, rec, metrics["auprc"])
        
    table_str = format_metrics_table(node_metrics, metrics_keys=["auprc", "best_f1", "precision_at_best", "recall_at_best"])
    logger.info(f"\n{table_str}")
    (out_dir / "node_level_table.txt").write_text(table_str)
    
    latex_str = format_latex_table(node_metrics, caption="Node-level detection on DARPA TC E3.")
    (out_dir / "node_level_table.tex").write_text(latex_str)
    plot_pr_curves(node_curves, out_dir / "pr_curves_node.png", "Node-Level PR Curves")

    # ── Save JSONs ──
    serializable = {}
    for name, metrics in event_metrics.items():
        serializable[f"{name}_Event"] = {k: float(v) for k, v in metrics.items()}
    for name, metrics in node_metrics.items():
        serializable[f"{name}_Node"] = {k: float(v) for k, v in metrics.items()}
        
    (out_dir / "metrics.json").write_text(json.dumps(serializable, indent=2))
    logger.info(f"\nComparison complete! Outputs saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Dual-Level Comparison")
    
    # HyperMamba
    parser.add_argument("--hypermamba_preds", type=str, default=None)
    
    # PIDSMaker
    parser.add_argument("--kairos_dir", type=str, default=None)
    parser.add_argument("--threatrace_dir", type=str, default=None)
    parser.add_argument("--magic_dir", type=str, default=None)
    parser.add_argument("--nid_to_uuid", type=str, default=None)
    parser.add_argument("--pidsmaker_gt", type=str, default=None,
                        help="Path to Ground_Truth/orthrus/E3-THEIA/")
    
    # Evaluation settings
    parser.add_argument("--dataset", type=str, default="theia")
    parser.add_argument("--label_type", type=str, default="crossprocess")
    parser.add_argument("--aggregation", type=str, default="max")
    
    # Output
    parser.add_argument("--out_dir", type=str, default="results/experiment1")
    
    args = parser.parse_args()
    
    from sklearn.metrics import precision_recall_curve
    run_comparison(args)


if __name__ == "__main__":
    main()
