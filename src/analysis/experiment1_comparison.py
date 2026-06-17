"""
Experiment 1: End-to-End Detection Comparison.

Generates overlaid PR curves and LaTeX tables comparing HyperMamba
against PIDSMaker baselines (KAIROS, ThreaTrace, MAGIC) on DARPA TC E3.

All models are evaluated at the NODE level using the same ground truth
and the same chronological test window (April 10+ events).

Usage:
    python -m src.analysis.experiment1_comparison \\
        --hypermamba_preds results/hypermamba/preds.pt \\
        --kairos_dir /path/to/PIDSMaker/kairos/edge_losses/test/model_epoch_0 \\
        --threatrace_dir /path/to/PIDSMaker/threatrace/edge_losses/test/model_epoch_0 \\
        --magic_dir /path/to/PIDSMaker/magic/edge_losses/test/model_epoch_0 \\
        --nid_to_uuid /path/to/nid_to_uuid.pkl \\
        --dataset theia \\
        --out_dir results/experiment1
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.evaluation.metrics import (
    aggregate_event_scores_to_nodes,
    aggregate_event_labels_to_nodes,
    compute_node_level_metrics,
    compute_pr_curve,
    compute_precision_at_fpr,
    format_metrics_table,
    format_latex_table,
)
from src.baselines.parse_pidsmaker import load_pidsmaker_scores


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# HyperMamba Predictions Loader
# ============================================================

def load_hypermamba_node_scores(
    preds_path: str | Path,
    aggregation: str = "max",
) -> tuple[dict[str, float], dict[str, int]]:
    """
    Load HyperMamba predictions (preds.pt) and aggregate to node-level.
    
    The preds.pt file is expected to contain:
        logits:      (N,) float — raw model output logits
        labels:      (N,) int — ground truth labels (0 or 1)
        entity_ids:  (N, 3) int — [subj_int_id, obj_int_id, obj2_int_id]
        event_uuids: (N, 3) str — [subj_uuid, obj_uuid, obj2_uuid]
        timestamps:  (N,) float — relative timestamps
    
    Returns:
        (node_scores, node_labels) — both as {uuid: value} dicts.
    """
    preds_path = Path(preds_path)
    if not preds_path.exists():
        raise FileNotFoundError(f"HyperMamba predictions not found: {preds_path}")
    
    logger.info(f"Loading HyperMamba predictions from {preds_path}...")
    data = torch.load(preds_path, map_location="cpu", weights_only=False)
    
    logits = data["logits"]
    labels = data["labels"]
    event_uuids = data["event_uuids"]
    
    # Convert logits to anomaly scores (sigmoid probabilities)
    if isinstance(logits, torch.Tensor):
        scores = torch.sigmoid(logits).numpy()
    else:
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    if isinstance(event_uuids, torch.Tensor):
        event_uuids = event_uuids.numpy()
    
    logger.info(f"  {len(scores):,} events, "
                f"{int((labels == 1).sum()):,} attack events")
    
    # Aggregate to node level
    node_scores = aggregate_event_scores_to_nodes(scores, event_uuids, method=aggregation)
    node_labels = aggregate_event_labels_to_nodes(labels, event_uuids)
    
    n_atk = sum(1 for v in node_labels.values() if v == 1)
    logger.info(f"  Node-level: {len(node_scores):,} nodes, "
                f"{n_atk:,} attack nodes")
    
    return node_scores, node_labels


# ============================================================
# Ground Truth from Our Labeled Data
# ============================================================

def load_ground_truth_from_dataset(
    data_root: str | Path,
    shard_ids: list[int],
    label_type: str = "broad",
) -> dict[str, int]:
    """
    Load ground truth node labels from our labeled parquet files.
    
    This ensures all models (HyperMamba and PIDSMaker baselines) are
    evaluated against exactly the same ground truth.
    """
    import pandas as pd
    
    data_root = Path(data_root)
    labeled_dir = data_root / "labeled"
    label_col = f"label_{label_type}"
    
    node_labels: dict[str, int] = {}
    
    for sid in shard_ids:
        df = pd.read_parquet(
            labeled_dir / f"labeled_shard{sid}.parquet",
            columns=["subject_uuid", "predicate_object_uuid",
                     "predicate_object2_uuid", label_col],
        )
        
        labels = df[label_col].values
        NIL_UUID = "00000000-0000-0000-0000-000000000000"
        
        for uuid_col in ["subject_uuid", "predicate_object_uuid", "predicate_object2_uuid"]:
            uuids = df[uuid_col].values
            for i in range(len(labels)):
                uuid = str(uuids[i])
                if uuid and uuid != NIL_UUID:
                    label = int(labels[i])
                    node_labels[uuid] = max(node_labels.get(uuid, 0), label)
        
        del df
    
    n_atk = sum(1 for v in node_labels.values() if v == 1)
    logger.info(f"Ground truth from shards {shard_ids}: "
                f"{len(node_labels):,} nodes, {n_atk:,} attack")
    return node_labels


# ============================================================
# Plotting
# ============================================================

def plot_pr_curves(
    model_curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out_path: str | Path,
    title: str = "Precision-Recall Curves — Node-Level Detection",
):
    """
    Plot overlaid PR curves for multiple models.
    
    Args:
        model_curves: {model_name: (precision, recall, auprc)}
        out_path:     Path to save the figure.
        title:        Plot title.
    """
    # Color palette — distinguishable, publication-quality
    colors = {
        "HyperMamba": "#2563EB",   # blue
        "KAIROS": "#DC2626",       # red
        "ThreaTrace": "#16A34A",   # green
        "MAGIC": "#9333EA",        # purple
    }
    default_colors = plt.cm.Set2(np.linspace(0, 1, 8))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, (name, (prec, rec, auprc)) in enumerate(model_curves.items()):
        color = colors.get(name, default_colors[i % len(default_colors)])
        ax.plot(rec, prec, color=color, linewidth=2.0,
                label=f"{name} (AUPRC={auprc:.4f})")
    
    # Formatting
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
    logger.info(f"PR curve saved to {out_path}")


def plot_pr_curves_log_scale(
    model_curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    out_path: str | Path,
    title: str = "PR Curves (Log-Scale Recall)",
):
    """
    PR curve with log-scale x-axis — better for visualizing
    performance at low recall (early detection regime).
    """
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
        # Filter out zero recall for log scale
        mask = rec > 0
        ax.plot(rec[mask], prec[mask], color=color, linewidth=2.0,
                label=f"{name} (AUPRC={auprc:.4f})")
    
    ax.set_xscale("log")
    ax.set_xlabel("Recall (log scale)", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim([-0.02, 1.02])
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"PR curve (log scale) saved to {out_path}")


# ============================================================
# Main Comparison
# ============================================================

def run_comparison(args):
    """Run the full Experiment 1 comparison."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ── Load ground truth ──
    # We use our own labeled data as ground truth for all models.
    # Test shards for theia are [8, 9], for trace [6].
    shard_config = {
        "theia": {"test": [8, 9]},
        "trace": {"test": [6]},
    }
    data_root = Path(f"data/processed/darpa_tc_e3/{args.dataset}")
    test_shards = shard_config[args.dataset]["test"]
    
    gt_labels = load_ground_truth_from_dataset(
        data_root, test_shards, label_type=args.label_type)
    
    # ── Load model scores ──
    all_model_scores: dict[str, dict[str, float]] = {}
    
    # HyperMamba
    if args.hypermamba_preds:
        hm_scores, hm_labels = load_hypermamba_node_scores(
            args.hypermamba_preds, aggregation=args.aggregation)
        all_model_scores["HyperMamba"] = hm_scores
        # Use HyperMamba's own labels as ground truth if no separate GT
        if not gt_labels:
            gt_labels = hm_labels
    
    # PIDSMaker baselines
    nid_to_uuid = args.nid_to_uuid
    
    for name, losses_dir in [
        ("KAIROS", args.kairos_dir),
        ("ThreaTrace", args.threatrace_dir),
        ("MAGIC", args.magic_dir),
    ]:
        if losses_dir:
            try:
                scores = load_pidsmaker_scores(
                    edge_losses_dir=losses_dir,
                    nid_to_uuid=nid_to_uuid,
                    aggregation=args.aggregation,
                    use_dst_node=True,
                )
                all_model_scores[name] = scores
                logger.info(f"  {name}: {len(scores):,} node scores loaded")
            except Exception as e:
                logger.error(f"  Failed to load {name}: {e}")
    
    if not all_model_scores:
        logger.error("No model scores loaded. Exiting.")
        sys.exit(1)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  EXPERIMENT 1: NODE-LEVEL DETECTION COMPARISON")
    logger.info(f"  Dataset: {args.dataset.upper()}")
    logger.info(f"  Label type: {args.label_type}")
    logger.info(f"  Aggregation: {args.aggregation}")
    logger.info(f"  Models: {list(all_model_scores.keys())}")
    logger.info(f"{'='*60}\n")
    
    # ── Compute metrics for each model ──
    all_metrics: dict[str, dict[str, float]] = {}
    model_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    
    for model_name, node_scores in all_model_scores.items():
        logger.info(f"\n── {model_name} ──")
        
        metrics = compute_node_level_metrics(node_scores, gt_labels)
        all_metrics[model_name] = metrics
        
        logger.info(f"  AUPRC:     {metrics['auprc']:.4f}")
        logger.info(f"  AUROC:     {metrics['auroc']:.4f}")
        logger.info(f"  Best F1:   {metrics['best_f1']:.4f}")
        logger.info(f"  Precision: {metrics['precision_at_best']:.4f}")
        logger.info(f"  Recall:    {metrics['recall_at_best']:.4f}")
        logger.info(f"  Nodes:     {metrics['n_nodes']:,} "
                     f"({metrics['n_attack_nodes']:,} attack)")
        
        # Precision@1% FPR
        p_at_fpr = compute_precision_at_fpr(node_scores, gt_labels, target_fpr=0.01)
        metrics["precision_at_1pct_fpr"] = p_at_fpr["precision"]
        metrics["recall_at_1pct_fpr"] = p_at_fpr["recall"]
        logger.info(f"  Precision@1%FPR: {p_at_fpr['precision']:.4f} "
                     f"(Recall: {p_at_fpr['recall']:.4f})")
        
        # PR curve data
        prec, rec, _ = compute_pr_curve(node_scores, gt_labels)
        model_curves[model_name] = (prec, rec, metrics["auprc"])
    
    # ── Generate outputs ──
    
    # 1. Text table
    table_str = format_metrics_table(all_metrics)
    logger.info(f"\n{table_str}")
    table_path = out_dir / "comparison_table.txt"
    table_path.write_text(table_str)
    
    # 2. LaTeX table
    latex_str = format_latex_table(
        all_metrics,
        caption=f"Node-level detection on DARPA TC E3 ({args.dataset.upper()}).",
        label=f"tab:exp1_{args.dataset}",
    )
    latex_path = out_dir / "comparison_table.tex"
    latex_path.write_text(latex_str)
    logger.info(f"LaTeX table saved to {latex_path}")
    
    # 3. PR curves
    plot_pr_curves(model_curves, out_dir / "pr_curves.png")
    plot_pr_curves_log_scale(model_curves, out_dir / "pr_curves_log.png")
    
    # 4. Save raw metrics as JSON
    import json
    metrics_path = out_dir / "metrics.json"
    # Convert numpy types to Python types for JSON serialization
    serializable = {}
    for model_name, metrics in all_metrics.items():
        serializable[model_name] = {
            k: float(v) if isinstance(v, (np.floating, float)) else int(v)
            for k, v in metrics.items()
        }
    metrics_path.write_text(json.dumps(serializable, indent=2))
    logger.info(f"Metrics JSON saved to {metrics_path}")
    
    # 5. Save raw PR curve data for reproducibility
    curves_data = {}
    for name, (prec, rec, auprc) in model_curves.items():
        curves_data[name] = {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "auprc": auprc,
        }
    curves_path = out_dir / "pr_curve_data.json"
    curves_path.write_text(json.dumps(curves_data))
    logger.info(f"PR curve data saved to {curves_path}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  COMPARISON COMPLETE")
    logger.info(f"  Results saved to: {out_dir}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: End-to-End Detection Comparison"
    )
    
    # Model predictions
    parser.add_argument("--hypermamba_preds", type=str, default=None,
                        help="Path to HyperMamba preds.pt file")
    parser.add_argument("--kairos_dir", type=str, default=None,
                        help="Path to KAIROS edge_losses/test/model_epoch_N/")
    parser.add_argument("--threatrace_dir", type=str, default=None,
                        help="Path to ThreaTrace edge_losses/test/model_epoch_N/")
    parser.add_argument("--magic_dir", type=str, default=None,
                        help="Path to MAGIC edge_losses/test/model_epoch_N/")
    
    # ID mapping
    parser.add_argument("--nid_to_uuid", type=str, default=None,
                        help="Path to NID→UUID mapping (pickle or CSV)")
    
    # Evaluation settings
    parser.add_argument("--dataset", type=str, default="theia",
                        choices=["theia", "trace"],
                        help="Dataset name for ground truth loading")
    parser.add_argument("--label_type", type=str, default="broad",
                        choices=["broad", "crossprocess", "l1", "narrow", "ioc"],
                        help="Label type for ground truth")
    parser.add_argument("--aggregation", type=str, default="max",
                        choices=["max", "mean", "sum", "p90", "p99"],
                        help="Node-level score aggregation method")
    
    # Output
    parser.add_argument("--out_dir", type=str, default="results/experiment1",
                        help="Output directory for plots and tables")
    
    args = parser.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()
