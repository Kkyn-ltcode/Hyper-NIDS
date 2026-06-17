"""
Centralized evaluation metrics for Experiment 1: End-to-End Detection.

Handles:
  1. Event-level → node-level score aggregation
  2. Precision, Recall, F1 at various thresholds
  3. PR curves with AUC
  4. Comparison across multiple models

Design principle: all functions are stateless and take numpy arrays.
Model-specific loading is NOT done here — each model's parser produces
a standard {uuid: float} dict, and this module operates on those dicts.

Usage:
    from src.evaluation.metrics import (
        aggregate_event_scores_to_nodes,
        compute_node_level_metrics,
        compute_pr_curve,
    )
"""

import numpy as np
from collections import defaultdict
from typing import Optional

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    f1_score,
)


# ============================================================
# Event → Node Aggregation
# ============================================================

def aggregate_event_scores_to_nodes(
    event_scores: np.ndarray,
    event_uuids: np.ndarray,
    method: str = "max",
) -> dict[str, float]:
    """
    Aggregate per-event anomaly scores to per-node (UUID) scores.
    
    Args:
        event_scores: (N,) float array of per-event anomaly scores.
        event_uuids:  (N, 3) str array — columns are [sub_uuid, obj_uuid, obj2_uuid].
                      Each event contributes its score to ALL entities it touches.
        method:       Aggregation method. 'max' (default), 'mean', 'sum', or 'p99'.
    
    Returns:
        dict mapping UUID string → aggregated anomaly score.
    """
    assert event_scores.shape[0] == event_uuids.shape[0], \
        f"Score/UUID length mismatch: {event_scores.shape[0]} vs {event_uuids.shape[0]}"
    
    NIL_UUID = "00000000-0000-0000-0000-000000000000"
    
    # Collect all scores for each unique UUID
    uuid_scores: dict[str, list[float]] = defaultdict(list)
    
    for i in range(len(event_scores)):
        score = float(event_scores[i])
        for col_idx in range(event_uuids.shape[1]):
            uuid = str(event_uuids[i, col_idx])
            if uuid and uuid != NIL_UUID:
                uuid_scores[uuid].append(score)
    
    # Aggregate
    result = {}
    for uuid, scores in uuid_scores.items():
        scores_arr = np.array(scores)
        if method == "max":
            result[uuid] = float(scores_arr.max())
        elif method == "mean":
            result[uuid] = float(scores_arr.mean())
        elif method == "sum":
            result[uuid] = float(scores_arr.sum())
        elif method == "p99":
            result[uuid] = float(np.percentile(scores_arr, 99))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    return result


def aggregate_event_labels_to_nodes(
    event_labels: np.ndarray,
    event_uuids: np.ndarray,
) -> dict[str, int]:
    """
    Aggregate per-event binary labels to per-node labels.
    A node is malicious (1) if ANY event it participates in is labeled 1.
    
    Args:
        event_labels: (N,) int array, 0=benign, 1=attack.
        event_uuids:  (N, 3) str array — [sub_uuid, obj_uuid, obj2_uuid].
    
    Returns:
        dict mapping UUID string → binary label (0 or 1).
    """
    NIL_UUID = "00000000-0000-0000-0000-000000000000"
    node_labels: dict[str, int] = {}
    
    for i in range(len(event_labels)):
        label = int(event_labels[i])
        for col_idx in range(event_uuids.shape[1]):
            uuid = str(event_uuids[i, col_idx])
            if uuid and uuid != NIL_UUID:
                # OR logic: node is attack if any event is attack
                if uuid not in node_labels or label == 1:
                    node_labels[uuid] = max(node_labels.get(uuid, 0), label)
    
    return node_labels


# ============================================================
# Metrics Computation
# ============================================================

def align_scores_and_labels(
    node_scores: dict[str, float],
    node_labels: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align node scores and labels into parallel arrays.
    Only includes nodes present in BOTH dictionaries.
    
    Returns:
        (scores, labels) — aligned numpy arrays.
    """
    common_uuids = sorted(set(node_scores.keys()) & set(node_labels.keys()))
    
    if not common_uuids:
        return np.array([]), np.array([])
    
    scores = np.array([node_scores[u] for u in common_uuids])
    labels = np.array([node_labels[u] for u in common_uuids])
    
    return scores, labels


def compute_node_level_metrics(
    node_scores: dict[str, float],
    node_labels: dict[str, int],
) -> dict[str, float]:
    """
    Compute standard detection metrics at the node level.
    
    Args:
        node_scores: {uuid: anomaly_score}
        node_labels: {uuid: 0 or 1}
    
    Returns:
        Dict with keys: auprc, auroc, best_f1, best_threshold,
        precision_at_best, recall_at_best, n_nodes, n_attack_nodes.
    """
    scores, labels = align_scores_and_labels(node_scores, node_labels)
    
    if len(scores) == 0 or labels.sum() == 0:
        return {
            "auprc": 0.0, "auroc": 0.0, "best_f1": 0.0,
            "best_threshold": 0.0, "precision_at_best": 0.0,
            "recall_at_best": 0.0, "n_nodes": len(scores),
            "n_attack_nodes": 0,
        }
    
    metrics = {}
    metrics["n_nodes"] = len(scores)
    metrics["n_attack_nodes"] = int(labels.sum())
    
    # AUPRC
    try:
        metrics["auprc"] = float(average_precision_score(labels, scores))
    except ValueError:
        metrics["auprc"] = 0.0
    
    # AUROC
    try:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        metrics["auroc"] = 0.0
    
    # Best F1 from PR curve
    try:
        prec, rec, thr = precision_recall_curve(labels, scores)
        f1_all = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        best_idx = np.argmax(f1_all)
        metrics["best_f1"] = float(f1_all[best_idx])
        metrics["best_threshold"] = float(thr[best_idx])
        metrics["precision_at_best"] = float(prec[best_idx])
        metrics["recall_at_best"] = float(rec[best_idx])
    except ValueError:
        metrics["best_f1"] = 0.0
        metrics["best_threshold"] = 0.0
        metrics["precision_at_best"] = 0.0
        metrics["recall_at_best"] = 0.0
    
    return metrics


def compute_event_level_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Compute standard detection metrics at the event level.
    
    Args:
        scores: (N,) float array of anomaly scores
        labels: (N,) int array of binary labels (0 or 1)
        
    Returns:
        Dict with keys: auprc, auroc, best_f1, best_threshold,
        precision_at_best, recall_at_best, n_events, n_attack_events.
    """
    if len(scores) == 0 or labels.sum() == 0:
        return {
            "auprc": 0.0, "auroc": 0.0, "best_f1": 0.0,
            "best_threshold": 0.0, "precision_at_best": 0.0,
            "recall_at_best": 0.0, "n_events": len(scores),
            "n_attack_events": 0,
        }
    
    metrics = {}
    metrics["n_events"] = len(scores)
    metrics["n_attack_events"] = int(labels.sum())
    
    # AUPRC
    try:
        metrics["auprc"] = float(average_precision_score(labels, scores))
    except ValueError:
        metrics["auprc"] = 0.0
    
    # AUROC
    try:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
    except ValueError:
        metrics["auroc"] = 0.0
    
    # Best F1 from PR curve
    try:
        prec, rec, thr = precision_recall_curve(labels, scores)
        f1_all = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
        best_idx = np.argmax(f1_all)
        metrics["best_f1"] = float(f1_all[best_idx])
        metrics["best_threshold"] = float(thr[best_idx])
        metrics["precision_at_best"] = float(prec[best_idx])
        metrics["recall_at_best"] = float(rec[best_idx])
    except ValueError:
        metrics["best_f1"] = 0.0
        metrics["best_threshold"] = 0.0
        metrics["precision_at_best"] = 0.0
        metrics["recall_at_best"] = 0.0
    
    return metrics


def compute_pr_curve(
    node_scores: dict[str, float],
    node_labels: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute full PR curve data for plotting.
    
    Returns:
        (precision, recall, thresholds) — from sklearn.
    """
    scores, labels = align_scores_and_labels(node_scores, node_labels)
    
    if len(scores) == 0 or labels.sum() == 0:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0])
    
    return precision_recall_curve(labels, scores)


def compute_precision_at_fpr(
    node_scores: dict[str, float],
    node_labels: dict[str, int],
    target_fpr: float = 0.01,
) -> dict[str, float]:
    """
    Compute Precision at a target False Positive Rate (FPR).
    
    Useful for realistic evaluation: in deployment, FPR must be tiny
    to avoid alert fatigue.
    
    Args:
        node_scores: {uuid: anomaly_score}
        node_labels: {uuid: 0 or 1}
        target_fpr:  Target FPR (e.g. 0.01 = 1%)
    
    Returns:
        Dict with precision, recall, threshold, actual_fpr at the target FPR.
    """
    scores, labels = align_scores_and_labels(node_scores, node_labels)
    
    if len(scores) == 0 or labels.sum() == 0:
        return {"precision": 0.0, "recall": 0.0, "threshold": 0.0, "actual_fpr": 0.0}
    
    # Sort by score descending
    sorted_idx = np.argsort(-scores)
    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]
    
    n_total = len(labels)
    n_pos = int(labels.sum())
    n_neg = n_total - n_pos
    
    # Walk through thresholds
    best_result = {"precision": 0.0, "recall": 0.0, "threshold": 0.0, "actual_fpr": 0.0}
    
    tp = 0
    fp = 0
    for i in range(n_total):
        if sorted_labels[i] == 1:
            tp += 1
        else:
            fp += 1
        
        current_fpr = fp / max(n_neg, 1)
        
        if current_fpr <= target_fpr:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / n_pos if n_pos > 0 else 0.0
            best_result = {
                "precision": precision,
                "recall": recall,
                "threshold": float(sorted_scores[i]),
                "actual_fpr": current_fpr,
            }
        else:
            break
    
    return best_result


# ============================================================
# Formatting & Comparison Utilities
# ============================================================

def format_metrics_table(
    model_metrics: dict[str, dict[str, float]],
    metrics_keys: Optional[list[str]] = None,
) -> str:
    """
    Format a comparison table as a string.
    
    Args:
        model_metrics: {model_name: {metric_name: value}}
        metrics_keys:  Which metrics to include. Default: all standard ones.
    
    Returns:
        Formatted string table.
    """
    if metrics_keys is None:
        metrics_keys = ["auprc", "auroc", "best_f1", "precision_at_best", "recall_at_best"]
    
    # Header
    header = f"{'Model':<20s}" + "".join(f"{k:>15s}" for k in metrics_keys)
    lines = [header, "-" * len(header)]
    
    for model_name, metrics in model_metrics.items():
        row = f"{model_name:<20s}"
        for k in metrics_keys:
            val = metrics.get(k, 0.0)
            row += f"{val:>15.4f}"
        lines.append(row)
    
    return "\n".join(lines)


def format_latex_table(
    model_metrics: dict[str, dict[str, float]],
    metrics_keys: Optional[list[str]] = None,
    caption: str = "Node-level detection performance on DARPA TC E3.",
    label: str = "tab:experiment1",
) -> str:
    """
    Generate a LaTeX table for the paper.
    
    Args:
        model_metrics: {model_name: {metric_name: value}}
        metrics_keys:  Which metrics to include.
        caption:       Table caption.
        label:         LaTeX label.
    
    Returns:
        LaTeX table string.
    """
    if metrics_keys is None:
        metrics_keys = ["auprc", "auroc", "best_f1", "precision_at_best", "recall_at_best"]
    
    # Pretty names for column headers
    pretty_names = {
        "auprc": "AUPRC",
        "auroc": "AUROC",
        "best_f1": "Best F1",
        "precision_at_best": "Precision",
        "recall_at_best": "Recall",
        "n_nodes": "\\# Nodes",
        "n_attack_nodes": "\\# Attack",
    }
    
    col_spec = "l" + "c" * len(metrics_keys)
    header_row = " & ".join([pretty_names.get(k, k) for k in metrics_keys])
    
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f"Model & {header_row} \\\\",
        r"\midrule",
    ]
    
    # Find best value per column for bolding
    best_vals = {}
    for k in metrics_keys:
        vals = [m.get(k, 0.0) for m in model_metrics.values()]
        best_vals[k] = max(vals) if vals else 0.0
    
    for model_name, metrics in model_metrics.items():
        cells = []
        for k in metrics_keys:
            val = metrics.get(k, 0.0)
            formatted = f"{val:.4f}"
            if val == best_vals[k] and val > 0:
                formatted = f"\\textbf{{{formatted}}}"
            cells.append(formatted)
        lines.append(f"{model_name} & {' & '.join(cells)} \\\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)
