"""
KAIROS evaluation: reconstruction-based anomaly detection.

Two evaluation modes:
  1. Window-level: flag 15-min windows as anomalous (native KAIROS)
  2. Event-level: compute AUPRC/AUROC for THyN comparison

Usage:
    python -m baselines.kairos.evaluate --dataset theia
    python -m baselines.kairos.evaluate --dataset theia --split test
"""

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    LastNeighborLoader, IdentityMessage, LastAggregator
)
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_fscore_support,
)

from baselines.kairos.config import (
    DATASET_CONFIGS, GRAPHS_DIR, MODELS_DIR,
    NODE_EMBEDDING_DIM, NODE_STATE_DIM, NEIGHBOR_SIZE,
    EDGE_DIM, TIME_DIM, BATCH_SIZE, TIME_WINDOW_SIZE,
    build_rel2id,
)
from baselines.kairos.model import GraphAttentionEmbedding, LinkPredictor


def tensor_find(t, x):
    idx = np.argwhere(t.cpu().numpy() == x)
    return idx[0][0] + 1

def sequential_batches(data, batch_size):
    num_events = data.num_events
    for start in range(0, num_events, batch_size):
        end = min(start + batch_size, num_events)
        yield (
            data.src[start:end],
            data.dst[start:end],
            data.t[start:end],
            data.msg[start:end],
        )

@torch.no_grad()
def compute_reconstruction_losses(data, memory, gnn, link_pred,
                                  neighbor_loader, assoc, device,
                                  node_emb_dim):
    memory.eval()
    gnn.eval()
    link_pred.eval()
    memory.reset_state()
    neighbor_loader.reset_state()

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    all_timestamps = []
    use_amp = (device.type == 'cuda')

    for src, pos_dst, t, msg in sequential_batches(data, BATCH_SIZE):
        src = src.to(device)
        pos_dst = pos_dst.to(device)
        t = t.to(device)
        msg = msg.to(device)

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # # Mixed‑precision forward pass
        # with torch.cuda.amp.autocast(enabled=use_amp):
        #     z, last_update = memory(n_id)
        #     z = gnn(z, last_update, edge_index,
        #             data.t[e_id].to(device), data.msg[e_id].to(device))
        #     pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        # # Compute loss in float32 for numerical consistency
        # y_true = torch.argmax(msg[:, node_emb_dim:-node_emb_dim], dim=1)
        # loss = criterion(pos_out.float(), y_true)

                # No autocast – full precision
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index,
                data.t[e_id].to(device), data.msg[e_id].to(device))
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        # Ensure float32 for stable cross-entropy
        y_true = torch.argmax(msg[:, node_emb_dim:-node_emb_dim], dim=1)
        loss = criterion(pos_out.float(), y_true)   # keep loss in float32

        all_losses.append(loss.cpu().numpy())
        all_timestamps.append(t.cpu().numpy())

        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

    return np.concatenate(all_losses), np.concatenate(all_timestamps)


def window_level_evaluation(losses, timestamps, labels,
                            window_size=TIME_WINDOW_SIZE):
    """Aggregate per-event losses into time windows, evaluate.

    Each window's anomaly score is the mean reconstruction loss.
    """
    t_min = timestamps.min()
    t_max = timestamps.max()
    n_windows = int((t_max - t_min) / window_size) + 1

    window_scores = np.zeros(n_windows)
    window_labels = np.zeros(n_windows)
    window_counts = np.zeros(n_windows)

    for i in range(len(losses)):
        w_idx = int((timestamps[i] - t_min) / window_size)
        w_idx = min(w_idx, n_windows - 1)
        window_scores[w_idx] += losses[i]
        window_counts[w_idx] += 1
        if labels[i] == 1:
            window_labels[w_idx] = 1

    # Average loss per window
    valid = window_counts > 0
    window_scores[valid] /= window_counts[valid]

    # Only evaluate windows with events
    ws = window_scores[valid]
    wl = window_labels[valid]

    n_attack_windows = int(wl.sum())
    n_total_windows = len(wl)

    print(f"\n  Window-Level Evaluation ({window_size/6e10:.0f}-min windows)")
    print(f"    Total windows:  {n_total_windows}")
    print(f"    Attack windows: {n_attack_windows}")

    if n_attack_windows == 0 or n_attack_windows == n_total_windows:
        print(f"    SKIPPED: {'all' if n_attack_windows == n_total_windows else 'no'} "
              f"windows are attack — cannot compute meaningful metrics")
        return {}

    auprc = average_precision_score(wl, ws)
    auroc = roc_auc_score(wl, ws)

    # Threshold-based: use validation set max as threshold
    # (in actual KAIROS, this is the beta parameter)
    # For now, use Youden's J statistic
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(wl, ws)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    pred = (ws > best_threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        wl, pred, average="binary", zero_division=0)

    print(f"    AUPRC:     {auprc:.4f}")
    print(f"    AUROC:     {auroc:.4f}")
    print(f"    Precision: {p:.4f}")
    print(f"    Recall:    {r:.4f}")
    print(f"    F1:        {f1:.4f}")

    return {
        "window_auprc": auprc,
        "window_auroc": auroc,
        "window_precision": p,
        "window_recall": r,
        "window_f1": f1,
        "n_windows": n_total_windows,
        "n_attack_windows": n_attack_windows,
    }


def event_level_evaluation(losses, labels):
    """Evaluate at event level using reconstruction loss as anomaly score."""
    valid = labels >= 0  # exclude padding
    losses = losses[valid]
    labels = labels[valid]

    n_attack = int(labels.sum())
    n_total = len(labels)

    print(f"\n  Event-Level Evaluation")
    print(f"    Total events:  {n_total:,}")
    print(f"    Attack events: {n_attack:,} ({100*n_attack/n_total:.2f}%)")

    if n_attack == 0 or n_attack == n_total:
        print(f"    SKIPPED: cannot compute metrics")
        return {}

    auprc = average_precision_score(labels, losses)
    auroc = roc_auc_score(labels, losses)

    print(f"    AUPRC: {auprc:.4f}")
    print(f"    AUROC: {auroc:.4f}")

    return {
        "event_auprc": auprc,
        "event_auroc": auroc,
        "n_events": n_total,
        "n_attack": n_attack,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KAIROS baseline")
    parser.add_argument("--dataset", default="theia",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--split", default="test",
                        choices=["val", "test"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    graphs_dir = GRAPHS_DIR / args.dataset
    models_dir = MODELS_DIR / args.dataset

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ========================================================
    # Load model
    # ========================================================
    print("Loading model...")
    model_parts = torch.load(
        models_dir / "model.pt", map_location=device, weights_only=False)
    memory, gnn, link_pred, neighbor_loader = model_parts

    model_config = torch.load(
        models_dir / "model_config.pt", weights_only=False)
    max_node_num = model_config["max_node_num"]
    assoc = torch.empty(max_node_num, dtype=torch.long, device=device)

    for name, param in memory.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN/Inf in memory.{name}")
    for name, param in gnn.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN/Inf in gnn.{name}")
    for name, param in link_pred.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN/Inf in link_pred.{name}")

    # ========================================================
    # Load test data
    # ========================================================
    print(f"Loading {args.split} data...")
    test_data = torch.load(
        graphs_dir / f"{args.split}.TemporalData.pt",
        weights_only=False
    ).to(device)
    labels = np.load(graphs_dir / f"{args.split}_labels.npy")
    print(f"  Edges: {test_data.num_events:,}")

    # ========================================================
    # Compute losses
    # ========================================================
    print("Computing reconstruction losses...")
    t0 = time.time()
    losses, timestamps = compute_reconstruction_losses(
        test_data, memory, gnn, link_pred,
        neighbor_loader, assoc, device, NODE_EMBEDDING_DIM,
    )
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Mean loss: {losses.mean():.4f}")
    print(f"  Max loss:  {losses.max():.4f}")

    # ========================================================
    # Evaluate
    # ========================================================
    print("\n" + "=" * 60)
    print(f"KAIROS RESULTS — {args.dataset.upper()} ({args.split})")
    print("=" * 60)

    # Event-level (for THyN comparison)
    event_results = event_level_evaluation(losses, labels)

    # Window-level (native KAIROS)
    window_results = window_level_evaluation(
        losses, timestamps, labels)

    # Save results
    results = {
        "dataset": args.dataset,
        "split": args.split,
        **event_results,
        **window_results,
    }
    results_path = models_dir / f"results_{args.split}.pt"
    torch.save(results, results_path)
    print(f"\nResults saved to {results_path}")

    # Throughput
    throughput = len(losses) / elapsed
    print(f"\nThroughput: {throughput:,.0f} events/s")


if __name__ == "__main__":
    main()
