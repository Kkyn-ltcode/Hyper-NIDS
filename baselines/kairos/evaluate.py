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
    precision_recall_curve,
)

from baselines.kairos.config import (
    DATASET_CONFIGS, GRAPHS_DIR, MODELS_DIR,
    NODE_EMBEDDING_DIM, NODE_STATE_DIM, NEIGHBOR_SIZE,
    EDGE_DIM, TIME_DIM, BATCH_SIZE, TIME_WINDOW_SIZE,
    build_rel2id,
)
from baselines.kairos.model import GraphAttentionEmbedding, LinkPredictor


def find_best_threshold(scores, labels):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = np.argmax(f1)
    return thresholds[best_idx], f1[best_idx]


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
                                  neighbor_loader, assoc,
                                  compute_device, node_emb_dim):
    """Compute per-event reconstruction loss.

    Memory and neighbor_loader stay on CPU to avoid OOM on large graphs.
    Only the GNN forward pass runs on GPU.
    """
    memory.eval()
    gnn.eval()
    link_pred.eval()
    memory.reset_state()
    neighbor_loader.reset_state()

    # Ensure memory internals are on CPU
    cpu = torch.device("cpu")
    memory = memory.to(cpu)

    # GNN + link_pred on compute device
    gnn = gnn.to(compute_device)
    link_pred = link_pred.to(compute_device)

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    all_timestamps = []

    n_batches = (data.num_events + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, (src, pos_dst, t, msg) in enumerate(
            sequential_batches(data, BATCH_SIZE)):
        # Keep src/dst/t/msg on CPU for memory & neighbor_loader
        src_cpu = src.to(cpu)
        pos_dst_cpu = pos_dst.to(cpu)
        t_cpu = t.to(cpu)
        msg_cpu = msg.to(cpu)

        # Neighbor lookup (CPU)
        n_id = torch.cat([src_cpu, pos_dst_cpu]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0))

        # Memory forward (CPU)
        z, last_update = memory(n_id)

        # GNN forward (GPU)
        z_gpu = z.to(compute_device)
        last_update_gpu = last_update.to(compute_device)
        edge_index_gpu = edge_index.to(compute_device)
        e_t = data.t[e_id].to(compute_device)
        e_msg = data.msg[e_id].to(compute_device)

        z_gpu = gnn(z_gpu, last_update_gpu, edge_index_gpu, e_t, e_msg)

        # Link prediction (GPU)
        src_idx = assoc[src_cpu].to(compute_device)
        dst_idx = assoc[pos_dst_cpu].to(compute_device)
        pos_out = link_pred(z_gpu[src_idx], z_gpu[dst_idx])

        # Loss (GPU → CPU)
        msg_gpu = msg_cpu.to(compute_device)
        y_true = torch.argmax(msg_gpu[:, node_emb_dim:-node_emb_dim], dim=1)
        loss = criterion(pos_out.float(), y_true)

        all_losses.append(loss.cpu().numpy())
        all_timestamps.append(t_cpu.numpy())

        # Update memory & neighbor loader (CPU)
        memory.update_state(src_cpu, pos_dst_cpu, t_cpu, msg_cpu)
        neighbor_loader.insert(src_cpu, pos_dst_cpu)

        if (batch_idx + 1) % 500 == 0 or batch_idx == 0:
            print(f"    Batch {batch_idx+1}/{n_batches}")

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

    # Threshold-based: use Youden's J statistic
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

    # Device for compute (GNN/LinkPred)
    if args.device == "auto":
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        compute_device = torch.device(args.device)
    print(f"Compute device: {compute_device}")

    # ========================================================
    # Load model — always to CPU first to avoid OOM
    # ========================================================
    print("Loading model to CPU...")
    model_parts = torch.load(
        models_dir / "model.pt", map_location="cpu", weights_only=False)
    memory, gnn, link_pred, _ = model_parts
    neighbor_loader = LastNeighborLoader(max_node_num, size=NEIGHBOR_SIZE)

    model_config = torch.load(
        models_dir / "model_config.pt", weights_only=False)
    max_node_num = model_config["max_node_num"]
    print(f"  max_node_num: {max_node_num:,}")
    print(f"  TGN memory size: ~{max_node_num * NODE_STATE_DIM * 4 / 1e9:.2f} GB")

    # assoc stays on CPU (indexed by CPU n_id)
    assoc = torch.empty(max_node_num, dtype=torch.long)

    # Sanity check model weights
    for name, param in memory.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"  WARNING: NaN/Inf in memory.{name}")
    for name, param in gnn.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"  WARNING: NaN/Inf in gnn.{name}")
    for name, param in link_pred.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"  WARNING: NaN/Inf in link_pred.{name}")

    # ========================================================
    # Load test data — keep on CPU, batch to GPU
    # ========================================================
    print(f"Loading {args.split} data (CPU)...")
    test_data = torch.load(
        graphs_dir / f"{args.split}.TemporalData.pt",
        weights_only=False
    )
    # Ensure data is on CPU
    test_data.src = test_data.src.to("cpu")
    test_data.dst = test_data.dst.to("cpu")
    test_data.t = test_data.t.to("cpu")
    test_data.msg = test_data.msg.to("cpu")

    labels = np.load(graphs_dir / f"{args.split}_labels.npy")
    print(f"  Edges: {test_data.num_events:,}")

    if compute_device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

    # ========================================================
    # Compute losses
    # ========================================================
    print("Computing reconstruction losses...")
    print("  Strategy: TGN memory on CPU, GNN/LinkPred on GPU")
    t0 = time.time()
    losses, timestamps = compute_reconstruction_losses(
        test_data, memory, gnn, link_pred,
        neighbor_loader, assoc, compute_device, NODE_EMBEDDING_DIM,
    )

    # ========================================================
    # Threshold handling
    # ========================================================
    if args.split == 'val':
        threshold, best_f1 = find_best_threshold(losses, labels)
        print(f"Validation best F1: {best_f1:.4f} at threshold {threshold:.4f}")
        torch.save({'threshold': threshold}, models_dir / 'threshold.pt')
        event_results = event_level_evaluation(losses, labels)
        event_results['val_f1'] = best_f1
    elif args.split == 'test':
        thresh_path = models_dir / 'threshold.pt'
        if thresh_path.exists():
            thresh_data = torch.load(thresh_path, weights_only=False)
            threshold = thresh_data['threshold']
            preds = (losses > threshold).astype(int)
            mask = labels >= 0
            p, r, f1, _ = precision_recall_fscore_support(
                labels[mask], preds[mask], average='binary', zero_division=0)
            print(f"\n  Event-Level F1 (threshold={threshold:.4f}):")
            print(f"    Precision: {p:.4f}")
            print(f"    Recall:    {r:.4f}")
            print(f"    F1:        {f1:.4f}")
            event_results = event_level_evaluation(losses, labels)
            event_results['precision'] = p
            event_results['recall'] = r
            event_results['f1'] = f1
            event_results['threshold'] = threshold
        else:
            print("  WARNING: No threshold.pt found. Run --split val first.")
            event_results = event_level_evaluation(losses, labels)
    else:
        event_results = event_level_evaluation(losses, labels)

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
    event_results_final = event_level_evaluation(losses, labels)

    # Window-level (native KAIROS)
    window_results = window_level_evaluation(
        losses, timestamps, labels)

    # Save results
    results = {
        "dataset": args.dataset,
        "split": args.split,
        **event_results,
        **event_results_final,
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
