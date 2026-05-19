"""
KAIROS training script for HyperMamba-NIDS baseline comparison.

Trains TGN with self-supervised edge-type prediction, following the
original KAIROS paper (Zhu et al., USENIX Security 2023).

Memory-safe: TGNMemory + LastNeighborLoader stay on CPU,
only GNN/LinkPred forward pass runs on GPU.

Usage:
    python -m baselines.kairos.train --dataset theia
    python -m baselines.kairos.train --dataset theia --epochs 50
"""

import argparse
import copy
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
from tqdm import tqdm

from baselines.kairos.config import (
    DATASET_CONFIGS, GRAPHS_DIR, MODELS_DIR,
    NODE_EMBEDDING_DIM, NODE_STATE_DIM, NEIGHBOR_SIZE,
    EDGE_DIM, TIME_DIM, BATCH_SIZE, LR, EPS, WEIGHT_DECAY,
    EPOCH_NUM, build_rel2id,
)
from baselines.kairos.model import GraphAttentionEmbedding, LinkPredictor


def sequential_batches(data, batch_size):
    """Yield mini-batches from TemporalData in temporal order."""
    num_events = data.num_events
    for start in range(0, num_events, batch_size):
        end = min(start + batch_size, num_events)
        yield (
            data.src[start:end],
            data.dst[start:end],
            data.t[start:end],
            data.msg[start:end],
        )


def train_epoch(train_data, memory, gnn, link_pred, optimizer,
                neighbor_loader, assoc, criterion, compute_device,
                node_emb_dim):
    """Train one epoch. Memory on CPU, GNN/LinkPred on GPU."""
    cpu = torch.device("cpu")

    memory.train()
    gnn.train()
    link_pred.train()
    memory.reset_state()
    neighbor_loader.reset_state()
    total_loss = 0.0
    n_processed = 0

    for src, pos_dst, t, msg in sequential_batches(train_data, BATCH_SIZE):
        # Keep on CPU for memory/neighbor_loader
        src_cpu = src.to(cpu)
        pos_dst_cpu = pos_dst.to(cpu)
        t_cpu = t.to(cpu)
        msg_cpu = msg.to(cpu)

        optimizer.zero_grad()

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
        e_t = train_data.t[e_id].to(compute_device)
        e_msg = train_data.msg[e_id].to(compute_device)

        z_gpu = gnn(z_gpu, last_update_gpu, edge_index_gpu, e_t, e_msg)

        # Link prediction (GPU)
        src_idx = assoc[src_cpu].to(compute_device)
        dst_idx = assoc[pos_dst_cpu].to(compute_device)
        pos_out = link_pred(z_gpu[src_idx], z_gpu[dst_idx])

        # Loss (GPU)
        msg_gpu = msg_cpu.to(compute_device)
        y_true = torch.argmax(msg_gpu[:, node_emb_dim:-node_emb_dim], dim=1)

        loss = criterion(pos_out, y_true)
        loss.backward()
        optimizer.step()
        memory.detach()

        # Update memory & neighbor loader (CPU)
        memory.update_state(src_cpu, pos_dst_cpu, t_cpu, msg_cpu)
        neighbor_loader.insert(src_cpu, pos_dst_cpu)

        total_loss += loss.item() * len(src_cpu)
        n_processed += len(src_cpu)

    return total_loss / n_processed


def main():
    parser = argparse.ArgumentParser(
        description="Train KAIROS baseline")
    parser.add_argument("--dataset", default="theia",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=EPOCH_NUM)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Setup
    cfg = DATASET_CONFIGS[args.dataset]
    rel2id = build_rel2id(cfg["include_edge_type"])
    n_edge_types = len(cfg["include_edge_type"])

    graphs_dir = GRAPHS_DIR / args.dataset
    models_dir = MODELS_DIR / args.dataset
    models_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    logging.basicConfig(
        filename=str(models_dir / "training.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("kairos_train")

    # Compute device (for GNN/LinkPred only)
    if args.device == "auto":
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        compute_device = torch.device(args.device)
    cpu = torch.device("cpu")
    print(f"Compute device: {compute_device}")

    # ========================================================
    # Load training data — keep on CPU
    # ========================================================
    print("Loading training data (CPU)...")
    train_data = torch.load(
        graphs_dir / "train.TemporalData.pt",
        weights_only=False
    )
    # Ensure CPU
    train_data.src = train_data.src.to(cpu)
    train_data.dst = train_data.dst.to(cpu)
    train_data.t = train_data.t.to(cpu)
    train_data.msg = train_data.msg.to(cpu)
    print(f"  Train edges: {train_data.num_events:,}")

    # Load metadata
    meta = torch.load(graphs_dir / "metadata.pt", weights_only=False)
    max_node_num = meta["num_nodes"] + 1
    node_feat_size = meta["msg_dim"]
    print(f"  Max nodes: {max_node_num:,}")
    print(f"  Msg dim: {node_feat_size}")
    print(f"  TGN memory size: ~{max_node_num * NODE_STATE_DIM * 4 / 1e9:.2f} GB (on CPU)")

    # ========================================================
    # Initialize models
    # ========================================================
    print("Initializing models...")

    # Memory stays on CPU (too large for GPU with TRACE's 7.6M nodes)
    memory = TGNMemory(
        max_node_num,
        node_feat_size,
        NODE_STATE_DIM,
        TIME_DIM,
        message_module=IdentityMessage(
            node_feat_size, NODE_STATE_DIM, TIME_DIM),
        aggregator_module=LastAggregator(),
    ).to(cpu)

    # GNN + LinkPred on compute device
    # IMPORTANT: deep-copy time_enc so .to(cuda) doesn't drag memory's copy
    gnn = GraphAttentionEmbedding(
        in_channels=NODE_STATE_DIM,
        out_channels=EDGE_DIM,
        msg_dim=node_feat_size,
        time_enc=copy.deepcopy(memory.time_enc),
    ).to(compute_device)

    link_pred = LinkPredictor(
        in_channels=EDGE_DIM,
        out_channels=n_edge_types,
    ).to(compute_device)

    optimizer = torch.optim.Adam(
        list(memory.parameters()) +
        list(gnn.parameters()) +
        list(link_pred.parameters()),
        lr=LR, eps=EPS, weight_decay=WEIGHT_DECAY,
    )

    # Neighbor loader on CPU
    neighbor_loader = LastNeighborLoader(
        max_node_num, size=NEIGHBOR_SIZE, device=cpu)

    # assoc on CPU
    assoc = torch.empty(max_node_num, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

    if compute_device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")

    # ========================================================
    # Train
    # ========================================================
    print(f"\nTraining for {args.epochs} epochs...")
    t_total = time.time()

    for epoch in tqdm(range(1, args.epochs + 1)):
        t0 = time.time()
        loss = train_epoch(
            train_data, memory, gnn, link_pred, optimizer,
            neighbor_loader, assoc, criterion, compute_device,
            NODE_EMBEDDING_DIM,
        )
        elapsed = time.time() - t0
        msg = f"Epoch {epoch:02d}, Loss: {loss:.4f}, Time: {elapsed:.1f}s"
        logger.info(msg)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {msg}")

    total_time = time.time() - t_total
    print(f"\nTraining complete. Total time: {total_time:.1f}s")

    # Save model — move everything to CPU for portable checkpoints
    memory = memory.to(cpu)
    gnn = gnn.to(cpu)
    link_pred = link_pred.to(cpu)
    model = [memory, gnn, link_pred, neighbor_loader]
    torch.save(model, models_dir / "model.pt")
    print(f"Model saved to {models_dir / 'model.pt'}")

    # Also save model config for reproducibility
    model_config = {
        "max_node_num": max_node_num,
        "node_feat_size": node_feat_size,
        "n_edge_types": n_edge_types,
        "node_state_dim": NODE_STATE_DIM,
        "edge_dim": EDGE_DIM,
        "time_dim": TIME_DIM,
        "neighbor_size": NEIGHBOR_SIZE,
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
        "lr": LR,
    }
    torch.save(model_config, models_dir / "model_config.pt")


if __name__ == "__main__":
    main()
