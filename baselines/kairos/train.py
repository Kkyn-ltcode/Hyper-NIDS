"""
KAIROS training script for HyperMamba-NIDS baseline comparison.

Trains TGN with self-supervised edge-type prediction, following the
original KAIROS paper (Zhu et al., USENIX Security 2023).

Usage:
    python -m baselines.kairos.train --dataset theia
    python -m baselines.kairos.train --dataset theia --epochs 50
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
from tqdm import tqdm

from baselines.kairos.config import (
    DATASET_CONFIGS, GRAPHS_DIR, MODELS_DIR,
    NODE_EMBEDDING_DIM, NODE_STATE_DIM, NEIGHBOR_SIZE,
    EDGE_DIM, TIME_DIM, BATCH_SIZE, LR, EPS, WEIGHT_DECAY,
    EPOCH_NUM, build_rel2id,
)
from baselines.kairos.model import GraphAttentionEmbedding, LinkPredictor


def tensor_find(t, x):
    """Find index of value x in tensor t."""
    idx = np.argwhere(t.cpu().numpy() == x)
    return idx[0][0] + 1

def sequential_batches(data, batch_size):
    """
    Generator that yields mini-batches from TemporalData in temporal order.
    Replaces data.seq_batches(batch_size) for older PyG versions.
    """
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
                neighbor_loader, assoc, criterion, device, node_emb_dim):
    memory.train()
    gnn.train()
    link_pred.train()
    memory.reset_state()
    neighbor_loader.reset_state()
    total_loss = 0.0
    n_processed = 0

    for src, pos_dst, t, msg in sequential_batches(train_data, BATCH_SIZE):
        src = src.to(device)
        pos_dst = pos_dst.to(device)
        t = t.to(device)
        msg = msg.to(device)

        optimizer.zero_grad()

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index,
                train_data.t[e_id].to(device), train_data.msg[e_id].to(device))
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        # Extract edge type labels (one-hot in msg)
        y_true = torch.argmax(msg[:, node_emb_dim:-node_emb_dim], dim=1)

        loss = criterion(pos_out, y_true)
        loss.backward()
        optimizer.step()
        memory.detach()

        # Update memory & neighbor loader
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        total_loss += loss.item() * len(src)
        n_processed += len(src)

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

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ========================================================
    # Load training data
    # ========================================================
    print("Loading training data...")
    train_data = torch.load(
        graphs_dir / "train.TemporalData.pt",
        weights_only=False
    ).to(device)
    print(f"  Train edges: {train_data.num_events:,}")

    # Load metadata
    meta = torch.load(graphs_dir / "metadata.pt", weights_only=False)
    max_node_num = meta["num_nodes"] + 1
    node_feat_size = meta["msg_dim"]
    print(f"  Max nodes: {max_node_num:,}")
    print(f"  Msg dim: {node_feat_size}")

    # ========================================================
    # Initialize models
    # ========================================================
    print("Initializing models...")

    memory = TGNMemory(
        max_node_num,
        node_feat_size,
        NODE_STATE_DIM,
        TIME_DIM,
        message_module=IdentityMessage(
            node_feat_size, NODE_STATE_DIM, TIME_DIM),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=NODE_STATE_DIM,
        out_channels=EDGE_DIM,
        msg_dim=node_feat_size,
        time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(
        in_channels=EDGE_DIM,
        out_channels=n_edge_types,
    ).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) |
        set(gnn.parameters()) |
        set(link_pred.parameters()),
        lr=LR, eps=EPS, weight_decay=WEIGHT_DECAY,
    )

    neighbor_loader = LastNeighborLoader(
        max_node_num, size=NEIGHBOR_SIZE, device=device)

    assoc = torch.empty(max_node_num, dtype=torch.long, device=device)
    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # Train
    # ========================================================
    print(f"\nTraining for {args.epochs} epochs...")
    t_total = time.time()

    for epoch in tqdm(range(1, args.epochs + 1)):
        t0 = time.time()
        loss = train_epoch(
            train_data, memory, gnn, link_pred, optimizer,
            neighbor_loader, assoc, criterion, device,
            NODE_EMBEDDING_DIM,
        )
        elapsed = time.time() - t0
        msg = f"Epoch {epoch:02d}, Loss: {loss:.4f}, Time: {elapsed:.1f}s"
        logger.info(msg)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  {msg}")

    total_time = time.time() - t_total
    print(f"\nTraining complete. Total time: {total_time:.1f}s")

    # Save model
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
