"""
Extract per-event embeddings from a trained KAIROS model.

For each event, extracts:
  - z_src: GNN output embedding for the source node (EDGE_DIM)
  - z_dst: GNN output embedding for the destination node (EDGE_DIM)
  - recon_loss: per-event reconstruction loss (scalar)

Saves per-shard .npz files with keys: embeddings, labels, losses.

Usage:
    python -m baselines.kairos.extract_embeddings --dataset theia
    python -m baselines.kairos.extract_embeddings --dataset trace
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    LastNeighborLoader, IdentityMessage, LastAggregator
)

from baselines.kairos.config import (
    DATASET_CONFIGS, GRAPHS_DIR, MODELS_DIR, ARTIFACT_DIR,
    NODE_EMBEDDING_DIM, NODE_STATE_DIM, NEIGHBOR_SIZE,
    EDGE_DIM, TIME_DIM, BATCH_SIZE,
)


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
def extract_embeddings(data, memory, gnn, link_pred,
                       neighbor_loader, assoc,
                       compute_device, node_emb_dim):
    """Extract per-event GNN embeddings and reconstruction losses.

    Returns:
        embeddings: (num_events, EDGE_DIM * 2) — concat(z_src, z_dst)
        losses: (num_events,) — per-event reconstruction loss
    """
    cpu = torch.device("cpu")

    memory.eval()
    gnn.eval()
    link_pred.eval()
    memory.reset_state()
    neighbor_loader.reset_state()

    # Ensure memory on CPU, break shared time_enc
    memory = memory.to(cpu)
    gnn.time_enc = copy.deepcopy(gnn.time_enc)
    gnn = gnn.to(compute_device)
    link_pred = link_pred.to(compute_device)

    criterion = nn.CrossEntropyLoss(reduction="none")

    all_embeddings = []
    all_losses = []

    n_batches = (data.num_events + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, (src, pos_dst, t, msg) in enumerate(
            sequential_batches(data, BATCH_SIZE)):
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

        # Extract per-event embeddings
        src_idx = assoc[src_cpu].to(compute_device)
        dst_idx = assoc[pos_dst_cpu].to(compute_device)
        z_src = z_gpu[src_idx]  # (batch, EDGE_DIM)
        z_dst = z_gpu[dst_idx]  # (batch, EDGE_DIM)

        # Concatenate src+dst embeddings
        event_emb = torch.cat([z_src, z_dst], dim=-1)  # (batch, EDGE_DIM*2)
        all_embeddings.append(event_emb.cpu().numpy())

        # Also compute reconstruction loss
        pos_out = link_pred(z_src, z_dst)
        msg_gpu = msg_cpu.to(compute_device)
        y_true = torch.argmax(msg_gpu[:, node_emb_dim:-node_emb_dim], dim=1)
        loss = criterion(pos_out.float(), y_true)
        all_losses.append(loss.cpu().numpy())

        # Update memory (CPU)
        memory.update_state(src_cpu, pos_dst_cpu, t_cpu, msg_cpu)
        neighbor_loader.insert(src_cpu, pos_dst_cpu)

        if (batch_idx + 1) % 500 == 0 or batch_idx == 0:
            print(f"    Batch {batch_idx+1}/{n_batches}")

    embeddings = np.concatenate(all_embeddings)  # (N, EDGE_DIM*2)
    losses = np.concatenate(all_losses)           # (N,)
    return embeddings, losses


def main():
    parser = argparse.ArgumentParser(
        description="Extract KAIROS embeddings for supervised head")
    parser.add_argument("--dataset", default="theia",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    graphs_dir = GRAPHS_DIR / args.dataset
    models_dir = MODELS_DIR / args.dataset
    output_dir = ARTIFACT_DIR / "embeddings" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        compute_device = torch.device(args.device)
    print(f"Compute device: {compute_device}")

    # ========================================================
    # Load model
    # ========================================================
    print("Loading model to CPU...")
    model_parts = torch.load(
        models_dir / "model.pt", map_location="cpu", weights_only=False)
    memory, gnn, link_pred, neighbor_loader = model_parts

    model_config = torch.load(
        models_dir / "model_config.pt", weights_only=False)
    max_node_num = model_config["max_node_num"]
    print(f"  max_node_num: {max_node_num:,}")

    assoc = torch.empty(max_node_num, dtype=torch.long)

    # ========================================================
    # Extract for each split
    # ========================================================
    for split in ["train", "val", "test"]:
        data_path = graphs_dir / f"{split}.TemporalData.pt"
        labels_path = graphs_dir / f"{split}_labels.npy"

        if not data_path.exists():
            print(f"\n  {split}: data not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Extracting {split} embeddings...")
        print(f"{'='*60}")

        data = torch.load(data_path, weights_only=False)
        data.src = data.src.to("cpu")
        data.dst = data.dst.to("cpu")
        data.t = data.t.to("cpu")
        data.msg = data.msg.to("cpu")

        labels = np.load(labels_path)
        print(f"  Events: {data.num_events:,}")
        print(f"  Attack: {int((labels == 1).sum()):,} ({100*(labels==1).mean():.1f}%)")

        # Need fresh memory state for each split
        # Reload model for each split to get clean state
        model_parts = torch.load(
            models_dir / "model.pt", map_location="cpu", weights_only=False)
        mem, gnn_m, lp_m, nl_m = model_parts

        t0 = time.time()
        embeddings, losses = extract_embeddings(
            data, mem, gnn_m, lp_m, nl_m, assoc,
            compute_device, NODE_EMBEDDING_DIM
        )
        elapsed = time.time() - t0

        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Throughput: {data.num_events/elapsed:,.0f} events/s")

        # Save
        out_path = output_dir / f"{split}_embeddings.npz"
        np.savez_compressed(
            out_path,
            embeddings=embeddings,
            labels=labels,
            losses=losses,
        )
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Saved to {out_path} ({size_mb:.1f} MB)")

        del data, embeddings, losses, labels, mem, gnn_m, lp_m, nl_m
        import gc; gc.collect()
        if compute_device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n✓ All embeddings extracted to {output_dir}")


if __name__ == "__main__":
    main()
