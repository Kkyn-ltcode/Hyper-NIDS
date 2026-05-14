"""
Convert HyperMamba-NIDS Parquet shards to KAIROS TemporalData format.

This bypasses KAIROS's PostgreSQL pipeline by reading directly from
our labeled Parquet files and producing PyG TemporalData objects.

KAIROS TemporalData has:
  - src: (E,) int64 — source node integer IDs
  - dst: (E,) int64 — destination node integer IDs
  - t:   (E,) int64 — nanosecond timestamps
  - msg: (E, D) float — concatenation of [src_feat, edge_onehot, dst_feat]

Usage:
    python -m baselines.kairos.data_converter --dataset theia
"""

import argparse
import hashlib
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import TemporalData

from baselines.kairos.config import (
    DATASET_CONFIGS, ARTIFACT_DIR, GRAPHS_DIR,
    NODE_EMBEDDING_DIM, build_rel2id,
)


# ============================================================
# Helpers (from KAIROS utils)
# ============================================================

def string_to_hash(s: str) -> str:
    """SHA-256 hash of a string (KAIROS convention)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def path_to_hierarchy(p: str) -> list[str]:
    """Convert file path to hierarchical representation.

    '/usr/bin/bash' → ['usr', 'usr/bin', 'usr/bin/bash']
    """
    p = str(p).strip()  # ensure string
    if not p or p.lower() == "nan":
        return ["unknown"]
    parts = p.split("/")
    result = []
    for part in parts:
        if not part:
            continue
        if result:
            result.append(result[-1] + "/" + part)
        else:
            result.append(part)
    return result


def ip_to_hierarchy(ip: str) -> list[str]:
    """Convert IP to hierarchical representation.

    '192.168.1.1' → ['192', '192.168', '192.168.1', '192.168.1.1']
    """
    ip = str(ip).strip()
    if not ip or ip.lower() == "nan":
        return ["unknown"]
    parts = ip.split(".")
    result = []
    for part in parts:
        if result:
            result.append(result[-1] + "." + part)
        else:
            result.append(part)
    return result


def build_node_features(node_labels: dict[int, tuple[str, str]],
                        n_features: int = NODE_EMBEDDING_DIM
                        ) -> np.ndarray:
    """Build node feature vectors using FeatureHasher.

    Args:
        node_labels: {node_id: (node_type, label_string)}
            node_type is one of: 'subject', 'file', 'netflow'
            label_string is the path, IP:port, or cmd_line

    Returns:
        node2vec: (num_nodes, n_features) float32 array
    """
    max_id = max(node_labels.keys()) + 1
    fh = FeatureHasher(n_features=n_features, input_type="string")

    # Collect token lists for all nodes
    token_lists = []
    for nid in range(max_id):
        if nid in node_labels:
            ntype, label = node_labels[nid]
            if ntype == "netflow":
                hierarchy = ["netflow"] + ip_to_hierarchy(label.split(":")[0])
            elif ntype == "file":
                hierarchy = ["file"] + path_to_hierarchy(label)
            else:  # subject
                hierarchy = ["subject"] + path_to_hierarchy(label)
        else:
            hierarchy = ["unknown"]
        token_lists.append(hierarchy)

    # Transform all at once
    # FeatureHasher expects an iterable of iterables of strings
    node2vec = fh.transform(token_lists).toarray().astype(np.float32)
    return node2vec


# ============================================================
# Main conversion
# ============================================================

def convert_dataset(dataset: str):
    """Convert a dataset from Parquet to KAIROS TemporalData."""

    cfg = DATASET_CONFIGS[dataset]
    data_dir = cfg["data_dir"]
    labeled_dir = data_dir / "labeled"
    include_types = set(cfg["include_edge_type"])
    reversed_types = set(cfg["edge_reversed"])
    rel2id = build_rel2id(cfg["include_edge_type"])

    # Create output directories
    out_dir = GRAPHS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"KAIROS DATA CONVERTER: {dataset.upper()}")
    print("=" * 60)

    # ========================================================
    # Step 1: Build entity vocabulary
    # ========================================================
    print("\nStep 1: Build entity vocabulary...")
    t0 = time.time()

    subjects_df = pd.read_parquet(data_dir / "subjects.parquet")
    objects_df = pd.read_parquet(data_dir / "objects.parquet")

    # Build UUID → integer ID mapping
    uuid_to_id = {}
    node_labels = {}  # id → (type, label_string)
    next_id = 0

    # Subjects: use process_path or cmd_line
    for _, row in subjects_df.iterrows():
        uid = row["uuid"]
        if uid not in uuid_to_id:
            uuid_to_id[uid] = next_id
            # Safe extraction: convert to string, replace NaN/"nan" with "unknown"
            label = row.get("process_path", None)
            if pd.isna(label) or str(label).strip() == "" or str(label).lower() == "nan":
                label = row.get("cmd_line", None)
            if pd.isna(label) or str(label).strip() == "" or str(label).lower() == "nan":
                label = "unknown"
            else:
                label = str(label)
            node_labels[next_id] = ("subject", label)
            next_id += 1

    # Objects: use filename, or IP:port
    for _, row in objects_df.iterrows():
        uid = row["uuid"]
        if uid not in uuid_to_id:
            uuid_to_id[uid] = next_id
            otype = row.get("object_type", "unknown")
            if otype == "FILE" or otype == "MEMORY":
                label = row.get("filename", None)
                if pd.isna(label) or str(label).strip() == "" or str(label).lower() == "nan":
                    label = "unknown"
                else:
                    label = str(label)
                node_labels[next_id] = ("file", label)
            elif otype == "NETFLOW":
                addr = row.get("remote_address", None)
                port = row.get("remote_port", None)
                if pd.isna(addr) or str(addr).strip() == "":
                    addr = ""
                else:
                    addr = str(addr)
                if pd.isna(port) or str(port).strip() == "":
                    port = ""
                else:
                    port = str(port)
                label = f"{addr}:{port}" if addr or port else "unknown"
                node_labels[next_id] = ("netflow", label)
            else:
                node_labels[next_id] = ("file", "unknown")
            next_id += 1

    num_nodes = next_id
    print(f"  Nodes: {num_nodes:,}")
    print(f"    Subjects: {len(subjects_df):,}")
    print(f"    Objects:  {len(objects_df):,}")
    print(f"  Time: {time.time()-t0:.1f}s")

    del subjects_df, objects_df
    gc.collect()

    # ========================================================
    # Step 2: Build node features
    # ========================================================
    print("\nStep 2: Build node features (FeatureHasher)...")
    t0 = time.time()
    node2vec = build_node_features(node_labels, NODE_EMBEDDING_DIM)
    print(f"  Shape: {node2vec.shape}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Save node features
    torch.save(node2vec, out_dir / "node2higvec.pt")

    # ========================================================
    # Step 3: Build edge type one-hot vectors
    # ========================================================
    n_edge_types = len(cfg["include_edge_type"])
    rel_onehot = torch.nn.functional.one_hot(
        torch.arange(n_edge_types), num_classes=n_edge_types
    ).float()
    rel2vec = {}
    for etype in cfg["include_edge_type"]:
        idx = rel2id[etype] - 1  # 0-indexed
        rel2vec[etype] = rel_onehot[idx]
    torch.save(rel2vec, out_dir / "rel2vec.pt")
    print(f"\n  Edge types: {n_edge_types}")

    # ========================================================
    # Step 4: Convert shards to TemporalData
    # ========================================================
    all_shards = sorted(labeled_dir.glob("labeled_shard*.parquet"))
    print(f"\nStep 4: Convert {len(all_shards)} shards to TemporalData...")

    # Determine shard groups
    all_shard_indices = cfg["train_shards"] + cfg["val_shards"] + cfg["test_shards"]
    shard_groups = {}
    for idx in cfg["train_shards"]:
        shard_groups[idx] = "train"
    for idx in cfg["val_shards"]:
        shard_groups[idx] = "val"
    for idx in cfg["test_shards"]:
        shard_groups[idx] = "test"

    for split in ["train", "val", "test"]:
        split_indices = [i for i, g in shard_groups.items() if g == split]
        if not split_indices:
            continue

        all_src, all_dst, all_t, all_msg = [], [], [], []
        all_labels = []
        total_events = 0
        total_filtered = 0

        for shard_idx in split_indices:
            shard_path = labeled_dir / f"labeled_shard{shard_idx}.parquet"
            if not shard_path.exists():
                print(f"  WARNING: {shard_path} not found, skipping")
                continue

            df = pd.read_parquet(shard_path, columns=[
                "type", "timestamp_nanos",
                "subject_uuid", "predicate_object_uuid",
                "label_broad",
            ])

            total_events += len(df)

            # Filter to included event types
            mask = df["type"].isin(include_types)
            df = df[mask].reset_index(drop=True)
            total_filtered += len(df)

            # Map UUIDs to integer IDs
            src_ids = df["subject_uuid"].map(uuid_to_id)
            dst_ids = df["predicate_object_uuid"].map(uuid_to_id)

            # Drop events with unmapped entities
            valid = src_ids.notna() & dst_ids.notna()
            df = df[valid].reset_index(drop=True)
            src_ids = src_ids[valid].astype(int).values
            dst_ids = dst_ids[valid].astype(int).values

            # Handle reversed edges
            for i, etype in enumerate(df["type"].values):
                if etype in reversed_types:
                    src_ids[i], dst_ids[i] = dst_ids[i], src_ids[i]

            # Build message vectors: [src_feat, edge_onehot, dst_feat]
            timestamps = df["timestamp_nanos"].values.astype(np.int64)
            labels = df["label_broad"].values.astype(np.int8)

            msgs = []
            for i in range(len(df)):
                src_feat = torch.from_numpy(node2vec[src_ids[i]])
                edge_feat = rel2vec[df["type"].iloc[i]]
                dst_feat = torch.from_numpy(node2vec[dst_ids[i]])
                msgs.append(torch.cat([src_feat, edge_feat, dst_feat]))

            all_src.append(torch.tensor(src_ids, dtype=torch.long))
            all_dst.append(torch.tensor(dst_ids, dtype=torch.long))
            all_t.append(torch.tensor(timestamps, dtype=torch.long))
            if msgs:
                all_msg.append(torch.stack(msgs))
            all_labels.append(labels)

            print(f"    Shard {shard_idx}: {len(df):,} events "
                  f"(filtered from {(~mask).sum() + (~valid).sum():,})")

            del df
            gc.collect()

        if not all_src:
            continue

        # Combine
        dataset_td = TemporalData(
            src=torch.cat(all_src),
            dst=torch.cat(all_dst),
            t=torch.cat(all_t),
            msg=torch.cat(all_msg),
        )

        # Sort by timestamp
        sort_idx = dataset_td.t.argsort()
        dataset_td.src = dataset_td.src[sort_idx]
        dataset_td.dst = dataset_td.dst[sort_idx]
        dataset_td.t = dataset_td.t[sort_idx]
        dataset_td.msg = dataset_td.msg[sort_idx]

        labels_concat = np.concatenate(all_labels)[sort_idx.numpy()]

        # Save
        torch.save(dataset_td, out_dir / f"{split}.TemporalData.pt")
        np.save(out_dir / f"{split}_labels.npy", labels_concat)

        n_attack = int(labels_concat.sum())
        print(f"\n  {split}: {len(dataset_td.src):,} edges "
              f"({total_filtered:,} after type filter from {total_events:,})")
        print(f"    Attack events: {n_attack:,} "
              f"({100*n_attack/len(labels_concat):.2f}%)")
        print(f"    msg shape: {dataset_td.msg.shape}")

        del all_src, all_dst, all_t, all_msg, all_labels
        gc.collect()

    # Save metadata
    meta = {
        "num_nodes": num_nodes,
        "uuid_to_id_size": len(uuid_to_id),
        "n_edge_types": n_edge_types,
        "include_edge_type": cfg["include_edge_type"],
        "msg_dim": NODE_EMBEDDING_DIM * 2 + n_edge_types,
    }
    torch.save(meta, out_dir / "metadata.pt")
    print(f"\n  Metadata: {meta}")
    print(f"\nDone. Outputs in {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Parquet shards to KAIROS TemporalData")
    parser.add_argument("--dataset", default="theia",
                        choices=list(DATASET_CONFIGS.keys()))
    args = parser.parse_args()
    convert_dataset(args.dataset)


if __name__ == "__main__":
    main()
