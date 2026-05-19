"""
Convert HyperMamba-NIDS Parquet shards to KAIROS TemporalData format.

This bypasses KAIROS's PostgreSQL pipeline by reading directly from
our labeled Parquet files and producing PyG TemporalData objects.

Memory-safe: uses vectorized operations and batched FeatureHasher
for datasets with 100M+ nodes (e.g., full TRACE).

Usage:
    python -m baselines.kairos.data_converter --dataset theia
    python -m baselines.kairos.data_converter --dataset trace
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
# Helpers
# ============================================================

def path_to_hierarchy(p: str) -> list[str]:
    """'/usr/bin/bash' → ['usr', 'usr/bin', 'usr/bin/bash']"""
    p = str(p).strip()
    if not p or p.lower() == "nan":
        return ["unknown"]
    parts = [x for x in p.split("/") if x]
    result = []
    for part in parts:
        if result:
            result.append(result[-1] + "/" + part)
        else:
            result.append(part)
    return result or ["unknown"]


def ip_to_hierarchy(ip: str) -> list[str]:
    """'192.168.1.1' → ['192', '192.168', '192.168.1', '192.168.1.1']"""
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
    return result or ["unknown"]


def build_node_features_batched(node_labels: dict, n_features: int = NODE_EMBEDDING_DIM,
                                batch_size: int = 1_000_000) -> np.ndarray:
    """Build node features in batches to avoid OOM on large graphs.

    Args:
        node_labels: {node_id: (node_type, label_string)}
        n_features: feature hash dimension
        batch_size: nodes per batch for FeatureHasher

    Returns:
        (max_node_id+1, n_features) float32 array
    """
    max_id = max(node_labels.keys()) + 1
    print(f"    Allocating feature matrix: ({max_id:,}, {n_features}) "
          f"= {max_id * n_features * 4 / 1e9:.2f} GB")

    node2vec = np.zeros((max_id, n_features), dtype=np.float32)
    fh = FeatureHasher(n_features=n_features, input_type="string")

    n_batches = (max_id + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, max_id)

        token_lists = []
        for nid in range(start, end):
            if nid in node_labels:
                ntype, label = node_labels[nid]
                if ntype == "netflow":
                    hierarchy = ["netflow"] + ip_to_hierarchy(label.split(":")[0])
                elif ntype == "file":
                    hierarchy = ["file"] + path_to_hierarchy(label)
                else:
                    hierarchy = ["subject"] + path_to_hierarchy(label)
            else:
                hierarchy = ["unknown"]
            token_lists.append(hierarchy)

        batch_features = fh.transform(token_lists).toarray().astype(np.float32)
        node2vec[start:end] = batch_features

        if (batch_idx + 1) % 20 == 0 or batch_idx == 0 or batch_idx == n_batches - 1:
            print(f"    Batch {batch_idx+1}/{n_batches} "
                  f"(nodes {start:,}–{end:,})")

        del token_lists, batch_features

    return node2vec


def build_entity_vocab_vectorized(subjects_df, objects_df):
    """Build UUID→ID mapping using vectorized pandas, not iterrows().

    Returns:
        uuid_to_id: dict mapping UUID string → integer ID
        node_labels: dict mapping integer ID → (type, label_string)
    """
    uuid_to_id = {}
    node_labels = {}
    next_id = 0

    # --- Subjects: vectorized ---
    # Get unique subjects
    subj_uuids = subjects_df["uuid"].values
    # Pre-compute labels vectorized
    if "process_path" in subjects_df.columns:
        subj_labels = subjects_df["process_path"].fillna("")
        # Fall back to cmd_line where process_path is empty
        if "cmd_line" in subjects_df.columns:
            empty_mask = (subj_labels == "") | (subj_labels.str.lower() == "nan")
            subj_labels[empty_mask] = subjects_df.loc[empty_mask, "cmd_line"].fillna("")
    elif "cmd_line" in subjects_df.columns:
        subj_labels = subjects_df["cmd_line"].fillna("")
    else:
        subj_labels = pd.Series("unknown", index=subjects_df.index)

    subj_labels = subj_labels.fillna("unknown").replace({"": "unknown", "nan": "unknown"})

    for i in range(len(subjects_df)):
        uid = subj_uuids[i]
        if uid not in uuid_to_id:
            uuid_to_id[uid] = next_id
            node_labels[next_id] = ("subject", str(subj_labels.iloc[i]))
            next_id += 1

    n_subjects = next_id
    print(f"    Subjects: {n_subjects:,}")

    # --- Objects: vectorized ---
    obj_uuids = objects_df["uuid"].values
    obj_types = objects_df.get("object_type", pd.Series("unknown", index=objects_df.index)).fillna("unknown").values

    # Pre-compute object labels
    filenames = objects_df.get("filename", pd.Series("", index=objects_df.index)).fillna("").values
    remote_addrs = objects_df.get("remote_address", pd.Series("", index=objects_df.index)).fillna("").values
    remote_ports = objects_df.get("remote_port", pd.Series("", index=objects_df.index)).fillna("").values

    for i in range(len(objects_df)):
        uid = obj_uuids[i]
        if uid not in uuid_to_id:
            uuid_to_id[uid] = next_id
            otype = str(obj_types[i])
            if otype in ("FILE", "MEMORY"):
                label = str(filenames[i]) if filenames[i] else "unknown"
                if label.lower() == "nan" or not label.strip():
                    label = "unknown"
                node_labels[next_id] = ("file", label)
            elif otype == "NETFLOW":
                addr = str(remote_addrs[i]) if remote_addrs[i] else ""
                port = str(remote_ports[i]) if remote_ports[i] else ""
                if addr.lower() == "nan":
                    addr = ""
                if port.lower() == "nan":
                    port = ""
                label = f"{addr}:{port}" if addr or port else "unknown"
                node_labels[next_id] = ("netflow", label)
            else:
                node_labels[next_id] = ("file", "unknown")
            next_id += 1

    n_objects = next_id - n_subjects
    print(f"    Objects:  {n_objects:,}")
    print(f"    Total:    {next_id:,}")

    return uuid_to_id, node_labels


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

    out_dir = GRAPHS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"KAIROS DATA CONVERTER: {dataset.upper()}")
    print("=" * 60)

    # ========================================================
    # Step 1: Build entity vocabulary (vectorized)
    # ========================================================
    print("\nStep 1: Build entity vocabulary...")
    t0 = time.time()

    subjects_df = pd.read_parquet(data_dir / "subjects.parquet")
    objects_df = pd.read_parquet(data_dir / "objects.parquet")

    uuid_to_id, node_labels = build_entity_vocab_vectorized(subjects_df, objects_df)
    num_nodes = len(uuid_to_id)
    print(f"  Time: {time.time()-t0:.1f}s")

    del subjects_df, objects_df
    gc.collect()

    # ========================================================
    # Step 2: Build node features (batched)
    # ========================================================
    print("\nStep 2: Build node features (batched FeatureHasher)...")
    t0 = time.time()
    node2vec = build_node_features_batched(node_labels, NODE_EMBEDDING_DIM)
    print(f"  Shape: {node2vec.shape}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Free node_labels — no longer needed
    del node_labels
    gc.collect()

    # ========================================================
    # Step 3: Build edge type one-hot vectors
    # ========================================================
    n_edge_types = len(cfg["include_edge_type"])
    rel_onehot = torch.nn.functional.one_hot(
        torch.arange(n_edge_types), num_classes=n_edge_types
    ).float()
    rel2vec = {}
    for etype in cfg["include_edge_type"]:
        idx = rel2id[etype] - 1
        rel2vec[etype] = rel_onehot[idx]
    print(f"\n  Edge types: {n_edge_types}")

    # ========================================================
    # Step 4: Convert shards to TemporalData
    # ========================================================
    all_shards = sorted(labeled_dir.glob("labeled_shard*.parquet"))
    print(f"\nStep 4: Convert {len(all_shards)} shards to TemporalData...")

    shard_groups = {}
    for idx in cfg["train_shards"]:
        shard_groups[idx] = "train"
    for idx in cfg["val_shards"]:
        shard_groups[idx] = "val"
    for idx in cfg["test_shards"]:
        shard_groups[idx] = "test"

    # Pre-compute edge type vectors as numpy for fast access
    rel2vec_np = {k: v.numpy() for k, v in rel2vec.items()}
    msg_dim = NODE_EMBEDDING_DIM * 2 + n_edge_types

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

            # Map UUIDs to integer IDs (vectorized)
            src_ids = df["subject_uuid"].map(uuid_to_id)
            dst_ids = df["predicate_object_uuid"].map(uuid_to_id)

            valid = src_ids.notna() & dst_ids.notna()
            df = df[valid].reset_index(drop=True)
            src_ids = src_ids[valid].astype(int).values.copy()
            dst_ids = dst_ids[valid].astype(int).values.copy()

            # Handle reversed edges (vectorized)
            types_arr = df["type"].values
            for rev_type in reversed_types:
                rev_mask = types_arr == rev_type
                if rev_mask.any():
                    src_ids[rev_mask], dst_ids[rev_mask] = \
                        dst_ids[rev_mask].copy(), src_ids[rev_mask].copy()

            # Build messages vectorized: [src_feat, edge_onehot, dst_feat]
            timestamps = df["timestamp_nanos"].values.astype(np.int64)
            labels = df["label_broad"].values.astype(np.int8)

            n = len(df)
            msg_array = np.zeros((n, msg_dim), dtype=np.float32)

            # Source features
            msg_array[:, :NODE_EMBEDDING_DIM] = node2vec[src_ids]

            # Edge type features
            unique_types = np.unique(types_arr)
            for etype in unique_types:
                if etype in rel2vec_np:
                    type_mask = types_arr == etype
                    msg_array[type_mask, NODE_EMBEDDING_DIM:NODE_EMBEDDING_DIM+n_edge_types] = \
                        rel2vec_np[etype]

            # Destination features
            msg_array[:, NODE_EMBEDDING_DIM+n_edge_types:] = node2vec[dst_ids]

            all_src.append(torch.tensor(src_ids, dtype=torch.long))
            all_dst.append(torch.tensor(dst_ids, dtype=torch.long))
            all_t.append(torch.tensor(timestamps, dtype=torch.long))
            all_msg.append(torch.from_numpy(msg_array))
            all_labels.append(labels)

            print(f"    Shard {shard_idx}: {n:,} events "
                  f"(filtered {total_events - total_filtered:,})")

            del df, msg_array
            gc.collect()

        if not all_src:
            continue

        dataset_td = TemporalData(
            src=torch.cat(all_src),
            dst=torch.cat(all_dst),
            t=torch.cat(all_t),
            msg=torch.cat(all_msg),
        )

        sort_idx = dataset_td.t.argsort()
        dataset_td.src = dataset_td.src[sort_idx]
        dataset_td.dst = dataset_td.dst[sort_idx]
        dataset_td.t = dataset_td.t[sort_idx]
        dataset_td.msg = dataset_td.msg[sort_idx]

        labels_concat = np.concatenate(all_labels)[sort_idx.numpy()]

        torch.save(dataset_td, out_dir / f"{split}.TemporalData.pt")
        np.save(out_dir / f"{split}_labels.npy", labels_concat)

        n_attack = int(labels_concat.sum())
        print(f"\n  {split}: {len(dataset_td.src):,} edges "
              f"({total_filtered:,} after type filter from {total_events:,})")
        print(f"    Attack events: {n_attack:,} "
              f"({100*n_attack/len(labels_concat):.2f}%)")
        print(f"    msg shape: {dataset_td.msg.shape}")

        del all_src, all_dst, all_t, all_msg, all_labels, dataset_td
        gc.collect()

    meta = {
        "num_nodes": num_nodes,
        "uuid_to_id_size": len(uuid_to_id),
        "n_edge_types": n_edge_types,
        "include_edge_type": cfg["include_edge_type"],
        "msg_dim": msg_dim,
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
