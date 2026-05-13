"""
Entity Vocabulary & Incidence Graph Construction.

Builds the structural skeleton of the provenance hypergraph:
    1. Gathers all entity UUIDs across all shards
    2. Filters out the null UUID sentinel (placeholder for missing obj2)
    3. Assigns contiguous integer IDs
    4. Builds sparse incidence matrix in COO/CSR format
    5. Computes and validates hypergraph statistics

Usage:
    python -m src.pipeline.build_graph --dataset theia
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3"
)

# Sentinel UUIDs to exclude from entity vocabulary.
# These are CDM placeholders, not real system entities.
EXCLUDED_UUIDS = {
    "00000000-0000-0000-0000-000000000000",  # Null UUID (97.2% of obj2)
}


def main():
    parser = argparse.ArgumentParser(
        description="Build entity vocabulary and incidence graph")
    parser.add_argument("--dataset", default="theia",
                        choices=["theia", "trace"])
    args = parser.parse_args()

    labeled_dir = DATA_ROOT / args.dataset / "labeled"
    graph_dir = DATA_ROOT / args.dataset / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(labeled_dir.glob("labeled_shard*.parquet"))
    n_shards = len(shard_files)

    print("=" * 60)
    print(f"BUILD GRAPH: {args.dataset.upper()}")
    print("=" * 60)
    print(f"  Shards: {n_shards}")
    print(f"  Excluded UUIDs: {EXCLUDED_UUIDS}")

    # ============================================================
    # Step 1: Gather all entity UUIDs (excluding sentinels)
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 1: Gather Entity UUIDs")
    print(f"{'='*60}")

    t0 = time.time()
    subject_uuids = set()
    object_uuids = set()

    for f in shard_files:
        shard_name = f.stem
        print(f"  Scanning {shard_name}...")

        df = pd.read_parquet(f, columns=[
            "subject_uuid", "predicate_object_uuid",
            "predicate_object2_uuid",
        ])

        subs = df["subject_uuid"].dropna().unique()
        subject_uuids.update(s for s in subs if s not in EXCLUDED_UUIDS)

        objs = df["predicate_object_uuid"].dropna().unique()
        object_uuids.update(o for o in objs if o not in EXCLUDED_UUIDS)

        obj2s = df["predicate_object2_uuid"].dropna().unique()
        object_uuids.update(o for o in obj2s if o not in EXCLUDED_UUIDS)

        del df
        gc.collect()

    overlap = subject_uuids & object_uuids
    all_uuids = subject_uuids | object_uuids

    print(f"\n  Subjects:  {len(subject_uuids):,}")
    print(f"  Objects:   {len(object_uuids):,}")
    print(f"  Overlap:   {len(overlap):,}")
    print(f"  Total:     {len(all_uuids):,}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ============================================================
    # Step 2: Assign integer IDs
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 2: Assign Integer IDs")
    print(f"{'='*60}")

    sorted_uuids = sorted(all_uuids)
    uuid_to_id = {uuid: i for i, uuid in enumerate(sorted_uuids)}
    num_entities = len(uuid_to_id)

    # Entity type: 0 = subject only, 1 = object only, 2 = both
    entity_type = np.zeros(num_entities, dtype=np.int8)
    for uuid, idx in uuid_to_id.items():
        is_sub = uuid in subject_uuids
        is_obj = uuid in object_uuids
        if is_sub and is_obj:
            entity_type[idx] = 2
        elif is_obj:
            entity_type[idx] = 1

    vocab_path = graph_dir / "entity_vocab.npz"
    np.savez(
        vocab_path,
        uuids=np.array(sorted_uuids, dtype=object),
        entity_type=entity_type,
        num_entities=num_entities,
    )

    type_names = {0: "subject_only", 1: "object_only", 2: "both"}
    for t_val in [0, 1, 2]:
        cnt = int((entity_type == t_val).sum())
        print(f"  {type_names[t_val]:15s}: {cnt:,}")
    print(f"  Total entities: {num_entities:,}")
    print(f"  Saved to {vocab_path.name}")

    del subject_uuids, object_uuids, overlap, all_uuids, sorted_uuids
    gc.collect()

    # ============================================================
    # Step 3: Build incidence list (COO format)
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 3: Build Incidence (COO)")
    print(f"{'='*60}")

    t0 = time.time()
    he_global_offset = 0
    shard_offsets = []

    all_he_indices = []
    all_ent_indices = []
    all_labels_broad = []
    all_labels_narrow = []
    all_labels_ioc = []
    all_event_types = []
    all_timestamps = []
    total_events = 0

    # Track hyperedge sizes (will vary now: 2 or 3)
    size_2_count = 0
    size_3_count = 0

    for f in shard_files:
        shard_name = f.stem
        shard_idx = int(shard_name.replace("labeled_shard", ""))
        print(f"  Processing {shard_name}...")

        LABEL_COLS = ["label_broad", "label_narrow", "label_ioc",
                      "type", "timestamp_nanos"]
        available = pd.read_parquet(f, columns=None).columns.tolist()
        load_cols = ["subject_uuid", "predicate_object_uuid",
                     "predicate_object2_uuid"] + \
                    [c for c in LABEL_COLS if c in available]
        df = pd.read_parquet(f, columns=load_cols)
        n = len(df)

        # Map UUIDs to integer IDs (excluded UUIDs will map to NaN)
        def map_col(series):
            return np.fromiter(
                (uuid_to_id.get(u, -1) for u in series.fillna("")),
                dtype=np.int64, count=len(series)
            )

        subj_ids = map_col(df["subject_uuid"])
        obj1_ids = map_col(df["predicate_object_uuid"])
        obj2_ids = map_col(df["predicate_object2_uuid"])

        # Count hyperedge sizes
        has_sub = subj_ids >= 0
        has_obj = obj1_ids >= 0
        has_obj2 = obj2_ids >= 0
        sizes = has_sub.astype(int) + has_obj.astype(int) + has_obj2.astype(int)

        # Filter out degenerate size-0/1 hyperedges
        valid_he = sizes >= 2
        n_degenerate = int((~valid_he).sum())
        if n_degenerate > 0:
            print(f"    ⚠ Filtering {n_degenerate} degenerate hyperedges "
                  f"(size < 2)")
            # Zero out degenerate entries so they don't get COO entries
            subj_ids[~valid_he] = -1
            obj1_ids[~valid_he] = -1
            obj2_ids[~valid_he] = -1

        size_2_count += int((sizes == 2).sum())
        size_3_count += int((sizes == 3).sum())

        # Use n (total events) as offset unit — degenerate events
        # still occupy an index slot (labels array aligns with events)
        he_ids = np.arange(he_global_offset, he_global_offset + n,
                           dtype=np.int64)
        shard_offsets.append((shard_idx, he_global_offset,
                              he_global_offset + n,
                              int(valid_he.sum())))  # valid count for reference

        # Build COO entries (skip sentinel -1)
        for ent_col in [subj_ids, obj1_ids, obj2_ids]:
            valid = ent_col >= 0
            all_he_indices.append(he_ids[valid])
            all_ent_indices.append(ent_col[valid])

        he_global_offset += n
        total_events += n
        n_sz3 = int((sizes == 3).sum())
        print(f"    {n:,} events | size-2: {n - n_sz3:,} | "
              f"size-3: {n_sz3:,} ({100*n_sz3/n:.1f}%)")

        for col, store in [
            ("label_broad",    all_labels_broad),
            ("label_narrow",   all_labels_narrow),
            ("label_ioc",      all_labels_ioc),
        ]:
            arr = df[col].values.astype(np.int8) if col in df.columns \
                  else np.full(n, -1, dtype=np.int8)
            store.append(arr)

        if "type" in df.columns:
            all_event_types.append(df["type"].astype("category").cat.codes.values.astype(np.int16))
        if "timestamp_nanos" in df.columns:
            all_timestamps.append(df["timestamp_nanos"].values.astype(np.int64))

        del df, subj_ids, obj1_ids, obj2_ids, he_ids
        gc.collect()

    # Concatenate
    he_indices = np.concatenate(all_he_indices)
    ent_indices = np.concatenate(all_ent_indices)
    del all_he_indices, all_ent_indices
    gc.collect()

    num_hyperedges = total_events
    print(f"\n  Total hyperedges:  {num_hyperedges:,}")
    print(f"  Size-2:            {size_2_count:,} ({100*size_2_count/num_hyperedges:.1f}%)")
    print(f"  Size-3:            {size_3_count:,} ({100*size_3_count/num_hyperedges:.1f}%)")
    print(f"  COO entries:       {len(he_indices):,}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Save shard offsets
    offsets_path = graph_dir / "shard_offsets.npz"
    np.savez(
        offsets_path,
        shard_idx=np.array([s[0] for s in shard_offsets]),
        start=np.array([s[1] for s in shard_offsets]),
        end=np.array([s[2] for s in shard_offsets]),
        n_valid=np.array([s[3] for s in shard_offsets]),
    )

    # ============================================================
    # Step 4: Build sparse incidence matrix
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 4: Build Sparse Incidence Matrix")
    print(f"{'='*60}")

    t0 = time.time()

    H_coo = sparse.coo_matrix(
        (np.ones(len(he_indices), dtype=np.int8),
         (ent_indices, he_indices)),
        shape=(num_entities, num_hyperedges),
    )

    H_csr = H_coo.tocsr()
    del H_coo
    gc.collect()

    incidence_path = graph_dir / "incidence.npz"
    sparse.save_npz(incidence_path, H_csr)
    labels_path = graph_dir / "hyperedge_labels.npz"
    np.savez_compressed(
        labels_path,
        y_broad=np.concatenate(all_labels_broad),
        y_narrow=np.concatenate(all_labels_narrow) if all_labels_narrow else np.array([]),
        y_ioc=np.concatenate(all_labels_ioc) if all_labels_ioc else np.array([]),
    )

    meta_path = graph_dir / "hyperedge_metadata.npz"
    np.savez_compressed(
        meta_path,
        event_type=np.concatenate(all_event_types) if all_event_types else np.array([]),
        timestamp_nanos=np.concatenate(all_timestamps) if all_timestamps else np.array([]),
    )
    print(f"    hyperedge_labels.npz, hyperedge_metadata.npz saved")

    print(f"  Shape: {H_csr.shape}")
    print(f"  Non-zeros: {H_csr.nnz:,}")
    print(f"  Density: {H_csr.nnz / (H_csr.shape[0] * H_csr.shape[1]):.2e}")
    print(f"  File: {incidence_path.name} "
          f"({incidence_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ============================================================
    # Step 5: Hypergraph statistics
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 5: Hypergraph Statistics")
    print(f"{'='*60}")

    # Node degrees
    node_degrees = np.array(H_csr.sum(axis=1)).flatten()

    print(f"\n  Node degree statistics:")
    print(f"    Mean:   {node_degrees.mean():.1f}")
    print(f"    Median: {np.median(node_degrees):.0f}")
    print(f"    Max:    {node_degrees.max():,}")
    print(f"    P95:    {np.percentile(node_degrees, 95):.0f}")
    print(f"    P99:    {np.percentile(node_degrees, 99):.0f}")

    # Top-degree entities
    top_k = 10
    top_idx = np.argsort(node_degrees)[-top_k:][::-1]
    vocab_data = np.load(vocab_path, allow_pickle=True)
    all_uuids_arr = vocab_data["uuids"]

    print(f"\n  Top {top_k} highest-degree entities:")
    print(f"    {'Rank':>4s} {'Degree':>10s} {'Type':>12s} {'UUID':>40s}")
    print(f"    {'─'*70}")
    for rank, idx in enumerate(top_idx):
        deg = int(node_degrees[idx])
        etype = type_names[int(entity_type[idx])]
        uuid = str(all_uuids_arr[idx])[:36]
        print(f"    {rank+1:>4d} {deg:>10,} {etype:>12s} {uuid:>40s}")

    # Hyperedge sizes
    print(f"\n  Hyperedge size distribution:")
    n_degen = num_hyperedges - size_2_count - size_3_count
    if n_degen > 0:
        print(f"    size <2: {n_degen:,} (filtered)")
    print(f"    size 2:  {size_2_count:,} ({100*size_2_count/num_hyperedges:.1f}%)")
    print(f"    size 3:  {size_3_count:,} ({100*size_3_count/num_hyperedges:.1f}%)")

    # Entity type breakdown
    print(f"\n  Entity type breakdown:")
    for t_val in [0, 1, 2]:
        mask = entity_type == t_val
        cnt = int(mask.sum())
        if cnt > 0:
            avg_deg = node_degrees[mask].mean()
            max_deg = int(node_degrees[mask].max())
            print(f"    {type_names[t_val]:15s}: {cnt:>10,} entities, "
                  f"avg degree={avg_deg:.1f}, max={max_deg:,}")

    del H_csr, node_degrees
    gc.collect()

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Entities:        {num_entities:,}")
    print(f"  Hyperedges:      {num_hyperedges:,}")
    print(f"    Size-2:        {size_2_count:,} ({100*size_2_count/num_hyperedges:.1f}%)")
    print(f"    Size-3:        {size_3_count:,} ({100*size_3_count/num_hyperedges:.1f}%)")
    print(f"  COO entries:     {len(he_indices):,}")
    print(f"\n  Null UUID filtered: ✓")
    print(f"\n  Files saved to {graph_dir}/:")
    print(f"    entity_vocab.npz    ({vocab_path.stat().st_size/1e6:.1f} MB)")
    print(f"    incidence.npz       ({incidence_path.stat().st_size/1e6:.1f} MB)")
    print(f"    shard_offsets.npz")


if __name__ == "__main__":
    main()
