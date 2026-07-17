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
                        choices=["theia", "trace", "cadets", "trace-1"])
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
    # Step 1: Gather all entity UUIDs (excluding sentinels) AND
    #         tally total_events & total_nnz for preallocation
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 1: Gather Entity UUIDs & Tally Counts")
    print(f"{'='*60}")

    t0 = time.time()
    subject_uuids = set()
    object_uuids = set()

    total_events = 0
    total_nnz = 0

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

        # ---- NEW: count total events and nnz while we have the df ----
        has_sub = df["subject_uuid"].notna() & ~df["subject_uuid"].isin(EXCLUDED_UUIDS)
        has_obj = df["predicate_object_uuid"].notna() & ~df["predicate_object_uuid"].isin(EXCLUDED_UUIDS)
        has_obj2 = df["predicate_object2_uuid"].notna() & ~df["predicate_object2_uuid"].isin(EXCLUDED_UUIDS)

        sizes = has_sub.astype(np.int8) + has_obj.astype(np.int8) + has_obj2.astype(np.int8)
        valid = sizes >= 2

        total_events += len(df)
        total_nnz += int(sizes[valid].sum())
        # -------------------------------------------------------------

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
    # Step 3: Build incidence list (COO format) with preallocation
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 3: Build Incidence (COO) – Preallocated")
    print(f"{'='*60}")

    t0 = time.time()
    shard_offsets = []

    # Preallocate the big arrays now that we know the final sizes
    # Use int32 for both indices and indptr to avoid silent upcast
    assert total_nnz < 2**31, "nnz too large for int32 — keep int64 here"
    csc_indices = np.empty(total_nnz, dtype=np.int32)
    csc_indptr = np.empty(total_events + 1, dtype=np.int32)
    csc_indptr[0] = 0
    nnz_cursor = 0
    he_cursor = 0

    # Preallocate label arrays
    y_broad = np.full(total_events, -1, dtype=np.int8)
    y_narrow = np.full(total_events, -1, dtype=np.int8)
    y_ioc = np.full(total_events, -1, dtype=np.int8)
    event_type_arr = np.full(total_events, -1, dtype=np.int16)
    timestamp_arr = np.full(total_events, -1, dtype=np.int64)

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

        # Map UUIDs to integer IDs (excluded UUIDs will map to -1)
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
            # Zero out degenerate entries so they contribute nothing
            subj_ids[~valid_he] = -1
            obj1_ids[~valid_he] = -1
            obj2_ids[~valid_he] = -1

        size_2_count += int((sizes == 2).sum())
        size_3_count += int((sizes == 3).sum())

        # Record shard offset
        shard_offsets.append((shard_idx, he_cursor, he_cursor + n,
                              int(valid_he.sum())))

        # Build valid entity array and sizes
        ent_array = np.column_stack([subj_ids, obj1_ids, obj2_ids])
        valid_mask = ent_array >= 0
        valid_mask[~valid_he] = False   # ignore degenerate rows
        shard_sizes = valid_mask.sum(axis=1)   # length n
        shard_indices = ent_array[valid_mask]  # length k

        # ---- Write into preallocated slices ----
        k = len(shard_indices)
        csc_indices[nnz_cursor : nnz_cursor + k] = shard_indices.astype(np.int32)
        csc_indptr[he_cursor + 1 : he_cursor + 1 + n] = (
            nnz_cursor + np.cumsum(shard_sizes, dtype=np.int32)
        )

        nnz_cursor += k
        he_cursor += n
        # -----------------------------------------

        # Fill label arrays
        if "label_broad" in df.columns:
            y_broad[he_cursor - n : he_cursor] = df["label_broad"].values.astype(np.int8)
        if "label_narrow" in df.columns:
            y_narrow[he_cursor - n : he_cursor] = df["label_narrow"].values.astype(np.int8)
        if "label_ioc" in df.columns:
            y_ioc[he_cursor - n : he_cursor] = df["label_ioc"].values.astype(np.int8)
        if "type" in df.columns:
            event_type_arr[he_cursor - n : he_cursor] = (
                df["type"].astype("category").cat.codes.values.astype(np.int16)
            )
        if "timestamp_nanos" in df.columns:
            timestamp_arr[he_cursor - n : he_cursor] = (
                df["timestamp_nanos"].values.astype(np.int64)
            )

        n_sz3 = int((sizes == 3).sum())
        print(f"    {n:,} events | size-2: {n - n_sz3:,} | "
              f"size-3: {n_sz3:,} ({100*n_sz3/n:.1f}%)")

        del df, subj_ids, obj1_ids, obj2_ids
        gc.collect()

    # ---- Sanity checks ----
    assert nnz_cursor == total_nnz, f"nnz mismatch: {nnz_cursor} != {total_nnz}"
    assert he_cursor == total_events, f"event count mismatch: {he_cursor} != {total_events}"

    num_hyperedges = total_events
    print(f"\n  Total hyperedges:  {num_hyperedges:,}")
    print(f"  Size-2:            {size_2_count:,} ({100*size_2_count/num_hyperedges:.1f}%)")
    print(f"  Size-3:            {size_3_count:,} ({100*size_3_count/num_hyperedges:.1f}%)")
    print(f"  Non-zero entries:  {len(csc_indices):,}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ---- Free uuid_to_id (no longer needed) ----
    del uuid_to_id
    gc.collect()

    # ---- Save labels and metadata NOW (before sparse matrix memory spike) ----
    labels_path = graph_dir / "hyperedge_labels.npz"
    np.savez_compressed(
        labels_path,
        y_broad=y_broad,
        y_narrow=y_narrow,
        y_ioc=y_ioc,
    )
    meta_path = graph_dir / "hyperedge_metadata.npz"
    np.savez_compressed(
        meta_path,
        event_type=event_type_arr,
        timestamp_nanos=timestamp_arr,
    )
    print(f"    hyperedge_labels.npz, hyperedge_metadata.npz saved")

    # ---- Free label arrays to reclaim memory ----
    del y_broad, y_narrow, y_ioc, event_type_arr, timestamp_arr
    gc.collect()

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
    # Step 4: Build sparse incidence matrix (CSC only, no CSR conversion)
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 4: Build Sparse Incidence Matrix (CSC)")
    print(f"{'='*60}")

    t0 = time.time()

    csc_data = np.ones(total_nnz, dtype=np.int8)
    H_csc = sparse.csc_matrix(
        (csc_data, csc_indices, csc_indptr),
        shape=(num_entities, num_hyperedges),
    )
    # Free the raw arrays now that they're referenced inside H_csc
    del csc_data, csc_indices, csc_indptr
    gc.collect()

    # Save CSC directly (no CSR conversion)
    incidence_path = graph_dir / "incidence.npz"
    sparse.save_npz(incidence_path, H_csc)  # save_npz handles CSC/CSR transparently

    print(f"  Shape: {H_csc.shape}")
    print(f"  Non-zeros: {H_csc.nnz:,}")
    print(f"  Density: {H_csc.nnz / (H_csc.shape[0] * H_csc.shape[1]):.2e}")
    print(f"  File: {incidence_path.name} "
          f"({incidence_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ============================================================
    # Step 5: Hypergraph statistics (compute directly on CSC)
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 5: Hypergraph Statistics")
    print(f"{'='*60}")

    # Node degrees: sum over rows (axis=1) works on CSC, but may be slow.
    # Faster: use bincount on row indices (csc_indices) – but we no longer have
    # the raw arrays; we can get them from H_csc.indices, H_csc.indptr.
    # We'll use the built-in method; for large matrices it's fine.
    node_degrees = np.array(H_csc.sum(axis=1)).flatten()

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

    nnz_count = H_csc.nnz
    del H_csc, node_degrees
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
    print(f"  Non-zero entries:{nnz_count:,}")
    print(f"\n  Null UUID filtered: ✓")
    print(f"\n  Files saved to {graph_dir}/:")
    print(f"    entity_vocab.npz    ({vocab_path.stat().st_size/1e6:.1f} MB)")
    print(f"    incidence.npz       ({incidence_path.stat().st_size/1e6:.1f} MB)")
    print(f"    shard_offsets.npz")


if __name__ == "__main__":
    main()
