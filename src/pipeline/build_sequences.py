"""
Build Per-Subject Temporal Sequences.

Produces chronologically ordered event sequences for each subject entity.
These feed the Mamba/SSM temporal encoder.

Output format (CSR-style):
    subject_seq_he[offset[i]:offset[i+1]] = global hyperedge IDs
        for subject i, sorted by timestamp.

Usage:
    python -m src.pipeline.build_sequences --dataset theia
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3"
)


def main():
    parser = argparse.ArgumentParser(
        description="Build per-subject temporal sequences")
    parser.add_argument("--dataset", default="theia",
                        choices=["theia", "trace"])
    args = parser.parse_args()

    labeled_dir = DATA_ROOT / args.dataset / "labeled"
    graph_dir = DATA_ROOT / args.dataset / "graph"

    shard_files = sorted(labeled_dir.glob("labeled_shard*.parquet"))

    print("=" * 60)
    print(f"BUILD SEQUENCES: {args.dataset.upper()}")
    print("=" * 60)

    # Load entity vocabulary
    vocab_data = np.load(graph_dir / "entity_vocab.npz", allow_pickle=True)
    uuids_arr = vocab_data["uuids"]
    entity_type = vocab_data["entity_type"]
    uuid_to_id = {str(u): i for i, u in enumerate(uuids_arr)}
    num_entities = len(uuid_to_id)
    print(f"  Entities: {num_entities:,}")

    # Load shard offsets
    offsets_data = np.load(graph_dir / "shard_offsets.npz")
    shard_starts = {int(s): int(st)
                    for s, st in zip(offsets_data["shard_idx"],
                                     offsets_data["start"])}

    # ============================================================
    # Step 1: Gather (subject_int, timestamp_nanos, global_he_id)
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 1: Gather Subject Events")
    print(f"{'='*60}")

    t0 = time.time()

    # Collect arrays per shard, then concatenate
    all_subj_ids = []
    all_timestamps = []
    all_he_ids = []
    total = 0

    for f in shard_files:
        shard_name = f.stem
        shard_idx = int(shard_name.replace("labeled_shard", ""))
        he_offset = shard_starts[shard_idx]
        print(f"  Loading {shard_name} (offset={he_offset:,})...")

        df = pd.read_parquet(f, columns=[
            "subject_uuid", "timestamp_nanos",
        ])
        n = len(df)

        # Map subject UUIDs to integer IDs (vectorized)
        df = df.reset_index(drop=True)
        subj_ids = df["subject_uuid"].map(uuid_to_id)
        valid = subj_ids.notna().values
        subj_ids = subj_ids[valid].astype(np.int64).values
        ts = df.loc[valid, "timestamp_nanos"].values.astype(np.int64)
        he_ids = np.arange(he_offset, he_offset + n,
                           dtype=np.int64)[valid]

        all_subj_ids.append(subj_ids)
        all_timestamps.append(ts)
        all_he_ids.append(he_ids)
        total += len(subj_ids)

        n_dropped = n - len(subj_ids)
        if n_dropped > 0:
            print(f"    ⚠ Dropped {n_dropped} events (unmapped subjects)")
        print(f"    {len(subj_ids):,} events")

        del df, subj_ids, ts, he_ids
        gc.collect()

    # Concatenate all
    print(f"\n  Concatenating {total:,} events...")
    subj_ids = np.concatenate(all_subj_ids)
    timestamps = np.concatenate(all_timestamps)
    he_ids = np.concatenate(all_he_ids)
    del all_subj_ids, all_timestamps, all_he_ids
    gc.collect()

    print(f"  Memory: {(subj_ids.nbytes + timestamps.nbytes + he_ids.nbytes) / 1e9:.1f} GB")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ============================================================
    # Step 2: Sort by (subject_id, timestamp)
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 2: Sort by (subject, timestamp)")
    print(f"{'='*60}")

    t0 = time.time()
    # Lexicographic sort: primary key = subject_id, secondary = timestamp
    sort_idx = np.lexsort((timestamps, subj_ids))
    subj_ids = subj_ids[sort_idx]
    timestamps = timestamps[sort_idx]
    he_ids = he_ids[sort_idx]
    del sort_idx
    gc.collect()
    print(f"  Sorted {total:,} events in {time.time()-t0:.1f}s")

    # ============================================================
    # Step 3: Build CSR-style offset array
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 3: Build Offset Array")
    print(f"{'='*60}")

    t0 = time.time()

    # Find where subject IDs change
    changes = np.where(np.diff(subj_ids) != 0)[0] + 1
    # Subject IDs at each group start
    group_starts = np.concatenate([[0], changes])
    group_subj_ids = subj_ids[group_starts]

    # Build offset array for ALL entities (not just those with events)
    # offset[i] = start index for entity i in he_ids
    # offset[num_entities] = total length
    # np.bincount counts events per subject_id directly
    seq_lens_sparse = np.bincount(subj_ids, minlength=num_entities).astype(np.int64)
    offset = np.zeros(num_entities + 1, dtype=np.int64)
    offset[1:] = np.cumsum(seq_lens_sparse)

    n_subjects_with_events = len(group_starts)
    print(f"  Subjects with events: {n_subjects_with_events:,}")
    print(f"  Total entities:       {num_entities:,}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ============================================================
    # Step 4: Save
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 4: Save Sequences")
    print(f"{'='*60}")

    seq_path = graph_dir / "subject_sequences.npz"
    np.savez_compressed(
        seq_path,
        he_ids=he_ids,
        timestamps=timestamps,
        offset=offset,
        num_entities=num_entities,
        total_events=total,
    )
    print(f"  Saved to {seq_path.name} "
          f"({seq_path.stat().st_size / 1e6:.0f} MB)")

    # ============================================================
    # Step 5: Validate sequence statistics
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 5: Sequence Statistics")
    print(f"{'='*60}")

    # Compute per-subject sequence lengths
    seq_lens = np.diff(offset)
    nonzero_lens = seq_lens[seq_lens > 0]

    print(f"\n  All entities:")
    print(f"    With events:   {len(nonzero_lens):,}")
    print(f"    Without events:{(seq_lens == 0).sum():,}")

    print(f"\n  Sequence length (entities with events):")
    print(f"    Mean:   {nonzero_lens.mean():.0f}")
    print(f"    Median: {np.median(nonzero_lens):.0f}")
    print(f"    Min:    {nonzero_lens.min():,}")
    print(f"    Max:    {nonzero_lens.max():,}")
    print(f"    P95:    {np.percentile(nonzero_lens, 95):.0f}")
    print(f"    P99:    {np.percentile(nonzero_lens, 99):.0f}")
    print(f"    >1K:    {(nonzero_lens > 1_000).sum():,}")
    print(f"    >10K:   {(nonzero_lens > 10_000).sum():,}")
    print(f"    >100K:  {(nonzero_lens > 100_000).sum():,}")
    print(f"    >1M:    {(nonzero_lens > 1_000_000).sum():,}")

    # Breakdown by entity type
    print(f"\n  By entity type:")
    type_names = {0: "subject_only", 1: "object_only", 2: "both"}
    for t_val in [0, 1, 2]:
        mask = entity_type == t_val
        t_lens = seq_lens[mask]
        nonzero = t_lens[t_lens > 0]
        if len(nonzero) > 0:
            print(f"    {type_names[t_val]:15s}: "
                  f"{len(nonzero):,} subjects, "
                  f"mean={nonzero.mean():.0f}, "
                  f"max={nonzero.max():,}")

    # ============================================================
    # Step 6: Spot-check attack sequences
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 6: Spot-Check Attack Sequences")
    print(f"{'='*60}")

    # Load ground truth to identify attack subjects
    from src.data.ground_truth import load_ground_truth, build_attack_subject_uuids

    gt = load_ground_truth(args.dataset)
    subjects_df = pd.read_parquet(
        DATA_ROOT / args.dataset / "subjects.parquet")

    attack_sub_uuids = build_attack_subject_uuids(subjects_df, gt)
    attack_sub_ids = [uuid_to_id[u] for u in attack_sub_uuids
                      if u in uuid_to_id]
    del subjects_df

    # Find top-3 attack subjects by sequence length
    attack_lens = [(sid, seq_lens[sid]) for sid in attack_sub_ids
                   if seq_lens[sid] > 0]
    attack_lens.sort(key=lambda x: -x[1])

    print(f"\n  Attack subjects: {len(attack_sub_ids):,}")
    print(f"  Attack subjects with events: {len(attack_lens):,}")

    # Build global HE ID → (shard_idx, local_idx) lookup
    print(f"\n  Building shard lookup table...")
    shard_idx_arr = offsets_data["shard_idx"]
    shard_start_arr = offsets_data["start"]
    shard_end_arr = offsets_data["end"]
    # Sort by start for binary search
    shard_order = np.argsort(shard_start_arr)
    shard_starts_sorted = shard_start_arr[shard_order]
    shard_idx_sorted = shard_idx_arr[shard_order]

    def resolve_he(global_he_id):
        """Map global HE ID → (shard_idx, local_index)."""
        pos = np.searchsorted(shard_starts_sorted, global_he_id, side="right") - 1
        s_idx = int(shard_idx_sorted[pos])
        local_idx = int(global_he_id - shard_starts_sorted[pos])
        return s_idx, local_idx

    # Cache loaded shards to avoid re-reading
    shard_cache = {}

    def get_event_info(global_he_id):
        """Fetch event type, label, timestamp from the correct shard."""
        s_idx, local_idx = resolve_he(global_he_id)
        if s_idx not in shard_cache:
            shard_cache[s_idx] = pd.read_parquet(
                labeled_dir / f"labeled_shard{s_idx}.parquet",
                columns=["type", "label_broad", "timestamp"],
            )
        df = shard_cache[s_idx]
        return (
            df.iloc[local_idx]["type"],
            int(df.iloc[local_idx]["label_broad"]),
            df.iloc[local_idx]["timestamp"],
        )

    for rank, (sid, slen) in enumerate(attack_lens[:3]):
        uuid = str(uuids_arr[sid])[:36]
        print(f"\n  ── Attack Subject #{rank+1}: {uuid} ──")
        print(f"     Sequence length: {slen:,}")

        # Get first 50 events
        start = offset[sid]
        end = min(offset[sid] + 50, offset[sid + 1])
        seq_he = he_ids[start:end]
        seq_ts = timestamps[start:end]

        print(f"     First {end - start} events:")
        print(f"     {'#':>4s} {'GlobalHE':>12s} {'Timestamp':>26s} "
              f"{'Type':>30s} {'Label':>6s}")
        print(f"     {'─'*82}")

        for i, (he, ts) in enumerate(zip(seq_he, seq_ts)):
            etype, label, ts_dt = get_event_info(he)
            label_str = "ATK" if label == 1 else "BEN" if label == 0 else "?"
            print(f"     {i:>4d} {he:>12,} {str(ts_dt):>26s} "
                  f"{etype:>30s} {label_str:>6s}")

    # Free shard cache
    del shard_cache
    gc.collect()

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Total events in sequences: {total:,}")
    print(f"  Subjects with events:      {n_subjects_with_events:,}")
    print(f"  Attack subjects:           {len(attack_lens):,}")
    print(f"  Max sequence length:       {nonzero_lens.max():,}")
    print(f"  Median sequence length:    {np.median(nonzero_lens):.0f}")
    print(f"  Saved: {seq_path.name}")


if __name__ == "__main__":
    main()
