"""
Hyperedge Mining: Extract concrete attack hyperedges from DARPA TC E3 Theia.

Maps multi-entity attack steps to natural hyperedge representations.
For each temporal window containing attack events, identifies clusters
of events that share entities (subjects/objects) and co-occur in time.

A hyperedge = a group of events that:
  1. Share at least one entity (subject or object UUID)
  2. Occur within epsilon seconds of each other
  3. Involve multiple distinct entity types

Usage:
    python -m src.coordination.hyperedge_mining
"""

import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.coordination.ground_truth_e3 import label_events


DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3" / "theia"
)


def find_hyperedges(events_chunk: pd.DataFrame, subjects_df: pd.DataFrame,
                    objects_df: pd.DataFrame, epsilon_sec: float = 5.0,
                    min_size: int = 3, max_size: int = 10) -> list[dict]:
    """
    Find natural hyperedges in a chunk of events.

    Algorithm:
        1. For each event e, find all events within epsilon_sec that share
           at least one entity (subject_uuid or predicate_object_uuid).
        2. Merge overlapping clusters.
        3. Return clusters of size >= min_size as hyperedges.
    """
    if len(events_chunk) == 0:
        return []

    ts = events_chunk["timestamp_nanos"].values.astype(np.float64)
    eps_ns = epsilon_sec * 1e9

    # Build entity -> event index maps
    sub_to_events = defaultdict(set)
    obj_to_events = defaultdict(set)

    for idx, row in events_chunk.iterrows():
        sub = row.get("subject_uuid")
        obj = row.get("predicate_object_uuid")
        if pd.notna(sub):
            sub_to_events[sub].add(idx)
        if pd.notna(obj):
            obj_to_events[obj].add(idx)

    # Union-Find for merging overlapping clusters
    parent = {idx: idx for idx in events_chunk.index}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # For each entity, merge events that share it AND are within epsilon
    for entity_map in [sub_to_events, obj_to_events]:
        for entity, event_idxs in entity_map.items():
            event_list = sorted(event_idxs)
            for i in range(len(event_list)):
                for j in range(i + 1, len(event_list)):
                    ti = ts[events_chunk.index.get_loc(event_list[i])]
                    tj = ts[events_chunk.index.get_loc(event_list[j])]
                    if abs(tj - ti) <= eps_ns:
                        union(event_list[i], event_list[j])

    # Collect clusters
    clusters = defaultdict(list)
    for idx in events_chunk.index:
        clusters[find(idx)].append(idx)

    # Filter by size and build hyperedge descriptors
    # Build lookup maps for enrichment
    sub_lookup = {}
    if len(subjects_df) > 0:
        sub_lookup = subjects_df.set_index("uuid")[["process_path", "cmd_line"]].to_dict("index")

    obj_lookup = {}
    if len(objects_df) > 0:
        cols_avail = [c for c in ["filename", "object_type", "remote_address", "remote_port"] if c in objects_df.columns]
        if cols_avail:
            obj_lookup = objects_df.set_index("uuid")[cols_avail].to_dict("index")

    hyperedges = []
    for root, members in clusters.items():
        if len(members) < min_size or len(members) > max_size:
            continue

        member_events = events_chunk.loc[members]
        timestamps = member_events["timestamp_nanos"].values.astype(np.float64)
        duration_sec = (timestamps.max() - timestamps.min()) / 1e9

        # Collect entities
        subjects = set(member_events["subject_uuid"].dropna().unique())
        objects = set(member_events["predicate_object_uuid"].dropna().unique())

        # Enrich with names
        subject_names = []
        for s in subjects:
            info = sub_lookup.get(s, {})
            path = info.get("process_path", "")
            cmd = info.get("cmd_line", "")
            name = path.split("/")[-1] if path else (cmd[:30] if cmd else s[:12])
            subject_names.append(name)

        object_names = []
        for o in objects:
            info = obj_lookup.get(o, {})
            fname = info.get("filename", "")
            otype = info.get("object_type", "")
            raddr = info.get("remote_address", "")
            if fname:
                name = fname.split("/")[-1] if "/" in str(fname) else str(fname)
            elif raddr:
                rport = info.get("remote_port", "")
                name = f"{raddr}:{rport}" if rport else str(raddr)
            else:
                name = f"[{otype}]" if otype else o[:12]
            object_names.append(name)

        event_types = member_events["type"].value_counts().to_dict()

        hyperedges.append({
            "size": len(members),
            "n_subjects": len(subjects),
            "n_objects": len(objects),
            "n_entities": len(subjects) + len(objects),
            "duration_sec": round(duration_sec, 3),
            "event_types": event_types,
            "subject_names": subject_names,
            "object_names": object_names[:10],  # Limit for readability
            "timestamp_start": pd.to_datetime(timestamps.min(), unit="ns"),
            "timestamp_end": pd.to_datetime(timestamps.max(), unit="ns"),
        })

    return hyperedges


def main():
    print("="*60)
    print("HYPEREDGE MINING: Attack Step Analysis")
    print("="*60)

    # Load data
    print("\nLoading data...")
    events_df = pd.read_parquet(DATA_DIR / "events.parquet")
    subjects_df = pd.read_parquet(DATA_DIR / "subjects.parquet")
    objects_df = pd.read_parquet(DATA_DIR / "objects.parquet")
    print(f"  Events: {len(events_df):,}")

    # Label events
    print("Labeling events...")
    event_labels = label_events(events_df, subjects_df, objects_df)
    n_attack = (event_labels == 1).sum()
    print(f"  Attack events: {n_attack:,}")

    # Focus on attack events only
    attack_events = events_df[event_labels == 1].copy()
    print(f"\nAnalyzing {len(attack_events):,} attack events...")

    # Sample attack windows to mine hyperedges (use 30-second windows)
    ts = attack_events["timestamp_nanos"].values.astype(np.float64)
    t_min, t_max = ts.min(), ts.max()
    window_ns = int(30 * 1e9)  # 30-second windows

    windows = np.arange(t_min, t_max + 1, window_ns)
    print(f"  Analyzing {len(windows)} x 30-second windows...")

    all_hyperedges = []
    windows_with_hyperedges = 0

    for i, w_start in enumerate(windows):
        w_end = w_start + window_ns
        mask = (attack_events["timestamp_nanos"] >= w_start) & \
               (attack_events["timestamp_nanos"] < w_end)
        chunk = attack_events[mask]

        if len(chunk) < 3:
            continue

        # Subsample large windows to avoid O(n^2) blowup
        if len(chunk) > 200:
            chunk = chunk.sample(200, random_state=42)

        hes = find_hyperedges(chunk, subjects_df, objects_df,
                              epsilon_sec=5.0, min_size=3, max_size=8)
        if hes:
            windows_with_hyperedges += 1
            all_hyperedges.extend(hes)

        if i > 0 and i % 500 == 0:
            print(f"    Window {i}/{len(windows)}: {len(all_hyperedges)} hyperedges found")

    print(f"\n  Total hyperedges found: {len(all_hyperedges)}")
    print(f"  Windows with hyperedges: {windows_with_hyperedges}")

    if not all_hyperedges:
        print("  No hyperedges found! Trying larger epsilon...")
        return

    # ============================================================
    # Statistics
    # ============================================================
    sizes = [h["size"] for h in all_hyperedges]
    n_entities = [h["n_entities"] for h in all_hyperedges]
    durations = [h["duration_sec"] for h in all_hyperedges]

    print(f"\n{'='*60}")
    print("HYPEREDGE STATISTICS")
    print(f"{'='*60}")
    print(f"  Count:              {len(all_hyperedges)}")
    print(f"  Size (events):      mean={np.mean(sizes):.1f}, "
          f"min={min(sizes)}, max={max(sizes)}, "
          f"median={np.median(sizes):.0f}")
    print(f"  Entities per HE:    mean={np.mean(n_entities):.1f}, "
          f"min={min(n_entities)}, max={max(n_entities)}")
    print(f"  Duration (sec):     mean={np.mean(durations):.2f}, "
          f"max={max(durations):.2f}")

    # Event type distribution across hyperedges
    type_counts = defaultdict(int)
    for h in all_hyperedges:
        for t, c in h["event_types"].items():
            type_counts[t] += c

    print(f"\n  Event types in hyperedges:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {t:30s} {c:>8,}")

    # ============================================================
    # Concrete Examples
    # ============================================================
    print(f"\n{'='*60}")
    print("CONCRETE HYPEREDGE EXAMPLES")
    print(f"{'='*60}")

    # Sort by entity diversity (most interesting first)
    interesting = sorted(all_hyperedges,
                         key=lambda h: (h["n_entities"], h["size"]),
                         reverse=True)

    for i, he in enumerate(interesting[:10]):
        print(f"\n--- Hyperedge {i+1} ---")
        print(f"  Size: {he['size']} events, "
              f"{he['n_subjects']} subjects, {he['n_objects']} objects")
        print(f"  Duration: {he['duration_sec']}s")
        print(f"  Time: {he['timestamp_start']} → {he['timestamp_end']}")
        print(f"  Event types: {he['event_types']}")
        print(f"  Subjects: {he['subject_names']}")
        print(f"  Objects: {he['object_names'][:5]}")

    # ============================================================
    # Pairwise vs Hyperedge comparison
    # ============================================================
    print(f"\n{'='*60}")
    print("PAIRWISE vs HYPEREDGE DECOMPOSITION")
    print(f"{'='*60}")

    # For the top example, show how pairwise edges lose context
    if interesting:
        he = interesting[0]
        n_events = he["size"]
        n_ent = he["n_entities"]

        # A pairwise graph of this hyperedge: at most n_events edges
        # A hyperedge captures all n_events in ONE structure
        n_pairwise_edges = n_events  # (subject, type, object) per event
        print(f"\n  Example hyperedge: {n_events} events, {n_ent} entities")
        print(f"  Pairwise decomposition: {n_pairwise_edges} separate edges")
        print(f"  → Loses the joint temporal context across all {n_events} events")
        print(f"  → A GNN must aggregate {n_pairwise_edges} edges to reconstruct")
        print(f"  → Our hyperedge captures this in ONE representation")


if __name__ == "__main__":
    main()
