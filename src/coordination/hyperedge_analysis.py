"""
Hyperedge Analysis for DARPA TC E3 Theia (fast, windowed).

Uses windowed Union-Find: divide into 30s windows, then build
hyperedges within each window. Runs in minutes, not hours.

Usage:
    python -m src.coordination.hyperedge_analysis
"""

import gc
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from src.coordination.ground_truth_e3 import label_events


DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3" / "theia"
)

ALL_EVENT_TYPES = [
    "EVENT_MPROTECT", "EVENT_RECVFROM", "EVENT_READ_SOCKET_PARAMS",
    "EVENT_READ", "EVENT_MMAP", "EVENT_OPEN", "EVENT_RECVMSG",
    "EVENT_SENDMSG", "EVENT_WRITE", "EVENT_SENDTO", "EVENT_CONNECT",
    "EVENT_UNLINK", "EVENT_CLONE", "EVENT_EXECUTE",
    "EVENT_WRITE_SOCKET_PARAMS", "EVENT_SHM",
    "EVENT_MODIFY_FILE_ATTRIBUTES", "EVENT_BOOT",
]


def build_hyperedges_windowed(events_df, event_labels,
                               window_sec=30.0, epsilon_sec=5.0,
                               min_size=2, max_events_per_window=5000):
    """
    Build hyperedges using windowed Union-Find.

    1. Split events into non-overlapping windows of window_sec.
    2. Within each window, group events sharing an entity within epsilon_sec.
    3. Skip windows with > max_events_per_window events (too dense).

    Returns list of hyperedge dicts.
    """
    ts_all = events_df["timestamp_nanos"].values.astype(np.float64)
    sub_all = events_df["subject_uuid"].values
    obj_all = events_df["predicate_object_uuid"].values
    type_all = events_df["type"].values.astype(str)
    labels_all = event_labels.values

    t_min, t_max = ts_all.min(), ts_all.max()
    window_ns = int(window_sec * 1e9)
    eps_ns = epsilon_sec * 1e9

    window_starts = np.arange(t_min, t_max + 1, window_ns)
    print(f"  Windows: {len(window_starts)} x {window_sec}s")

    all_hyperedges = []
    skipped = 0

    for w_idx, w_start in enumerate(window_starts):
        w_end = w_start + window_ns
        mask = (ts_all >= w_start) & (ts_all < w_end)
        indices = np.where(mask)[0]
        n = len(indices)

        if n < min_size:
            continue
        if n > max_events_per_window:
            skipped += 1
            continue

        # Local arrays for this window
        ts = ts_all[indices]
        subs = sub_all[indices]
        objs = obj_all[indices]

        # Entity -> local event indices
        entity_to_local = defaultdict(list)
        for local_i in range(n):
            s = subs[local_i]
            o = objs[local_i]
            if pd.notna(s):
                entity_to_local[s].append(local_i)
            if pd.notna(o):
                entity_to_local[o].append(local_i)

        # Union-Find (local indices)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Merge events sharing entity within epsilon
        for entity, local_indices in entity_to_local.items():
            if len(local_indices) < 2:
                continue
            li_sorted = sorted(local_indices, key=lambda i: ts[i])
            for j in range(len(li_sorted)):
                for k in range(j + 1, len(li_sorted)):
                    if ts[li_sorted[k]] - ts[li_sorted[j]] > eps_ns:
                        break
                    union(li_sorted[j], li_sorted[k])

        # Collect clusters
        clusters = defaultdict(list)
        for local_i in range(n):
            clusters[find(local_i)].append(local_i)

        for root, members in clusters.items():
            if len(members) < min_size:
                continue

            # Map back to global indices
            global_members = [indices[m] for m in members]
            member_ts = ts[members]
            member_subs = set()
            member_objs = set()
            type_counts = Counter()
            n_atk_events = 0

            for m in members:
                gi = indices[m]
                s = subs[m]
                o = objs[m]
                if pd.notna(s):
                    member_subs.add(s)
                if pd.notna(o):
                    member_objs.add(o)
                type_counts[type_all[gi]] += 1
                if labels_all[gi] == 1:
                    n_atk_events += 1

            all_hyperedges.append({
                "global_indices": global_members,
                "size": len(members),
                "n_subjects": len(member_subs),
                "n_objects": len(member_objs),
                "n_entities": len(member_subs) + len(member_objs),
                "duration_sec": (member_ts.max() - member_ts.min()) / 1e9,
                "ts_start": member_ts.min(),
                "event_type_counts": dict(type_counts),
                "subject_uuids": member_subs,
                "object_uuids": member_objs,
                "n_attack_events": n_atk_events,
                "n_benign_events": len(members) - n_atk_events,
            })

        if (w_idx + 1) % 200 == 0:
            print(f"    Window {w_idx+1}/{len(window_starts)}: "
                  f"{len(all_hyperedges):,} hyperedges so far")

    print(f"  Skipped {skipped} dense windows (>{max_events_per_window} events)")
    return all_hyperedges


def extract_he_features(hyperedges):
    """Extract per-hyperedge feature matrix."""
    rows = []
    for he in hyperedges:
        tc = he["event_type_counts"]
        total = he["size"]
        row = {
            "size": total,
            "n_subjects": he["n_subjects"],
            "n_objects": he["n_objects"],
            "n_entities": he["n_entities"],
            "duration_sec": he["duration_sec"],
            "n_unique_types": len(tc),
        }
        for et in ALL_EVENT_TYPES:
            row[f"frac_{et}"] = tc.get(et, 0) / total

        row["has_connect"] = int(tc.get("EVENT_CONNECT", 0) > 0)
        row["has_write"] = int(tc.get("EVENT_WRITE", 0) > 0)
        row["has_execute"] = int(tc.get("EVENT_EXECUTE", 0) > 0)
        row["has_read"] = int(tc.get("EVENT_READ", 0) > 0)
        row["has_sendto"] = int(tc.get("EVENT_SENDTO", 0) > 0)
        row["has_write_connect"] = int(
            tc.get("EVENT_WRITE", 0) > 0 and tc.get("EVENT_CONNECT", 0) > 0)
        row["has_wec"] = int(
            tc.get("EVENT_WRITE", 0) > 0 and
            tc.get("EVENT_EXECUTE", 0) > 0 and
            tc.get("EVENT_CONNECT", 0) > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def enrich_he(he, subjects_df, objects_df):
    """Add human-readable names."""
    sub_map = subjects_df.set_index("uuid")["process_path"].to_dict()
    obj_fname = objects_df.set_index("uuid")["filename"].to_dict() if "filename" in objects_df.columns else {}

    # Try to get remote_address if available
    obj_raddr = {}
    if "remote_address" in objects_df.columns:
        obj_raddr = objects_df.set_index("uuid")["remote_address"].to_dict()

    snames = []
    for s in he["subject_uuids"]:
        p = sub_map.get(s, "")
        snames.append(str(p).split("/")[-1] if p else s[:10])

    onames = []
    for o in list(he["object_uuids"])[:6]:
        f = obj_fname.get(o, "")
        r = obj_raddr.get(o, "")
        if pd.notna(f) and f:
            onames.append(str(f).split("/")[-1])
        elif pd.notna(r) and r:
            onames.append(str(r))
        else:
            onames.append(o[:10])

    return snames, onames


def main():
    print("=" * 60)
    print("HYPEREDGE ANALYSIS (Windowed, Fast)")
    print("=" * 60)

    # Load shard 0
    shard_dir = DATA_DIR / "shards"
    print("\nLoading shard 0...")
    events_df = pd.read_parquet(shard_dir / "events_shard0.parquet")
    subjects_df = pd.read_parquet(shard_dir / "subjects_shard0.parquet")
    objects_df = pd.read_parquet(shard_dir / "objects_shard0.parquet")
    print(f"  Events: {len(events_df):,}, Time: "
          f"{events_df['timestamp'].min()} → {events_df['timestamp'].max()}")

    # Label events
    print("\nLabeling events...")
    event_labels = label_events(events_df, subjects_df, objects_df)
    n_atk = (event_labels == 1).sum()
    print(f"  Attack: {n_atk:,} ({100*n_atk/len(events_df):.1f}%), "
          f"Benign: {len(events_df)-n_atk:,}")

    # ============================================================
    # Build hyperedges
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 1: Build Hyperedges (Δt=5s, 30s windows)")
    print(f"{'='*60}")
    t0 = time.time()
    hyperedges = build_hyperedges_windowed(
        events_df, event_labels,
        window_sec=30.0, epsilon_sec=5.0, min_size=2,
        max_events_per_window=5000,
    )
    print(f"  Total hyperedges: {len(hyperedges):,}")
    print(f"  Build time: {time.time()-t0:.1f}s")

    if not hyperedges:
        print("  No hyperedges found!")
        return

    # ============================================================
    # Label & class balance
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 2: Labeling & Class Balance")
    print(f"{'='*60}")

    # K=1: attack if ≥1 attack event
    labels_k1 = np.array([1 if he["n_attack_events"] >= 1 else 0
                           for he in hyperedges], dtype=np.int8)
    # K=2: attack if ≥2 attack events
    labels_k2 = np.array([1 if he["n_attack_events"] >= 2 else 0
                           for he in hyperedges], dtype=np.int8)

    n_atk_k1 = (labels_k1 == 1).sum()
    n_ben_k1 = (labels_k1 == 0).sum()
    n_atk_k2 = (labels_k2 == 1).sum()

    mixed = sum(1 for he in hyperedges
                if he["n_attack_events"] > 0 and he["n_benign_events"] > 0)

    print(f"\n  K=1 (≥1 attack event):")
    print(f"    Attack HEs: {n_atk_k1:,} ({100*n_atk_k1/len(hyperedges):.1f}%)")
    print(f"    Benign HEs: {n_ben_k1:,} ({100*n_ben_k1/len(hyperedges):.1f}%)")
    print(f"    Ratio:      1:{n_ben_k1/max(n_atk_k1,1):.1f}")
    print(f"\n  K=2 (≥2 attack events):")
    print(f"    Attack HEs: {n_atk_k2:,}")
    print(f"\n  Mixed HEs (attack+benign events): {mixed:,}")

    # ============================================================
    # Size distribution
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 3: Size Distribution")
    print(f"{'='*60}")

    sizes = np.array([he["size"] for he in hyperedges])
    ents = np.array([he["n_entities"] for he in hyperedges])
    durs = np.array([he["duration_sec"] for he in hyperedges])

    print(f"\n  ALL ({len(hyperedges):,}):")
    print(f"    Size:     mean={sizes.mean():.1f} med={np.median(sizes):.0f} "
          f"min={sizes.min()} max={sizes.max()}")
    print(f"    Entities: mean={ents.mean():.1f} med={np.median(ents):.0f}")
    print(f"    Duration: mean={durs.mean():.2f}s med={np.median(durs):.2f}s")

    # Size histogram
    print(f"\n    Size histogram:")
    for s in range(2, min(16, sizes.max() + 1)):
        cnt = (sizes == s).sum()
        bar = "█" * min(cnt // 50, 40)
        print(f"      {s:3d}: {cnt:>7,} {bar}")
    if sizes.max() >= 16:
        cnt = (sizes >= 16).sum()
        print(f"     16+: {cnt:>7,}")

    # Attack vs benign
    atk_mask = labels_k1 == 1
    ben_mask = labels_k1 == 0
    if atk_mask.sum() > 0:
        print(f"\n  ATTACK ({atk_mask.sum():,}): "
              f"size mean={sizes[atk_mask].mean():.1f}, "
              f"ents mean={ents[atk_mask].mean():.1f}")
    print(f"  BENIGN ({ben_mask.sum():,}): "
          f"size mean={sizes[ben_mask].mean():.1f}, "
          f"ents mean={ents[ben_mask].mean():.1f}")

    # ============================================================
    # Concrete examples
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 4: Concrete Examples")
    print(f"{'='*60}")

    # Pre-compute lookup maps once
    sub_map = subjects_df.set_index("uuid")["process_path"].to_dict()
    obj_fname = objects_df.set_index("uuid")["filename"].to_dict() if "filename" in objects_df.columns else {}
    obj_raddr = objects_df.set_index("uuid")["remote_address"].to_dict() if "remote_address" in objects_df.columns else {}

    def print_he(he, label_str):
        snames, onames = enrich_he(he, subjects_df, objects_df)
        ts_dt = pd.to_datetime(he["ts_start"], unit="ns")
        print(f"\n  [{label_str}] size={he['size']} "
              f"({he['n_attack_events']} atk, {he['n_benign_events']} ben) "
              f"dur={he['duration_sec']:.3f}s")
        print(f"    Time: {ts_dt}")
        print(f"    Subjects: {snames}")
        print(f"    Objects:  {onames}")
        print(f"    Types: {he['event_type_counts']}")

    # Attack with WRITE+CONNECT
    atk_wc = [he for he in hyperedges
              if he["n_attack_events"] >= 1
              and he["event_type_counts"].get("EVENT_WRITE", 0) > 0
              and he["event_type_counts"].get("EVENT_CONNECT", 0) > 0]
    print(f"\n  Attack HEs with WRITE+CONNECT: {len(atk_wc)}")
    for he in sorted(atk_wc, key=lambda h: -len(h["event_type_counts"]))[:3]:
        print_he(he, "ATK-WR+CN")

    # Attack with WRITE+EXECUTE
    atk_we = [he for he in hyperedges
              if he["n_attack_events"] >= 1
              and he["event_type_counts"].get("EVENT_WRITE", 0) > 0
              and he["event_type_counts"].get("EVENT_EXECUTE", 0) > 0]
    print(f"\n  Attack HEs with WRITE+EXECUTE: {len(atk_we)}")
    for he in sorted(atk_we, key=lambda h: -len(h["event_type_counts"]))[:3]:
        print_he(he, "ATK-WR+EX")

    # Most diverse attack HEs
    atk_hes = [he for he in hyperedges if he["n_attack_events"] >= 1]
    print(f"\n  Most diverse attack HEs:")
    for he in sorted(atk_hes, key=lambda h: -len(h["event_type_counts"]))[:3]:
        print_he(he, "ATK-DIV")

    # Benign examples
    ben_hes = [he for he in hyperedges if he["n_attack_events"] == 0]
    print(f"\n  Benign examples:")
    for he in sorted(ben_hes, key=lambda h: -len(h["event_type_counts"]))[:2]:
        print_he(he, "BENIGN")

    # ============================================================
    # RF sanity check
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 5: RF Sanity Check")
    print(f"{'='*60}")

    X_df = extract_he_features(hyperedges)
    y = labels_k1
    feat_names = X_df.columns.tolist()
    X = np.nan_to_num(X_df.values, nan=0.0, posinf=0.0, neginf=0.0)

    # Balanced subsample
    rng = np.random.default_rng(42)
    atk_idx = np.where(y == 1)[0]
    ben_idx = np.where(y == 0)[0]
    n_sample = min(len(atk_idx), len(ben_idx), 50000)

    if n_sample < 10:
        print(f"  Too few attack HEs ({len(atk_idx)}). Skipping RF.")
    else:
        s_atk = rng.choice(atk_idx, n_sample, replace=False)
        s_ben = rng.choice(ben_idx, n_sample, replace=False)
        idx = np.sort(np.concatenate([s_atk, s_ben]))
        X_s, y_s = X[idx], y[idx]

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_s, y_s, cv=cv, scoring="roc_auc")
        print(f"\n  AUC (5-fold, balanced): {scores.mean():.4f} ± {scores.std():.4f}")

        clf.fit(X_s, y_s)
        imps = sorted(zip(feat_names, clf.feature_importances_),
                       key=lambda x: -x[1])
        print(f"\n  Top features:")
        for name, imp in imps[:10]:
            print(f"    {name:35s} {imp:.4f}")

    # ============================================================
    # Single-feature confound
    # ============================================================
    print(f"\n{'='*60}")
    print("STEP 6: Single-Feature Confound Check")
    print(f"{'='*60}")

    print(f"\n  Features with AUC > 0.90 (potential confounds):")
    confounds_found = False
    for i, feat in enumerate(feat_names):
        vals = X[:, i]
        try:
            auc = roc_auc_score(y, vals)
            if auc < 0.1:
                auc = 1 - auc  # Flip if inversely correlated
            if auc > 0.90:
                print(f"    {feat:35s} AUC={auc:.4f}")
                confounds_found = True
        except ValueError:
            pass
    if not confounds_found:
        print(f"    None found — no single feature dominates.")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Hyperedges:     {len(hyperedges):,}")
    print(f"  Attack (K=1):   {n_atk_k1:,} ({100*n_atk_k1/len(hyperedges):.1f}%)")
    print(f"  Benign:         {n_ben_k1:,}")
    print(f"  Mixed:          {mixed:,}")
    print(f"  Avg size:       {sizes.mean():.1f}")
    print(f"  Avg entities:   {ents.mean():.1f}")
    print(f"  WR+CN attack:   {len(atk_wc)}")
    print(f"  WR+EX attack:   {len(atk_we)}")


if __name__ == "__main__":
    main()
