"""
Sanity Check: Reproduces the atomic hyperedge RF baseline.

Validates the modular pipeline by reproducing AUC 0.8918 on shard 0.
Uses the refactored modules (ground_truth, feature_extractor,
build_hypergraph, build_sequences).

Usage:
    python -m src.pipeline.sanity_check [--dataset theia] [--shard 0]
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data.ground_truth import load_ground_truth, label_events
from src.features.feature_extractor import extract_features
from src.data.build_hypergraph import build_incidence, incidence_stats
from src.data.build_sequences import build_subject_sequences, sequence_stats


DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3"
)


def get_data_dir(dataset: str, shard: int | None = None) -> Path:
    """Get data directory for a dataset and optional shard."""
    base = DATA_ROOT / dataset
    if shard is not None:
        return base / "shards"
    return base


def load_shard(dataset: str, shard: int) -> tuple:
    """Load a single shard's events, subjects, and objects."""
    shard_dir = get_data_dir(dataset, shard=shard)
    events = pd.read_parquet(shard_dir / f"events_shard{shard}.parquet")
    subjects = pd.read_parquet(shard_dir / f"subjects_shard{shard}.parquet")
    objects = pd.read_parquet(shard_dir / f"objects_shard{shard}.parquet")
    return events, subjects, objects


def main():
    parser = argparse.ArgumentParser(description="THyN pipeline sanity check")
    parser.add_argument("--dataset", default="theia",
                        choices=["theia", "trace"])
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print(f"SANITY CHECK: {args.dataset.upper()} shard {args.shard}")
    print("=" * 60)

    # ============================================================
    # Step 1: Load data
    # ============================================================
    print(f"\n[1/6] Loading data...")
    t0 = time.time()
    events_df, subjects_df, objects_df = load_shard(args.dataset, args.shard)
    print(f"  Events:   {len(events_df):,}")
    print(f"  Subjects: {len(subjects_df):,}")
    print(f"  Objects:  {len(objects_df):,}")
    print(f"  Time:     {events_df['timestamp'].min()} → "
          f"{events_df['timestamp'].max()}")
    print(f"  Load time: {time.time()-t0:.1f}s")

    # ============================================================
    # Step 2: Label events
    # ============================================================
    print(f"\n[2/6] Labeling events...")
    gt = load_ground_truth(args.dataset)
    labels = label_events(events_df, subjects_df, objects_df, gt)
    labels_arr = labels.values

    n_atk = (labels_arr == 1).sum()
    n_ben = (labels_arr == 0).sum()
    print(f"  Attack:  {n_atk:,} ({100*n_atk/len(events_df):.1f}%)")
    print(f"  Benign:  {n_ben:,} ({100*n_ben/len(events_df):.1f}%)")
    print(f"  Ratio:   1:{n_ben/max(n_atk,1):.1f}")

    # ============================================================
    # Step 3: Hyperedge sizes
    # ============================================================
    print(f"\n[3/6] Hyperedge size distribution...")
    has_sub = events_df["subject_uuid"].notna()
    has_obj = events_df["predicate_object_uuid"].notna()
    has_obj2 = events_df["predicate_object2_uuid"].notna()
    sizes = has_sub.astype(int) + has_obj.astype(int) + has_obj2.astype(int)

    for s in sorted(sizes.unique()):
        cnt = (sizes == s).sum()
        print(f"  size {s}: {cnt:>10,} ({100*cnt/len(events_df):.1f}%)")

    n_unique_obj2 = events_df["predicate_object2_uuid"].nunique()
    print(f"  Unique obj2: {n_unique_obj2:,}")

    # ============================================================
    # Step 4: Build hypergraph + sequences
    # ============================================================
    print(f"\n[4/6] Building hypergraph & sequences...")
    t0 = time.time()
    entity_vocab, incidence = build_incidence(events_df)
    hg_stats = incidence_stats(entity_vocab, incidence)
    print(f"  Entities:     {hg_stats['n_entities']:,}")
    print(f"  Hyperedges:   {hg_stats['n_hyperedges']:,}")
    print(f"  HE size:      mean={hg_stats['he_size_mean']:.2f}")
    print(f"  Node degree:  mean={hg_stats['node_degree_mean']:.1f}, "
          f"median={hg_stats['node_degree_median']:.0f}, "
          f"max={hg_stats['node_degree_max']:,}")

    sequences = build_subject_sequences(events_df)
    seq_stats = sequence_stats(sequences)
    print(f"\n  Subject sequences:")
    print(f"    Subjects:      {seq_stats['n_subjects']:,}")
    print(f"    Seq length:    mean={seq_stats['seq_len_mean']:.0f}, "
          f"median={seq_stats['seq_len_median']:.0f}, "
          f"max={seq_stats['seq_len_max']:,}")
    print(f"    Subjects >1K:  {seq_stats['subjects_gt_1000']:,}")
    print(f"    Subjects >10K: {seq_stats['subjects_gt_10000']:,}")
    print(f"    Subjects >100K:{seq_stats['subjects_gt_100000']:,}")
    print(f"  Build time: {time.time()-t0:.1f}s")

    # ============================================================
    # Step 5: Extract features
    # ============================================================
    print(f"\n[5/6] Extracting features...")
    t0 = time.time()
    X, feat_names, _ = extract_features(events_df)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Features: {X.shape[1]}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ============================================================
    # Step 6: RF sanity check
    # ============================================================
    print(f"\n[6/6] RF classification (balanced, 5-fold CV)...")
    rng = np.random.default_rng(42)
    atk_idx = np.where(labels_arr == 1)[0]
    ben_idx = np.where(labels_arr == 0)[0]
    n_sample = min(len(atk_idx), len(ben_idx), 100_000)

    if n_sample < 10:
        print(f"  Too few attack events ({len(atk_idx)}). Skipping.")
        return

    s_atk = rng.choice(atk_idx, n_sample, replace=False)
    s_ben = rng.choice(ben_idx, n_sample, replace=False)
    sample = np.sort(np.concatenate([s_atk, s_ben]))
    X_s, y_s = X[sample], labels_arr[sample]

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    t0 = time.time()
    auc_scores = cross_val_score(
        clf, X_s, y_s, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"  AUC-ROC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

    # Also compute AUPRC (baseline for the paper)
    auprc_scores = cross_val_score(
        clf, X_s, y_s, cv=cv, scoring="average_precision", n_jobs=-1)
    print(f"  AUPRC:   {auprc_scores.mean():.4f} ± {auprc_scores.std():.4f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Feature importances
    clf.fit(X_s, y_s)
    imps = sorted(zip(feat_names, clf.feature_importances_),
                  key=lambda x: -x[1])
    print(f"\n  Top 10 features:")
    for name, imp in imps[:10]:
        print(f"    {name:35s} {imp:.4f}")

    # Single-feature confound check
    print(f"\n  Single-feature AUC check (>0.70 = flag):")
    found = False
    for i, feat in enumerate(feat_names):
        vals = X_s[:, i]
        if np.std(vals) == 0:
            continue
        try:
            auc = roc_auc_score(y_s, vals)
            if auc < 0.5:
                auc = 1 - auc
            if auc > 0.70:
                print(f"    {feat:35s} AUC={auc:.4f}")
                found = True
        except ValueError:
            pass
    if not found:
        print(f"    None — no trivial confounds ✓")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Dataset:     {args.dataset} shard {args.shard}")
    print(f"  Events:      {len(events_df):,}")
    print(f"  HE sizes:    100% size-{int(sizes.mode().iloc[0])}")
    print(f"  Attack:      {n_atk:,} ({100*n_atk/len(events_df):.1f}%)")
    print(f"  Entities:    {hg_stats['n_entities']:,}")
    print(f"  Unique obj2: {n_unique_obj2:,}")
    print(f"  Subjects:    {seq_stats['n_subjects']:,}")
    print(f"  Max seq len: {seq_stats['seq_len_max']:,}")
    print(f"  AUC-ROC:     {auc_scores.mean():.4f}")
    print(f"  AUPRC:       {auprc_scores.mean():.4f}")
    print(f"\n  Expected AUC-ROC: 0.8918 (from original atomic check)")
    diff = abs(auc_scores.mean() - 0.8918)
    if diff < 0.005:
        print(f"  ✓ REPRODUCED (diff={diff:.4f})")
    else:
        print(f"  ⚠ MISMATCH (diff={diff:.4f}) — investigate")

    # Free memory
    del X, X_s
    gc.collect()


if __name__ == "__main__":
    main()
