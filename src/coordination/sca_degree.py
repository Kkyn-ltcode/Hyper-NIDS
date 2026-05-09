"""
Kill Switch Experiment: SCA-Degree Measurement (memory-safe).

Trains Random Forest classifiers at three feature levels
(individual, pairwise, group) to measure Detection Uplift.

Memory-safe: subsamples BEFORE feature extraction so it works
on 44M events with only 8 GB RAM.

Protocol:
    1. Subsample events (balanced attack/benign)
    2. Extract features on subsample only
    3. Train RF with 5-fold stratified CV
    4. Measure AUC-ROC at each level
    5. Detection Uplift = AUC_group - max(AUC_individual, AUC_pairwise)

Decision:
    Proceed if Detection Uplift > 0.02 on this full-dataset scenario.

Usage:
    python -m src.coordination.sca_degree
"""

import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.coordination.feature_extraction import (
    extract_individual_features,
    extract_group_features,
)
from src.coordination.ground_truth_e3 import (
    label_events,
    label_windows,
)

warnings.filterwarnings("ignore", category=UserWarning)


DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3" / "theia"
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load parsed DARPA TC E3 Theia data."""
    return (
        pd.read_parquet(DATA_DIR / "events.parquet"),
        pd.read_parquet(DATA_DIR / "subjects.parquet"),
        pd.read_parquet(DATA_DIR / "objects.parquet"),
    )


def eval_individual(events_df, event_labels, n_samples=200_000, seed=42):
    """
    Evaluate individual-level classifier.

    MEMORY-SAFE: subsamples events FIRST, then extracts features
    only on the subsample.
    """
    print("\n" + "="*60)
    print("LEVEL 1: Individual Features (per-event)")
    print("="*60)

    # Subsample for tractability (balanced)
    rng = np.random.default_rng(seed)
    atk_idx = np.where(event_labels == 1)[0]
    ben_idx = np.where(event_labels == 0)[0]
    n_per_class = min(n_samples // 2, len(atk_idx), len(ben_idx))

    sample_atk = rng.choice(atk_idx, n_per_class, replace=False)
    sample_ben = rng.choice(ben_idx, n_per_class, replace=False)
    sample_idx = np.sort(np.concatenate([sample_atk, sample_ben]))

    # Extract features ONLY on subsample
    events_sample = events_df.iloc[sample_idx].reset_index(drop=True)
    X_df = extract_individual_features(events_sample)
    X = X_df.values
    y = event_labels.iloc[sample_idx].values
    feat_names = X_df.columns.tolist()
    del events_sample; gc.collect()

    print(f"  Samples: {len(X):,} ({n_per_class:,} per class)")
    print(f"  Features: {X.shape[1]}")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 5-fold stratified CV
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    t0 = time.time()
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    elapsed = time.time() - t0

    auc_mean = scores.mean()
    auc_std = scores.std()
    print(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    # Feature importances
    clf.fit(X, y)
    importances = sorted(
        zip(feat_names, clf.feature_importances_), key=lambda x: -x[1]
    )
    print(f"  Top features:")
    for name, imp in importances[:5]:
        print(f"    {name:25s} {imp:.4f}")

    del X, y; gc.collect()
    return {"auc_mean": auc_mean, "auc_std": auc_std, "n_samples": n_samples}


def eval_pairwise(events_df, event_labels, n_samples=200_000, seed=42):
    """
    Evaluate pairwise-level classifier.

    MEMORY-SAFE: computes pair indices first, subsamples,
    then extracts features only on sampled pairs.
    """
    print("\n" + "="*60)
    print("LEVEL 2: Pairwise Features (consecutive pairs)")
    print("="*60)

    labels = event_labels.values
    pair_labels = np.maximum(labels[:-1], labels[1:])

    # Subsample pairs (balanced)
    rng = np.random.default_rng(seed)
    atk_pairs = np.where(pair_labels == 1)[0]
    ben_pairs = np.where(pair_labels == 0)[0]
    n_per_class = min(n_samples // 2, len(atk_pairs), len(ben_pairs))

    sample_atk = rng.choice(atk_pairs, n_per_class, replace=False)
    sample_ben = rng.choice(ben_pairs, n_per_class, replace=False)
    pair_idx = np.sort(np.concatenate([sample_atk, sample_ben]))

    # Gather the unique event indices needed (each pair uses idx and idx+1)
    needed_idx = np.unique(np.concatenate([pair_idx, pair_idx + 1]))
    events_subset = events_df.iloc[needed_idx].reset_index(drop=True)

    # Build mapping: original_idx -> subset_idx
    idx_map = {orig: new for new, orig in enumerate(needed_idx)}
    local_i = np.array([idx_map[i] for i in pair_idx])
    local_j = np.array([idx_map[i + 1] for i in pair_idx])

    # Extract individual features on subset only
    indiv = extract_individual_features(events_subset)
    feat_i = indiv.iloc[local_i].values
    feat_j = indiv.iloc[local_j].values

    # Interaction features
    ts = events_subset["timestamp_nanos"].values.astype(np.float64)
    time_gap = ((ts[local_j] - ts[local_i]) / 1e9).reshape(-1, 1)

    types = events_subset["type"].values.astype(str)
    same_type = (types[local_i] == types[local_j]).astype(np.float32).reshape(-1, 1)

    subs = events_subset["subject_uuid"].values.astype(str)
    same_sub = (subs[local_i] == subs[local_j]).astype(np.float32).reshape(-1, 1)

    objs = events_subset["predicate_object_uuid"].values.astype(str)
    same_obj = (objs[local_i] == objs[local_j]).astype(np.float32).reshape(-1, 1)

    X = np.hstack([feat_i, feat_j, time_gap, same_type, same_sub, same_obj])
    y = pair_labels[pair_idx]
    del events_subset, indiv, feat_i, feat_j; gc.collect()

    print(f"  Pairs: {len(X):,} ({n_per_class:,} per class)")
    print(f"  Features: {X.shape[1]}")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    t0 = time.time()
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    elapsed = time.time() - t0

    auc_mean = scores.mean()
    auc_std = scores.std()
    print(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    del X, y; gc.collect()
    return {"auc_mean": auc_mean, "auc_std": auc_std, "n_samples": n_samples}


def eval_group(events_df, event_labels, seed=42):
    """
    Evaluate group-level classifier.

    Uses multiple window sizes. With 44M events over 3 days,
    this gives hundreds of windows — proper CV is possible.
    """
    print("\n" + "="*60)
    print("LEVEL 3: Group Features (time windows)")
    print("="*60)

    all_X = []
    all_y = []

    for ws in [60, 120, 300]:
        print(f"  Extracting {ws}s windows...", end=" ", flush=True)
        group_feats = extract_group_features(
            events_df, window_seconds=ws, min_events=10
        )
        if len(group_feats) == 0:
            print("no windows")
            continue

        wl = label_windows(events_df, event_labels, window_seconds=ws)

        merged = group_feats.merge(
            wl[["window_start", "label", "attack_fraction"]],
            on="window_start", how="inner",
        )

        feat_cols = [c for c in merged.columns if c not in
                     ("window_start", "window_end", "window_start_dt",
                      "window_end_dt", "label", "attack_fraction")]

        X = merged[feat_cols].values
        y = merged["label"].values

        all_X.append(X)
        all_y.append(y)
        print(f"{len(X)} windows ({int(y.sum())} atk, {int((y==0).sum())} ben)")

    if not all_X:
        print("  ERROR: No windows")
        return {"auc_mean": 0.0, "auc_std": 0.0, "n_samples": 0}

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"\n  Total: {len(X)} windows ({int(y.sum())} atk, {int((y==0).sum())} ben)")
    print(f"  Features: {X.shape[1]}")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n_splits = min(5, int(y.sum()), int((y == 0).sum()))
    if n_splits < 2:
        print("  WARNING: Not enough for CV, using train-test split")
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=seed
        )
        clf.fit(X_tr, y_tr)
        auc_mean = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
        auc_std = 0.0
    else:
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1
        )
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        auc_mean = scores.mean()
        auc_std = scores.std()

    print(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f}")

    del X, y; gc.collect()
    return {"auc_mean": auc_mean, "auc_std": auc_std, "n_samples": len(all_X[0])}


def main():
    print("="*60)
    print("KILL SWITCH EXPERIMENT: SCA-Degree (Full Dataset)")
    print("="*60)

    print("\nLoading data...")
    events_df, subjects_df, objects_df = load_data()
    print(f"  Events: {len(events_df):,}")

    print("Labeling events...")
    event_labels = label_events(events_df, subjects_df, objects_df)
    n_atk = (event_labels == 1).sum()
    n_ben = (event_labels == 0).sum()
    print(f"  Attack: {n_atk:,} ({100*n_atk/len(events_df):.1f}%)")
    print(f"  Benign: {n_ben:,} ({100*n_ben/len(events_df):.1f}%)")

    # Free subjects/objects — only events + labels needed from here
    del subjects_df, objects_df; gc.collect()

    # Run classifiers at each level
    t0 = time.time()
    r_individual = eval_individual(events_df, event_labels)
    r_pairwise = eval_pairwise(events_df, event_labels)
    r_group = eval_group(events_df, event_labels)
    total_time = time.time() - t0

    # ============================================================
    # DECISION
    # ============================================================
    auc_i = r_individual["auc_mean"]
    auc_p = r_pairwise["auc_mean"]
    auc_g = r_group["auc_mean"]
    detection_uplift = auc_g - max(auc_i, auc_p)

    print("\n" + "="*60)
    print("KILL SWITCH RESULTS (Full 44M events)")
    print("="*60)
    print(f"\n  AUC Individual:    {auc_i:.4f}")
    print(f"  AUC Pairwise:      {auc_p:.4f}")
    print(f"  AUC Group:         {auc_g:.4f}")
    print(f"\n  Detection Uplift:  {detection_uplift:.4f}")

    threshold = 0.02
    if detection_uplift > threshold:
        decision = "GO ✓"
        print(f"\n  *** DECISION: {decision} ***")
        print(f"  Uplift ({detection_uplift:.4f}) > threshold ({threshold})")
    else:
        decision = "INVESTIGATE ⚠"
        print(f"\n  *** DECISION: {decision} ***")
        print(f"  Uplift ({detection_uplift:.4f}) <= threshold ({threshold})")

    print(f"\n  Total time: {total_time:.1f}s")

    # Save results
    results = {
        "dataset": "DARPA TC E3 Theia (all 10 shards)",
        "total_events": len(events_df),
        "attack_events": int(n_atk),
        "benign_events": int(n_ben),
        "individual_auc": round(auc_i, 4),
        "pairwise_auc": round(auc_p, 4),
        "group_auc": round(auc_g, 4),
        "detection_uplift": round(detection_uplift, 4),
        "decision": decision,
        "individual_details": r_individual,
        "pairwise_details": r_pairwise,
        "group_details": r_group,
    }
    out_path = DATA_DIR / "kill_switch_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
