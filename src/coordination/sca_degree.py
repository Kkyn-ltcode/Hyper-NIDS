"""
Kill Switch Experiment: SCA-Degree Measurement.

Trains Random Forest classifiers at three feature levels
(individual, pairwise, group) to measure Detection Uplift.

Protocol:
    1. Extract features at each level
    2. Label data using ground truth
    3. Train RF with 5-fold stratified CV
    4. Measure AUC-ROC at each level
    5. Detection Uplift = AUC_group - max(AUC_individual, AUC_pairwise)

Decision:
    Proceed if Detection Uplift > 0.1 on this scenario.

Usage:
    python -m src.coordination.sca_degree
"""

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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load parsed DARPA TC E3 Theia data."""
    d = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "darpa_tc_e3" / "theia"
    return (
        pd.read_parquet(d / "events.parquet"),
        pd.read_parquet(d / "subjects.parquet"),
        pd.read_parquet(d / "objects.parquet"),
    )


def eval_individual(events_df, event_labels, n_samples=200_000, seed=42):
    """
    Evaluate individual-level classifier.

    Subsamples events for tractability, extracts per-event features,
    trains RF, returns AUC via 5-fold CV.
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

    X_full = extract_individual_features(events_df)
    X = X_full.iloc[sample_idx].values
    y = event_labels.iloc[sample_idx].values

    print(f"  Samples: {len(X):,} ({n_per_class:,} per class)")
    print(f"  Features: {X.shape[1]}")

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 5-fold stratified CV
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    t0 = time.time()
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    elapsed = time.time() - t0

    auc_mean = scores.mean()
    auc_std = scores.std()
    print(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    # Feature importances (train on full sample)
    clf.fit(X, y)
    feat_names = X_full.columns.tolist()
    importances = sorted(zip(feat_names, clf.feature_importances_), key=lambda x: -x[1])
    print(f"  Top features:")
    for name, imp in importances[:5]:
        print(f"    {name:25s} {imp:.4f}")

    return {"auc_mean": auc_mean, "auc_std": auc_std, "n_samples": len(X)}


def eval_pairwise(events_df, event_labels, n_samples=200_000, seed=42):
    """
    Evaluate pairwise-level classifier.

    Creates consecutive event pairs, labels pair as attack if EITHER
    event is attack. Uses individual features of both events + interaction.
    """
    print("\n" + "="*60)
    print("LEVEL 2: Pairwise Features (consecutive pairs)")
    print("="*60)

    # Build pairs from consecutive events
    n = len(events_df)
    labels = event_labels.values

    # Pair label: 1 if either event is attack
    pair_labels = np.maximum(labels[:-1], labels[1:])

    # Subsample pairs (balanced)
    rng = np.random.default_rng(seed)
    atk_pairs = np.where(pair_labels == 1)[0]
    ben_pairs = np.where(pair_labels == 0)[0]
    n_per_class = min(n_samples // 2, len(atk_pairs), len(ben_pairs))

    sample_atk = rng.choice(atk_pairs, n_per_class, replace=False)
    sample_ben = rng.choice(ben_pairs, n_per_class, replace=False)
    pair_idx = np.sort(np.concatenate([sample_atk, sample_ben]))

    # Extract individual features
    indiv = extract_individual_features(events_df)
    feat_i = indiv.iloc[pair_idx].values
    feat_j = indiv.iloc[pair_idx + 1].values

    # Interaction features
    ts = events_df["timestamp_nanos"].values.astype(np.float64)
    time_gap = ((ts[pair_idx + 1] - ts[pair_idx]) / 1e9).reshape(-1, 1)

    types = events_df["type"].values.astype(str)
    same_type = (types[pair_idx] == types[pair_idx + 1]).astype(np.float32).reshape(-1, 1)

    subs = events_df["subject_uuid"].values.astype(str)
    same_sub = (subs[pair_idx] == subs[pair_idx + 1]).astype(np.float32).reshape(-1, 1)

    objs = events_df["predicate_object_uuid"].values.astype(str)
    same_obj = (objs[pair_idx] == objs[pair_idx + 1]).astype(np.float32).reshape(-1, 1)

    X = np.hstack([feat_i, feat_j, time_gap, same_type, same_sub, same_obj])
    y = pair_labels[pair_idx]

    print(f"  Pairs: {len(X):,} ({n_per_class:,} per class)")
    print(f"  Features: {X.shape[1]}")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    t0 = time.time()
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    elapsed = time.time() - t0

    auc_mean = scores.mean()
    auc_std = scores.std()
    print(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    return {"auc_mean": auc_mean, "auc_std": auc_std, "n_samples": len(X)}


def eval_group(events_df, event_labels, window_seconds=60.0, seed=42):
    """
    Evaluate group-level classifier.

    Extracts CD + aggregate features per time window,
    labels window as attack if ANY event is attack.

    Uses multiple window sizes to get enough samples.
    """
    print("\n" + "="*60)
    print("LEVEL 3: Group Features (time windows)")
    print("="*60)

    # Use multiple window sizes to create more samples
    all_X = []
    all_y = []

    for ws in [60, 120, 180, 300]:
        group_feats = extract_group_features(events_df, window_seconds=ws, min_events=10)
        if len(group_feats) == 0:
            continue

        wl = label_windows(events_df, event_labels, window_seconds=ws)

        # Merge features with labels
        merged = group_feats.merge(
            wl[["window_start", "label", "attack_fraction"]],
            on="window_start",
            how="inner",
        )

        feat_cols = [c for c in merged.columns if c not in
                     ("window_start", "window_end", "window_start_dt",
                      "window_end_dt", "label", "attack_fraction")]

        X = merged[feat_cols].values
        y = merged["label"].values

        all_X.append(X)
        all_y.append(y)
        print(f"  Window {ws}s: {len(X)} windows ({y.sum()} attack, {(y==0).sum()} benign), {X.shape[1]} features")

    if not all_X:
        print("  ERROR: No windows with enough events")
        return {"auc_mean": 0.0, "auc_std": 0.0, "n_samples": 0}

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"\n  Total windows: {len(X)} ({y.sum()} attack, {(y==0).sum()} benign)")
    print(f"  Features: {X.shape[1]}")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features (important for mixed-scale group features)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n_splits = min(5, int(y.sum()), int((y == 0).sum()))
    if n_splits < 2:
        print("  WARNING: Not enough samples for CV, using train-test split")
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)[:, 1]
        auc_mean = roc_auc_score(y_te, y_prob)
        auc_std = 0.0
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        auc_mean = scores.mean()
        auc_std = scores.std()

    print(f"  AUC: {auc_mean:.4f} ± {auc_std:.4f}")

    # Feature importances
    clf_full = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed)
    clf_full.fit(X, y)

    return {"auc_mean": auc_mean, "auc_std": auc_std, "n_samples": len(X)}


def main():
    print("="*60)
    print("KILL SWITCH EXPERIMENT: SCA-Degree Measurement")
    print("="*60)

    # Load data
    print("\nLoading data...")
    events_df, subjects_df, objects_df = load_data()
    print(f"  Events: {len(events_df):,}")

    # Label events
    print("Labeling events...")
    event_labels = label_events(events_df, subjects_df, objects_df)
    n_atk = (event_labels == 1).sum()
    n_ben = (event_labels == 0).sum()
    print(f"  Attack: {n_atk:,} ({100*n_atk/len(events_df):.1f}%)")
    print(f"  Benign: {n_ben:,} ({100*n_ben/len(events_df):.1f}%)")

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
    pairwise_max = max(auc_i, auc_p)
    detection_uplift = auc_g - pairwise_max
    pairwise_blindness = (0.5 - min(auc_i, auc_p)) * 2

    print("\n" + "="*60)
    print("KILL SWITCH RESULTS")
    print("="*60)
    print(f"\n  AUC Individual:    {auc_i:.4f}")
    print(f"  AUC Pairwise:      {auc_p:.4f}")
    print(f"  AUC Group:         {auc_g:.4f}")
    print(f"\n  Pairwise Blindness:  {pairwise_blindness:.4f}")
    print(f"  Detection Uplift:    {detection_uplift:.4f}")

    threshold = 0.1
    if detection_uplift > threshold:
        decision = "GO ✓"
        print(f"\n  *** DECISION: {decision} ***")
        print(f"  Detection Uplift ({detection_uplift:.4f}) > threshold ({threshold})")
        print(f"  Group features provide significant detection advantage.")
    else:
        decision = "INVESTIGATE ⚠"
        print(f"\n  *** DECISION: {decision} ***")
        print(f"  Detection Uplift ({detection_uplift:.4f}) <= threshold ({threshold})")
        print(f"  Need more scenarios or different feature engineering.")

    print(f"\n  Total time: {total_time:.1f}s")

    # Save results
    out_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "darpa_tc_e3" / "theia"
    results = {
        "individual_auc": round(auc_i, 4),
        "pairwise_auc": round(auc_p, 4),
        "group_auc": round(auc_g, 4),
        "pairwise_blindness": round(pairwise_blindness, 4),
        "detection_uplift": round(detection_uplift, 4),
        "decision": decision,
        "individual_details": r_individual,
        "pairwise_details": r_pairwise,
        "group_details": r_group,
    }
    with open(out_dir / "kill_switch_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_dir / 'kill_switch_results.json'}")


if __name__ == "__main__":
    main()
