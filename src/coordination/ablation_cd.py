"""
CD Ablation Study: Is the uplift real or a confound?

Three-way comparison on group features:
    A. FULL:      All 18 group features (TC, BD, ES, CD + aggregates)
    B. NO-CD:     14 aggregate features only (remove TC, BD, ES, CD)
    C. CD-ONLY:   4 CD features only (TC, BD, ES, CD)

If B ≈ A:  CD adds nothing → false positive, kill switch fails
If B << A: CD carries genuine discriminative power
If C low:  CD alone insufficient, but complementary with aggregates

Usage:
    python -m src.coordination.ablation_cd
"""

import gc
import time
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.coordination.feature_extraction import extract_group_features
from src.coordination.ground_truth_e3 import label_events, label_windows

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3" / "theia"
)

# The 4 CD-specific columns
CD_COLS = {"tc", "bd", "es", "cd"}

# Metadata columns (not features)
META_COLS = {"window_start", "window_end", "window_start_dt",
             "window_end_dt", "label", "attack_fraction"}


def run_classifier(X, y, label, seed=42):
    """Run RF with 5-fold stratified CV, return AUC."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=seed, n_jobs=-1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

    auc_mean = scores.mean()
    auc_std = scores.std()
    print(f"  {label:20s}  AUC = {auc_mean:.4f} ± {auc_std:.4f}  "
          f"({X.shape[1]} features, {len(X)} windows)")

    # Feature importances
    clf.fit(X, y)
    return auc_mean, auc_std, clf.feature_importances_


def main():
    print("="*60)
    print("ABLATION STUDY: Is CD the active ingredient?")
    print("="*60)

    # Load data
    print("\nLoading data...")
    events_df = pd.read_parquet(DATA_DIR / "events.parquet")
    subjects_df = pd.read_parquet(DATA_DIR / "subjects.parquet")
    objects_df = pd.read_parquet(DATA_DIR / "objects.parquet")
    print(f"  Events: {len(events_df):,}")

    # Label
    print("Labeling events...")
    event_labels = label_events(events_df, subjects_df, objects_df)
    del subjects_df, objects_df; gc.collect()

    # Build windows at multiple scales
    print("\nExtracting group features...")
    all_X_rows = []

    for ws in [60, 120, 300]:
        print(f"  Window {ws}s...", end=" ", flush=True)
        gf = extract_group_features(events_df, window_seconds=ws, min_events=10)
        wl = label_windows(events_df, event_labels, window_seconds=ws)

        merged = gf.merge(
            wl[["window_start", "label", "attack_fraction"]],
            on="window_start", how="inner",
        )
        all_X_rows.append(merged)
        n_atk = int(merged["label"].sum())
        n_ben = int((merged["label"] == 0).sum())
        print(f"{len(merged)} windows ({n_atk} atk, {n_ben} ben)")

    data = pd.concat(all_X_rows, ignore_index=True)
    y = data["label"].values

    # Identify feature groups
    all_feat_cols = [c for c in data.columns if c not in META_COLS]
    cd_feat_cols = [c for c in all_feat_cols if c in CD_COLS]
    agg_feat_cols = [c for c in all_feat_cols if c not in CD_COLS]

    print(f"\n  Total features:     {len(all_feat_cols)}")
    print(f"  CD features:        {cd_feat_cols}")
    print(f"  Aggregate features: {agg_feat_cols}")
    print(f"  Total windows:      {len(data)} ({int(y.sum())} atk, {int((y==0).sum())} ben)")

    # ============================================================
    # Three-way ablation
    # ============================================================
    print(f"\n{'='*60}")
    print("ABLATION RESULTS")
    print(f"{'='*60}\n")

    # A: Full (all features)
    X_full = data[all_feat_cols].values
    auc_full, std_full, imp_full = run_classifier(X_full, y, "A. FULL (all)")

    # B: Aggregates only (no CD)
    X_agg = data[agg_feat_cols].values
    auc_agg, std_agg, imp_agg = run_classifier(X_agg, y, "B. NO-CD (agg only)")

    # C: CD only
    X_cd = data[cd_feat_cols].values
    auc_cd, std_cd, imp_cd = run_classifier(X_cd, y, "C. CD-ONLY")

    # ============================================================
    # Analysis
    # ============================================================
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    cd_contribution = auc_full - auc_agg
    print(f"\n  AUC Full:           {auc_full:.4f}")
    print(f"  AUC No-CD:          {auc_agg:.4f}")
    print(f"  AUC CD-Only:        {auc_cd:.4f}")
    print(f"  CD contribution:    {cd_contribution:+.4f}")

    # Feature importances for full model
    print(f"\n  Feature importances (FULL model):")
    feat_imp = sorted(zip(all_feat_cols, imp_full), key=lambda x: -x[1])
    for name, imp in feat_imp:
        marker = " ← CD" if name in CD_COLS else ""
        print(f"    {name:25s} {imp:.4f}{marker}")

    # Verdict
    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")

    if auc_agg >= 0.99:
        print(f"\n  ⚠ CONFOUND DETECTED")
        print(f"  Aggregate-only AUC ({auc_agg:.4f}) ≈ Full AUC ({auc_full:.4f})")
        print(f"  CD components add negligible signal.")
        print(f"  The uplift is driven by volume/size, not coordination.")
        print(f"  → Kill switch is a FALSE POSITIVE.")
    elif auc_agg < 0.95:
        print(f"\n  ✓ CD IS THE ACTIVE INGREDIENT")
        print(f"  Removing CD drops AUC from {auc_full:.4f} → {auc_agg:.4f}")
        print(f"  CD contribution: {cd_contribution:+.4f}")
        print(f"  → Kill switch is GENUINE.")
    else:
        print(f"\n  ~ PARTIAL CONTRIBUTION")
        print(f"  CD contributes {cd_contribution:+.4f} on top of aggregates.")
        print(f"  Aggregates alone: {auc_agg:.4f}, Full: {auc_full:.4f}")
        if cd_contribution > 0.01:
            print(f"  → CD adds real but modest signal. Proceed with caution.")
        else:
            print(f"  → CD contribution marginal. Need stronger evidence.")


if __name__ == "__main__":
    main()
