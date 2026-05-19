"""
Supervised KAIROS baseline: train a logistic regression classifier on
KAIROS GNN embeddings for fair comparison with THyN.

Input:  Per-event embeddings from extract_embeddings.py
Output: AUPRC, AUROC, F1 — comparable to THyN results

Usage:
    python -m baselines.kairos.supervised_head --dataset theia
    python -m baselines.kairos.supervised_head --dataset trace
    python -m baselines.kairos.supervised_head --dataset theia --max-train 5000000
"""

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_fscore_support, precision_recall_curve,
)

from baselines.kairos.config import DATASET_CONFIGS, ARTIFACT_DIR


def find_best_f1_threshold(probs, labels):
    """Find threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = np.argmax(f1)
    return thresholds[best_idx], f1[best_idx]


def evaluate_split(name, probs, labels, threshold=None):
    """Evaluate predictions on a split."""
    valid = labels >= 0
    probs = probs[valid]
    labels = labels[valid]

    n_attack = int(labels.sum())
    n_total = len(labels)

    print(f"\n  {name} Evaluation")
    print(f"    Total events:  {n_total:,}")
    print(f"    Attack events: {n_attack:,} ({100*n_attack/n_total:.2f}%)")

    if n_attack == 0 or n_attack == n_total:
        print(f"    SKIPPED: cannot compute metrics")
        return {}

    auprc = average_precision_score(labels, probs)
    auroc = roc_auc_score(labels, probs)

    results = {
        f"{name.lower()}_auprc": auprc,
        f"{name.lower()}_auroc": auroc,
        f"{name.lower()}_n_events": n_total,
        f"{name.lower()}_n_attack": n_attack,
    }

    print(f"    AUPRC: {auprc:.4f}")
    print(f"    AUROC: {auroc:.4f}")

    # F1 with provided or best threshold
    if threshold is None:
        threshold, best_f1 = find_best_f1_threshold(probs, labels)
        print(f"    Best F1: {best_f1:.4f} (threshold={threshold:.4f})")
        results[f"{name.lower()}_best_f1"] = best_f1
        results[f"{name.lower()}_threshold"] = threshold
    else:
        preds = (probs > threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0)
        print(f"    F1: {f1:.4f} (P={p:.4f}, R={r:.4f}, threshold={threshold:.4f})")
        results[f"{name.lower()}_f1"] = f1
        results[f"{name.lower()}_precision"] = p
        results[f"{name.lower()}_recall"] = r

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Supervised KAIROS: logistic regression on GNN embeddings")
    parser.add_argument("--dataset", default="theia",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--max-train", type=int, default=5_000_000,
                        help="Max training examples (downsample if larger)")
    parser.add_argument("--include-loss", action="store_true", default=True,
                        help="Include reconstruction loss as extra feature")
    parser.add_argument("--no-include-loss", action="store_false",
                        dest="include_loss")
    args = parser.parse_args()

    emb_dir = ARTIFACT_DIR / "embeddings" / args.dataset
    models_dir = ARTIFACT_DIR / "models" / args.dataset

    print("=" * 60)
    print(f"SUPERVISED KAIROS — {args.dataset.upper()}")
    print("=" * 60)

    # ========================================================
    # Load embeddings
    # ========================================================
    print("\n[1/4] Loading embeddings...")

    splits = {}
    for split in ["train", "val", "test"]:
        path = emb_dir / f"{split}_embeddings.npz"
        if not path.exists():
            print(f"  {split}: NOT FOUND at {path}")
            continue

        data = np.load(path)
        emb = data["embeddings"]   # (N, EDGE_DIM*2)
        lab = data["labels"]       # (N,)
        loss = data["losses"]      # (N,)

        # Optionally append reconstruction loss as feature
        if args.include_loss:
            # Clip extreme losses for stability
            loss_clipped = np.clip(loss, 0, np.percentile(loss, 99.9))
            emb = np.hstack([emb, loss_clipped.reshape(-1, 1)])

        n_atk = int((lab == 1).sum())
        print(f"  {split}: {len(lab):,} events, "
              f"{n_atk:,} attack ({100*n_atk/len(lab):.1f}%), "
              f"emb_dim={emb.shape[1]}")

        splits[split] = {"X": emb, "y": lab}
        del data

    if "train" not in splits:
        print("ERROR: No training embeddings found. Run extract_embeddings first.")
        return

    # ========================================================
    # Downsample training data
    # ========================================================
    print(f"\n[2/4] Preparing training data (max {args.max_train:,})...")

    X_train = splits["train"]["X"]
    y_train = splits["train"]["y"]

    # Filter padding
    valid = y_train >= 0
    X_train = X_train[valid]
    y_train = y_train[valid]

    if len(X_train) > args.max_train:
        # Stratified downsample
        rng = np.random.RandomState(42)
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos

        # Take equal proportion from each class
        n_sample = args.max_train
        pos_ratio = n_pos / len(y_train)
        n_pos_sample = min(int(n_sample * pos_ratio), n_pos)
        n_neg_sample = min(n_sample - n_pos_sample, n_neg)

        pos_idx = rng.choice(np.where(y_train == 1)[0], n_pos_sample, replace=False)
        neg_idx = rng.choice(np.where(y_train == 0)[0], n_neg_sample, replace=False)
        idx = np.sort(np.concatenate([pos_idx, neg_idx]))

        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"  Downsampled: {len(y_train):,} events "
              f"({int(y_train.sum()):,} attack / "
              f"{int((y_train==0).sum()):,} benign)")
    else:
        print(f"  Using all: {len(y_train):,} events")

    # ========================================================
    # Standardize features
    # ========================================================
    print("\n[3/4] Training logistic regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Handle NaN/Inf from scaling
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # ========================================================
    # Train LogisticRegressionCV
    # ========================================================
    t0 = time.time()
    clf = LogisticRegressionCV(
        Cs=10,                    # 10 values of C to try
        cv=5,                     # 5-fold cross-validation
        scoring="average_precision",  # optimize for AUPRC
        solver="saga",            # fast for large datasets
        max_iter=500,
        n_jobs=-1,                # use all cores
        random_state=42,
        verbose=0,
    )
    clf.fit(X_train_scaled, y_train)
    train_time = time.time() - t0

    print(f"  Training time: {train_time:.1f}s")
    print(f"  Best C: {clf.C_[0]:.6f}")
    print(f"  CV scores: {clf.scores_[1].mean(axis=0).max():.4f} (best mean AUPRC)")

    # ========================================================
    # Evaluate on all splits
    # ========================================================
    print(f"\n[4/4] Evaluating...")
    all_results = {"dataset": args.dataset}
    val_threshold = None

    for split in ["train", "val", "test"]:
        if split not in splits:
            continue

        X = splits[split]["X"]
        y = splits[split]["y"]

        # Filter valid
        valid_mask = y >= 0
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]

        X_scaled = scaler.transform(X_valid)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        probs = clf.predict_proba(X_scaled)[:, 1]

        if split == "val":
            results = evaluate_split(split.upper(), probs, y_valid)
            # Save threshold for test
            if f"val_threshold" in results:
                val_threshold = results["val_threshold"]
        elif split == "test" and val_threshold is not None:
            # Evaluate test with val-optimized threshold
            results = evaluate_split(split.upper(), probs, y_valid,
                                     threshold=val_threshold)
            # Also compute best-threshold F1 for reference
            _, best_f1 = find_best_f1_threshold(probs, y_valid)
            results["test_best_f1"] = best_f1
            print(f"    Best possible F1: {best_f1:.4f}")
        else:
            results = evaluate_split(split.upper(), probs, y_valid)

        all_results.update(results)

    # ========================================================
    # Summary
    # ========================================================
    print(f"\n{'='*60}")
    print(f"SUPERVISED KAIROS RESULTS — {args.dataset.upper()}")
    print(f"{'='*60}")

    for key in ["val_auprc", "val_auroc", "test_auprc", "test_auroc",
                "test_f1", "test_best_f1"]:
        if key in all_results:
            print(f"  {key}: {all_results[key]:.4f}")

    # Save results
    import torch
    results_path = models_dir / "supervised_results.pt"
    torch.save(all_results, results_path)
    print(f"\nResults saved to {results_path}")

    # Also save the classifier for reproducibility
    import joblib
    clf_path = models_dir / "supervised_clf.joblib"
    joblib.dump({"clf": clf, "scaler": scaler}, clf_path)
    print(f"Classifier saved to {clf_path}")


if __name__ == "__main__":
    main()
