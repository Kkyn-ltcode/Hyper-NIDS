"""
Supervised KAIROS baseline: GPU-accelerated linear classifier on
KAIROS GNN embeddings for fair comparison with THyN.

Uses a single-layer PyTorch linear classifier (equivalent to logistic
regression) trained with Adam on GPU. Much faster than sklearn on
large datasets.

Usage:
    python -m baselines.kairos.supervised_head --dataset theia
    python -m baselines.kairos.supervised_head --dataset trace
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_fscore_support, precision_recall_curve,
)

from baselines.kairos.config import DATASET_CONFIGS, ARTIFACT_DIR, GRAPHS_DIR


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


class LinearClassifier(nn.Module):
    """Single linear layer — equivalent to logistic regression."""
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_linear_gpu(X_train, y_train, X_val, y_val, device,
                     epochs=30, batch_size=8192, lr=1e-3, patience=5):
    """Train a linear classifier on GPU with early stopping."""

    in_dim = X_train.shape[1]
    model = LinearClassifier(in_dim).to(device)

    # Class weight for imbalanced data
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device).float()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    # Build dataloaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    best_auprc = 0.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        avg_loss = total_loss / len(X_train)

        # Validate
        model.eval()
        with torch.no_grad():
            val_probs = predict_gpu(model, X_val, device, batch_size=batch_size*2)
        valid = y_val >= 0
        try:
            val_auprc = average_precision_score(y_val[valid], val_probs[valid])
        except ValueError:
            val_auprc = 0.0

        scheduler.step(val_auprc)

        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 5 == 0 or epoch == 1 or wait == 0:
            print(f"    Epoch {epoch:2d}: loss={avg_loss:.4f}, "
                  f"val_auprc={val_auprc:.4f} "
                  f"{'*' if wait == 0 else ''}")

        if wait >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break

    # Restore best model
    model.load_state_dict(best_state)
    print(f"    Best val AUPRC: {best_auprc:.4f}")
    return model


def predict_gpu(model, X, device, batch_size=16384):
    """Run inference on GPU in batches, return numpy probabilities."""
    model.eval()
    all_probs = []
    X_tensor = torch.from_numpy(X).float()

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        batch = X_tensor[start:end].to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs)


def main():
    parser = argparse.ArgumentParser(
        description="Supervised KAIROS: GPU linear classifier on GNN embeddings")
    parser.add_argument("--dataset", default="theia",
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--max-train", type=int, default=5_000_000,
                        help="Max training examples (downsample if larger)")
    parser.add_argument("--include-loss", action="store_true", default=True,
                        help="Include reconstruction loss as extra feature")
    parser.add_argument("--no-include-loss", action="store_false",
                        dest="include_loss")
    parser.add_argument("--train-labels", default=None,
                        help="Use alternate training labels (e.g., 'l1' for L1**). "
                             "Val/test always use broad labels.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    emb_dir = ARTIFACT_DIR / "embeddings" / args.dataset
    models_dir = ARTIFACT_DIR / "models" / args.dataset

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    label_tag = f" (train labels: {args.train_labels})" if args.train_labels else ""
    exp_name = f"supervised_kairos_{args.train_labels}" if args.train_labels else "supervised_kairos"

    print("=" * 60)
    print(f"SUPERVISED KAIROS — {args.dataset.upper()}{label_tag}")
    print("=" * 60)
    print(f"  Device: {device}")
    if args.train_labels:
        print(f"  Train labels: {args.train_labels} (val/test: broad)")

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
        emb = data["embeddings"].astype(np.float32)
        lab = data["labels"]  # broad labels from embeddings
        loss = data["losses"].astype(np.float32)

        if args.include_loss:
            loss_clipped = np.clip(loss, 0, np.percentile(loss, 99.9))
            emb = np.hstack([emb, loss_clipped.reshape(-1, 1)])

        # Override training labels if --train-labels specified
        if split == "train" and args.train_labels:
            alt_label_path = GRAPHS_DIR / args.dataset / f"train_labels_{args.train_labels}.npy"
            if not alt_label_path.exists():
                print(f"  ERROR: {alt_label_path} not found.")
                print(f"  Run: python -m baselines.kairos.relabel_kairos "
                      f"--dataset {args.dataset} --label-type {args.train_labels}")
                return
            lab_alt = np.load(alt_label_path)
            if len(lab_alt) != len(lab):
                print(f"  ERROR: label length mismatch: broad={len(lab):,}, "
                      f"{args.train_labels}={len(lab_alt):,}")
                return
            n_changed = int((lab != lab_alt).sum())
            print(f"  train: swapping to {args.train_labels} labels "
                  f"({n_changed:,} events changed)")
            lab = lab_alt

        n_atk = int((lab == 1).sum())
        print(f"  {split}: {len(lab):,} events, "
              f"{n_atk:,} attack ({100*n_atk/len(lab):.1f}%), "
              f"emb_dim={emb.shape[1]}")

        splits[split] = {"X": emb, "y": lab}
        del data

    if "train" not in splits:
        print("ERROR: No training embeddings found.")
        return

    # ========================================================
    # Prepare training data
    # ========================================================
    print(f"\n[2/4] Preparing training data (max {args.max_train:,})...")

    X_train = splits["train"]["X"]
    y_train = splits["train"]["y"]

    valid = y_train >= 0
    X_train = X_train[valid]
    y_train = y_train[valid]

    if len(X_train) > args.max_train:
        rng = np.random.RandomState(42)
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
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

    # Standardize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std

    # Prepare val
    X_val = splits.get("val", {}).get("X", None)
    y_val = splits.get("val", {}).get("y", None)
    if X_val is not None:
        valid_v = y_val >= 0
        X_val = (X_val[valid_v] - mean) / std
        y_val = y_val[valid_v]
    else:
        # Use last 10% of training as pseudo-val
        n_val = len(X_train) // 10
        X_val = X_train[-n_val:]
        y_val = y_train[-n_val:]
        X_train = X_train[:-n_val]
        y_train = y_train[:-n_val]
        print(f"  No val set found — using last {n_val:,} train events")

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    # ========================================================
    # Train on GPU
    # ========================================================
    print(f"\n[3/4] Training linear classifier on {device}...")
    t0 = time.time()
    model = train_linear_gpu(
        X_train, y_train, X_val, y_val, device,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    train_time = time.time() - t0
    print(f"  Total training time: {train_time:.1f}s")

    # ========================================================
    # Evaluate
    # ========================================================
    print(f"\n[4/4] Evaluating...")
    all_results = {"dataset": args.dataset}
    val_threshold = None

    for split in ["val", "test"]:
        if split not in splits:
            continue

        X = splits[split]["X"]
        y = splits[split]["y"]

        valid_mask = y >= 0
        X_valid = (X[valid_mask] - mean) / std
        X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)
        y_valid = y[valid_mask]

        probs = predict_gpu(model, X_valid, device)

        if split == "val":
            results = evaluate_split(split.upper(), probs, y_valid)
            if "val_threshold" in results:
                val_threshold = results["val_threshold"]
        elif split == "test" and val_threshold is not None:
            results = evaluate_split(split.upper(), probs, y_valid,
                                     threshold=val_threshold)
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

    for key in ["val_auprc", "val_auroc", "val_best_f1",
                "test_auprc", "test_auroc", "test_f1", "test_best_f1"]:
        if key in all_results:
            print(f"  {key}: {all_results[key]:.4f}")

    suffix = f"_{args.train_labels}" if args.train_labels else ""
    results_path = models_dir / f"supervised_results{suffix}.pt"
    torch.save(all_results, results_path)
    print(f"\nResults saved to {results_path}")

    model_path = models_dir / f"supervised_model{suffix}.pt"
    torch.save({"model": model.cpu().state_dict(), "mean": mean, "std": std},
               model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
