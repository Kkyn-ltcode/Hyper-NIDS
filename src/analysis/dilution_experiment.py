"""
Experiment 3: Dilution Resistance — "Benign Flood" Test

This script runs a trained model (HyperMamba or GRU-TGN) on the test set,
saves per-event predictions, and then performs dilution analysis: measuring
detection recall as a function of the benign event gap since each entity's
last attack interaction.

The core hypothesis: HyperMamba's Selective SSM resists state dilution from
benign noise better than GRU-based models, maintaining higher recall even
when hundreds of benign events have flooded an entity's state.

Workflow:
    1. Load trained checkpoint
    2. Warm up entity bank on train+val shards (eval mode)
    3. Run inference on test shards, collecting per-event predictions
    4. Compute dilution curves: recall vs benign-event-gap
    5. Save predictions .pt file + dilution results .csv + comparison plot

Usage:
    # Run for HyperMamba
    python -m src.analysis.dilution_experiment \\
        --checkpoint runs/your_hypermamba_run/best.pt \\
        --model_type hypermamba

    # Run for GRU-TGN
    python -m src.analysis.dilution_experiment \\
        --checkpoint runs/your_gru_tgn_run/best_model.pt \\
        --model_type gru_tgn

    # Compare two models (after running both)
    python -m src.analysis.dilution_experiment \\
        --compare results/dilution/hypermamba/preds.pt results/dilution/gru_tgn/preds.pt \\
        --compare_labels HyperMamba GRU-TGN
"""

import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader

from src.data.chrono_dataset import ChronoDataset

DATA_ROOT = Path("data/processed/darpa_tc_e3")
SHARD_CONFIG = {
    "theia": {"train": list(range(7)), "val": [7], "test": [8, 9]},
}


def load_model(checkpoint_path, model_type, train_ds, device):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt

    if model_type == "hypermamba":
        from src.model.hypermamba_full import HyperMambaFull
        model = HyperMambaFull(
            num_entities=train_ds.num_entities,
            n_cont_features=train_ds.n_cont_features,
            num_event_types=train_ds.num_event_types,
            d_model=256,
        ).to(device)
    elif model_type == "gru_tgn":
        from src.baselines.gru_tgn import GRUTGNBaseline
        model = GRUTGNBaseline(
            num_entities=train_ds.num_entities,
            n_cont_features=train_ds.n_cont_features,
            num_event_types=train_ds.num_event_types,
            d_model=256,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # strict=False handles minor architecture changes
    model.load_state_dict(state_dict, strict=False)
    return model


@torch.no_grad()
def warmup_bank(model, loaders, device):
    """Process data shards in eval mode to warm up entity states."""
    model.eval()
    model.reset_bank()
    total = 0
    for loader in loaders:
        for batch in loader:
            X_c = torch.nan_to_num(batch["X_cont"].to(device), nan=0.0).clamp(-20, 20)
            et = batch["event_type"].to(device)
            ent = batch["entity_ids"].to(device)
            ts = batch["timestamp"].to(device)
            _ = model(X_c, et, ent, ts)
            model.detach_bank()
            total += X_c.size(1)
    logging.info(f"  Bank warmup: {total:,} events")


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run inference and collect per-event predictions with metadata."""
    model.eval()

    all_logits = []
    all_labels = []
    all_entity_ids = []
    all_timestamps = []

    for batch in loader:
        X_c = torch.nan_to_num(batch["X_cont"].to(device), nan=0.0).clamp(-20, 20)
        et = batch["event_type"].to(device)
        y = batch["y"].to(device).float()
        ent = batch["entity_ids"].to(device)
        ts = batch["timestamp"].to(device)

        logits = model(X_c, et, ent, ts)
        model.detach_bank()

        all_logits.extend(logits.squeeze(0).cpu().numpy().tolist())
        all_labels.extend(y.squeeze(0).cpu().numpy().tolist())
        all_entity_ids.append(ent.squeeze(0).cpu().numpy())
        all_timestamps.extend(ts.squeeze(0).cpu().numpy().tolist())

    return {
        "logits": np.array(all_logits, dtype=np.float32),
        "labels": np.array(all_labels, dtype=np.int32),
        "entity_ids": np.concatenate(all_entity_ids, axis=0),
        "timestamps": np.array(all_timestamps, dtype=np.float32),
    }


def compute_dilution_curve(preds_dict):
    """Compute recall as a function of benign-event gap.

    For each attack event, we find the minimum benign-event-gap among its
    participating entities (min over subj/obj/obj2). The gap is the number
    of benign events that entity participated in since its last attack event.

    Returns a DataFrame with columns: event_idx, benign_gap, time_gap, pred_correct
    """
    logits = preds_dict["logits"]
    labels = preds_dict["labels"]
    entity_ids = preds_dict["entity_ids"]
    timestamps = preds_dict["timestamps"]

    # Find optimal threshold
    valid = labels >= 0
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits[valid], -50, 50)))
    try:
        prec, rec, thres = precision_recall_curve(labels[valid], probs)
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
        best_thresh = thres[np.argmax(f1)]
    except ValueError:
        best_thresh = 0.5

    preds = (logits >= best_thresh).astype(int)

    # Track per-entity benign count since last attack
    last_attack_idx = {}
    benign_count = defaultdict(int)

    results = []

    for i in range(len(labels)):
        is_attack = (labels[i] == 1)
        ents = [int(e) for e in entity_ids[i] if e >= 0]

        if is_attack:
            gaps = []
            time_gaps = []
            for e in ents:
                if e in last_attack_idx:
                    gaps.append(benign_count[e])
                    time_gaps.append(timestamps[i] - timestamps[last_attack_idx[e]])

            if gaps:
                results.append({
                    "event_idx": i,
                    "benign_gap": min(gaps),
                    "time_gap": min(time_gaps),
                    "pred_correct": int(preds[i] == 1),
                })

            for e in ents:
                last_attack_idx[e] = i
                benign_count[e] = 0
        else:
            for e in ents:
                if e in last_attack_idx:
                    benign_count[e] += 1

    return results, best_thresh


def bin_and_summarize(results):
    """Bin results by benign gap and compute recall per bin."""
    import pandas as pd
    df = pd.DataFrame(results)

    bins = [-1, 0, 10, 50, 200, 1000, np.inf]
    bin_labels = ["0", "1-10", "11-50", "51-200", "201-1K", ">1K"]
    df["gap_bucket"] = pd.cut(df["benign_gap"], bins=bins, labels=bin_labels)

    summary = df.groupby("gap_bucket", observed=False).agg(
        recall=("pred_correct", "mean"),
        count=("pred_correct", "count"),
    )
    return df, summary


def plot_comparison(summaries, labels, out_path):
    """Plot dilution resistance curves for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    bin_labels = ["0", "1-10", "11-50", "51-200", "201-1K", ">1K"]
    x = np.arange(len(bin_labels))
    width = 0.8 / len(summaries)

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    for idx, (summary, label) in enumerate(zip(summaries, labels)):
        recalls = []
        counts = []
        for b in bin_labels:
            if b in summary.index:
                recalls.append(summary.loc[b, "recall"])
                counts.append(int(summary.loc[b, "count"]))
            else:
                recalls.append(0.0)
                counts.append(0)

        offset = (idx - len(summaries) / 2 + 0.5) * width
        bars = ax.bar(x + offset, recalls, width, label=label,
                      color=colors[idx % len(colors)], alpha=0.85, edgecolor="white")

        # Annotate with count
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"n={c}", ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xlabel("Benign Events Since Last Attack (per-entity minimum)", fontsize=12)
    ax.set_ylabel("Recall (True Positive Rate)", fontsize=12)
    ax.set_title("Experiment 3: Dilution Resistance — Recall vs Benign Flood Gap", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logging.info(f"  Plot saved to {out_path}")


def run_single_model(args):
    """Run dilution analysis for a single model."""
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    shards = SHARD_CONFIG[args.dataset]
    data_dir = DATA_ROOT / args.dataset

    logging.info("Loading datasets...")
    train_ds = ChronoDataset(shard_ids=shards["train"], data_root=data_dir,
                             chunk_size=args.chunk_size, label_type=args.label_type)
    val_ds = ChronoDataset(shard_ids=shards["val"], data_root=data_dir,
                           chunk_size=args.chunk_size, label_type=args.label_type,
                           t0_nanos=train_ds.t0_nanos)
    test_ds = ChronoDataset(shard_ids=shards["test"], data_root=data_dir,
                            chunk_size=args.chunk_size, label_type=args.label_type,
                            t0_nanos=train_ds.t0_nanos)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    logging.info(f"Loading {args.model_type} from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model_type, train_ds, device)

    logging.info("Warming up bank...")
    warmup_bank(model, [train_loader, val_loader], device)

    logging.info("Collecting test predictions...")
    preds = collect_predictions(model, test_loader, device)

    out_dir = Path(args.out_dir) / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = out_dir / "preds.pt"
    torch.save(preds, preds_path)
    logging.info(f"  Predictions saved to {preds_path}")

    # Compute metrics
    valid = preds["labels"] >= 0
    probs = 1.0 / (1.0 + np.exp(-np.clip(preds["logits"][valid], -50, 50)))
    auprc = average_precision_score(preds["labels"][valid], probs)
    logging.info(f"  Test AUPRC: {auprc:.4f}")

    results, thresh = compute_dilution_curve(preds)
    logging.info(f"  Optimal threshold: {thresh:.4f}")
    logging.info(f"  Attack events with history: {len(results)}")

    df, summary = bin_and_summarize(results)
    df.to_csv(out_dir / "dilution_results.csv", index=False)

    logging.info("\n  Recall by Benign Event Gap:")
    for idx, row in summary.iterrows():
        logging.info(f"    Gap {idx:>10}: Recall {row['recall']:.4f} (n={int(row['count'])})")

    # Plot single model
    plot_comparison([summary], [args.model_type], out_dir / "dilution_curve.png")

    return preds_path


def run_comparison(args):
    """Compare dilution curves for multiple models."""
    summaries = []
    labels = args.compare_labels if args.compare_labels else \
        [f"Model {i+1}" for i in range(len(args.compare))]

    for pred_path, label in zip(args.compare, labels):
        logging.info(f"Loading predictions from {pred_path} ({label})...")
        preds = torch.load(pred_path)
        results, _ = compute_dilution_curve(preds)
        _, summary = bin_and_summarize(results)
        summaries.append(summary)

        logging.info(f"\n  {label} — Recall by Benign Event Gap:")
        for idx, row in summary.iterrows():
            logging.info(f"    Gap {idx:>10}: Recall {row['recall']:.4f} (n={int(row['count'])})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(summaries, labels, out_dir / "dilution_comparison.png")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Dilution Resistance Analysis")

    # Single model mode
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, choices=["hypermamba", "gru_tgn"],
                        help="Model type to load")

    # Comparison mode
    parser.add_argument("--compare", nargs="+", help="Paths to prediction .pt files to compare")
    parser.add_argument("--compare_labels", nargs="+", help="Labels for comparison models")

    # Common
    parser.add_argument("--dataset", default="theia")
    parser.add_argument("--label_type", default="crossprocess")
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--out_dir", default="results/dilution")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.compare:
        run_comparison(args)
    elif args.checkpoint and args.model_type:
        run_single_model(args)
    else:
        parser.error("Either --checkpoint + --model_type, or --compare is required")


if __name__ == "__main__":
    main()
