"""
Experiment 3: Dilution Resistance — "Benign Flood" Test

Measures how well each model retains attack-relevant state ("taint") across
intervals of benign activity. Three complementary analyses:

  Analysis A — Temporal Gap:  Recall vs wall-clock time since the entity's
                              last attack interaction (seconds → hours).
  Analysis B — Event Volume:  Recall vs total benign events an entity has
                              ever participated in. High-volume entities
                              (PIDs appearing in millions of events) have
                              their state overwritten far more than rare ones.
  Analysis C — First-Taint:   Recall on crossprocess propagation events where
                              the SUBJECT entity has never appeared in an
                              attack before (its taint comes purely from
                              upstream propagation through the state bank).

Design decisions (fixes from v1):
  - Uses SUBJECT entity gap only (slot 0), not min-over-roles, preventing
    one fresh-from-attack object from masking a large subject gap.
  - Includes first-taint events: entities with no prior attack history are
    assigned a gap equal to their total benign event count since first seen.
  - Uses time-based binning (seconds/minutes/hours) which produces well-
    populated buckets even on THEIA's burst-attack pattern.

Usage:
    # Single model
    python -m src.analysis.dilution_experiment \\
        --checkpoint ckpts/full_runs/<run>/best.pt \\
        --model_type hypermamba

    # Compare two models (after running both)
    python -m src.analysis.dilution_experiment \\
        --compare results/dilution/hypermamba/preds.pt results/dilution/gru_tgn/preds.pt \\
        --compare_labels HyperMamba GRU-TGN
"""

import argparse
import logging
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader

from src.data.chrono_dataset import ChronoDataset

DATA_ROOT = Path("data/processed/darpa_tc_e3")
SHARD_CONFIG = {
    "theia": {
        "train": [0, 1, 2, 3, 4, 5, 6, 7, 8],           # Apr 3–5
        "val":   [9, 10],                                   # Apr 5 (late)
        "test":  [11, 12, 13, 17, 18, 19, 20, 21, 22,      # Apr 9–13
                  23, 24, 14, 15, 16],                      # (chronological!)
    },
}


# ---------------------------------------------------------------------------
# Model loading & inference (unchanged from v1)
# ---------------------------------------------------------------------------

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


def _find_threshold(logits, labels):
    """Find the threshold that maximizes F1."""
    valid = labels >= 0
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits[valid], -50, 50)))
    try:
        prec, rec, thres = precision_recall_curve(labels[valid], probs)
        f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
        return float(thres[np.argmax(f1)])
    except ValueError:
        return 0.5


# ---------------------------------------------------------------------------
# Analysis A: Temporal Gap (recall vs wall-clock time since last attack)
# ---------------------------------------------------------------------------

def compute_temporal_gap_curve(preds_dict):
    """For each attack event, measure the subject entity's TIME gap since
    its last attack interaction. Includes first-taint events.

    Returns list of dicts with: event_idx, time_gap, benign_gap, pred_correct,
                                is_first_taint
    """
    logits = preds_dict["logits"]
    labels = preds_dict["labels"]
    entity_ids = preds_dict["entity_ids"]
    timestamps = preds_dict["timestamps"]

    best_thresh = _find_threshold(logits, labels)
    preds = (logits >= best_thresh).astype(int)

    # Per-entity tracking
    last_attack_time = {}          # entity_id → timestamp of last attack event
    last_attack_idx = {}           # entity_id → index of last attack event
    benign_count_since_attack = defaultdict(int)  # entity_id → count
    total_benign_count = defaultdict(int)          # entity_id → lifetime count
    first_seen_time = {}           # entity_id → timestamp of first appearance

    results = []

    for i in range(len(labels)):
        is_attack = (labels[i] == 1)

        # Subject entity is always slot 0
        subj_id = int(entity_ids[i][0])
        if subj_id < 0:
            # Invalid subject — skip
            if is_attack:
                pass  # can't analyze
            continue

        # Track first seen
        if subj_id not in first_seen_time:
            first_seen_time[subj_id] = timestamps[i]

        if is_attack:
            if subj_id in last_attack_time:
                # Recurring attack on this entity
                time_gap = timestamps[i] - last_attack_time[subj_id]
                benign_gap = benign_count_since_attack[subj_id]
                is_first = False
            else:
                # FIRST-TAINT: entity's first attack event
                # Gap = time since first seen, benign_gap = total benign events
                time_gap = timestamps[i] - first_seen_time[subj_id]
                benign_gap = total_benign_count[subj_id]
                is_first = True

            results.append({
                "event_idx": i,
                "time_gap": float(time_gap),
                "benign_gap": int(benign_gap),
                "pred_correct": int(preds[i] == 1),
                "is_first_taint": is_first,
            })

            # Update tracking
            last_attack_time[subj_id] = timestamps[i]
            last_attack_idx[subj_id] = i
            benign_count_since_attack[subj_id] = 0

        else:
            # Benign event: increment counters
            if subj_id in last_attack_time:
                benign_count_since_attack[subj_id] += 1
            total_benign_count[subj_id] += 1

    return results, best_thresh


def bin_temporal(results):
    """Bin by time gap (seconds) and by benign event gap."""
    df = pd.DataFrame(results)

    # Time-based bins
    time_bins = [-0.001, 1, 60, 3600, 86400, np.inf]
    time_labels = ["<1s", "1s–1min", "1min–1hr", "1hr–24hr", ">24hr"]
    df["time_bucket"] = pd.cut(df["time_gap"], bins=time_bins, labels=time_labels)

    time_summary = df.groupby("time_bucket", observed=False).agg(
        recall=("pred_correct", "mean"),
        count=("pred_correct", "count"),
    )

    # Benign-event bins (with first-taint events now included)
    event_bins = [-1, 0, 10, 50, 200, 1000, np.inf]
    event_labels = ["0", "1–10", "11–50", "51–200", "201–1K", ">1K"]
    df["event_bucket"] = pd.cut(df["benign_gap"], bins=event_bins, labels=event_labels)

    event_summary = df.groupby("event_bucket", observed=False).agg(
        recall=("pred_correct", "mean"),
        count=("pred_correct", "count"),
    )

    # First-taint only
    ft = df[df["is_first_taint"]]
    first_taint_summary = {
        "count": len(ft),
        "recall": float(ft["pred_correct"].mean()) if len(ft) > 0 else float("nan"),
    }

    return df, time_summary, event_summary, first_taint_summary


# ---------------------------------------------------------------------------
# Analysis B: Entity Volume (recall vs total lifetime benign events)
# ---------------------------------------------------------------------------

def compute_entity_volume_curve(preds_dict):
    """For each entity that appears in at least one attack event, compute:
      - total_events: how many total events (benign+attack) this entity
        participated in across the entire test set
      - attack_recall: fraction of its attack events correctly detected

    High-volume entities have their state overwritten by far more benign
    updates — this directly measures dilution resistance.
    """
    logits = preds_dict["logits"]
    labels = preds_dict["labels"]
    entity_ids = preds_dict["entity_ids"]

    best_thresh = _find_threshold(logits, labels)
    preds = (logits >= best_thresh).astype(int)

    # Count per-entity total events and attack performance
    entity_total = defaultdict(int)       # entity → total event count
    entity_attack_correct = defaultdict(int)
    entity_attack_total = defaultdict(int)

    for i in range(len(labels)):
        subj_id = int(entity_ids[i][0])
        if subj_id < 0:
            continue

        entity_total[subj_id] += 1

        if labels[i] == 1:
            entity_attack_total[subj_id] += 1
            if preds[i] == 1:
                entity_attack_correct[subj_id] += 1

    # Build per-entity results
    results = []
    for eid in entity_attack_total:
        results.append({
            "entity_id": eid,
            "total_events": entity_total[eid],
            "attack_events": entity_attack_total[eid],
            "correct": entity_attack_correct[eid],
            "recall": entity_attack_correct[eid] / entity_attack_total[eid],
        })

    return results


def bin_entity_volume(results):
    """Bin entities by total event volume."""
    df = pd.DataFrame(results)

    vol_bins = [0, 100, 1000, 10000, 100000, np.inf]
    vol_labels = ["<100", "100–1K", "1K–10K", "10K–100K", ">100K"]
    df["volume_bucket"] = pd.cut(df["total_events"], bins=vol_bins, labels=vol_labels)

    # Weight recall by attack events (not by entity count)
    def weighted_recall(group):
        return group["correct"].sum() / max(group["attack_events"].sum(), 1)

    summary = df.groupby("volume_bucket", observed=False).apply(
        weighted_recall
    ).rename("recall").to_frame()

    counts = df.groupby("volume_bucket", observed=False).agg(
        n_entities=("entity_id", "count"),
        n_attacks=("attack_events", "sum"),
    )
    summary = summary.join(counts)

    return df, summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_temporal_comparison(summaries, model_labels, out_path):
    """Bar chart: recall by time-gap bucket, side-by-side models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bucket_labels = ["<1s", "1s–1min", "1min–1hr", "1hr–24hr", ">24hr"]
    x = np.arange(len(bucket_labels))
    width = 0.8 / len(summaries)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    for idx, (summary, label) in enumerate(zip(summaries, model_labels)):
        recalls = []
        counts = []
        for b in bucket_labels:
            if b in summary.index:
                r = summary.loc[b, "recall"]
                recalls.append(r if not np.isnan(r) else 0.0)
                counts.append(int(summary.loc[b, "count"]))
            else:
                recalls.append(0.0)
                counts.append(0)

        offset = (idx - len(summaries) / 2 + 0.5) * width
        bars = ax.bar(x + offset, recalls, width, label=label,
                      color=colors[idx % len(colors)], alpha=0.85, edgecolor="white")
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"n={c:,}", ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xlabel("Time Since Subject's Last Attack Event", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Analysis A: Recall vs Temporal Gap (Subject Entity)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logging.info(f"  Temporal gap plot saved: {out_path}")


def plot_volume_comparison(summaries, model_labels, out_path):
    """Bar chart: recall by entity volume bucket."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bucket_labels = ["<100", "100–1K", "1K–10K", "10K–100K", ">100K"]
    x = np.arange(len(bucket_labels))
    width = 0.8 / len(summaries)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    for idx, (summary, label) in enumerate(zip(summaries, model_labels)):
        recalls = []
        n_attacks_list = []
        for b in bucket_labels:
            if b in summary.index:
                r = summary.loc[b, "recall"]
                recalls.append(r if not np.isnan(r) else 0.0)
                n_attacks_list.append(int(summary.loc[b, "n_attacks"]))
            else:
                recalls.append(0.0)
                n_attacks_list.append(0)

        offset = (idx - len(summaries) / 2 + 0.5) * width
        bars = ax.bar(x + offset, recalls, width, label=label,
                      color=colors[idx % len(colors)], alpha=0.85, edgecolor="white")
        for bar, n in zip(bars, n_attacks_list):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"n={n:,}", ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xlabel("Entity Lifetime Event Volume (Total Events in Test Set)", fontsize=12)
    ax.set_ylabel("Recall (weighted by attack events)", fontsize=12)
    ax.set_title("Analysis B: Recall vs Entity Event Volume", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logging.info(f"  Volume plot saved: {out_path}")


def plot_event_gap_comparison(summaries, model_labels, out_path):
    """Bar chart: recall by benign-event-gap bucket (with first-taint)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bucket_labels = ["0", "1–10", "11–50", "51–200", "201–1K", ">1K"]
    x = np.arange(len(bucket_labels))
    width = 0.8 / len(summaries)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    for idx, (summary, label) in enumerate(zip(summaries, model_labels)):
        recalls = []
        counts = []
        for b in bucket_labels:
            if b in summary.index:
                r = summary.loc[b, "recall"]
                recalls.append(r if not np.isnan(r) else 0.0)
                counts.append(int(summary.loc[b, "count"]))
            else:
                recalls.append(0.0)
                counts.append(0)

        offset = (idx - len(summaries) / 2 + 0.5) * width
        bars = ax.bar(x + offset, recalls, width, label=label,
                      color=colors[idx % len(colors)], alpha=0.85, edgecolor="white")
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"n={c:,}", ha="center", va="bottom", fontsize=7, color="gray")

    ax.set_xlabel("Benign Events Since Subject's Last Attack", fontsize=12)
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_title("Analysis C: Recall vs Benign Event Gap (incl. First-Taint)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logging.info(f"  Event gap plot saved: {out_path}")


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyze_single_preds(preds, label):
    """Run all three analyses on a single model's predictions."""
    # Analysis A & C: Temporal + event gap + first-taint
    results_ac, thresh = compute_temporal_gap_curve(preds)
    df_ac, time_summary, event_summary, ft_summary = bin_temporal(results_ac)

    # Analysis B: Entity volume
    results_b = compute_entity_volume_curve(preds)
    df_b, vol_summary = bin_entity_volume(results_b)

    # Log results
    logging.info(f"\n{'='*60}")
    logging.info(f"  {label} — Optimal threshold: {thresh:.4f}")
    logging.info(f"  Total attack events analyzed: {len(results_ac)}")

    logging.info(f"\n  Analysis A — Recall by Temporal Gap:")
    for idx, row in time_summary.iterrows():
        r = f"{row['recall']:.4f}" if not np.isnan(row['recall']) else "  n/a "
        logging.info(f"    {idx:>12}: Recall {r} (n={int(row['count']):,})")

    logging.info(f"\n  Analysis B — Recall by Entity Volume:")
    for idx, row in vol_summary.iterrows():
        r = f"{row['recall']:.4f}" if not np.isnan(row['recall']) else "  n/a "
        na = int(row['n_attacks']) if not np.isnan(row['n_attacks']) else 0
        ne = int(row['n_entities']) if not np.isnan(row['n_entities']) else 0
        logging.info(f"    {idx:>12}: Recall {r} ({ne} entities, {na:,} attacks)")

    logging.info(f"\n  Analysis C — First-Taint Events:")
    logging.info(f"    Count: {ft_summary['count']}")
    r = f"{ft_summary['recall']:.4f}" if not np.isnan(ft_summary['recall']) else "n/a"
    logging.info(f"    Recall: {r}")

    logging.info(f"{'='*60}")

    return {
        "time_summary": time_summary,
        "event_summary": event_summary,
        "vol_summary": vol_summary,
        "ft_summary": ft_summary,
        "df_temporal": df_ac,
        "df_volume": df_b,
    }


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

    # Compute overall metrics
    valid = preds["labels"] >= 0
    probs = 1.0 / (1.0 + np.exp(-np.clip(preds["logits"][valid], -50, 50)))
    auprc = average_precision_score(preds["labels"][valid], probs)
    logging.info(f"  Test AUPRC: {auprc:.4f}")

    analysis = analyze_single_preds(preds, args.model_type)

    # Save CSVs
    analysis["df_temporal"].to_csv(out_dir / "temporal_results.csv", index=False)
    analysis["df_volume"].to_csv(out_dir / "volume_results.csv", index=False)

    # Single-model plots
    plot_temporal_comparison([analysis["time_summary"]], [args.model_type],
                            out_dir / "temporal_gap.png")
    plot_volume_comparison([analysis["vol_summary"]], [args.model_type],
                           out_dir / "entity_volume.png")
    plot_event_gap_comparison([analysis["event_summary"]], [args.model_type],
                              out_dir / "event_gap.png")


def run_comparison(args):
    """Compare dilution curves for multiple models."""
    labels = args.compare_labels if args.compare_labels else \
        [f"Model {i+1}" for i in range(len(args.compare))]

    time_summaries = []
    event_summaries = []
    vol_summaries = []

    for pred_path, label in zip(args.compare, labels):
        logging.info(f"\nLoading predictions from {pred_path} ({label})...")
        preds = torch.load(pred_path, weights_only=False)
        analysis = analyze_single_preds(preds, label)
        time_summaries.append(analysis["time_summary"])
        event_summaries.append(analysis["event_summary"])
        vol_summaries.append(analysis["vol_summary"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_temporal_comparison(time_summaries, labels,
                            out_dir / "comparison_temporal_gap.png")
    plot_volume_comparison(vol_summaries, labels,
                           out_dir / "comparison_entity_volume.png")
    plot_event_gap_comparison(event_summaries, labels,
                              out_dir / "comparison_event_gap.png")

    logging.info(f"\n  All comparison plots saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Dilution Resistance Analysis (v2)")

    # Single model mode
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, choices=["hypermamba", "gru_tgn"],
                        help="Model type to load")

    # Comparison mode
    parser.add_argument("--compare", nargs="+",
                        help="Paths to prediction .pt files to compare")
    parser.add_argument("--compare_labels", nargs="+",
                        help="Labels for comparison models")

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
