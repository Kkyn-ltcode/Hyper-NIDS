"""
Experiment 2: Ablation Study with Training Curves.

Loads history.pt files from 3 ablation variant runs (full, no_cross_entity, no_state)
and produces:
  1. Training curve plot (Val AUPRC & Test AUPRC side-by-side)
  2. Combined 4-panel figure (Train Loss, Val Loss, Val AUPRC, Test AUPRC)
  3. Final metrics comparison table (printed)
  4. LaTeX table (saved to .tex)
  5. Metrics JSON (saved to .json)

Usage (explicit paths):
    python -m src.analysis.experiment2_ablation \\
        --full_history ckpts/full_runs/<full_run>/history.pt \\
        --no_cross_entity_history ckpts/full_runs/<nce_run>/history.pt \\
        --no_state_history ckpts/full_runs/<ns_run>/history.pt \\
        --out_dir results/experiment2

Usage (auto-discover):
    python -m src.analysis.experiment2_ablation \\
        --auto_discover ckpts/full_runs/ \\
        --out_dir results/experiment2
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import json
import os
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANT_META = {
    "full": {
        "label": "Full HyperMamba",
        "color": "#2563EB",
        "marker": "o",
        "linestyle": "-",
        "discover_key": "ablation-full",
    },
    "no_cross_entity": {
        "label": "No Cross-Entity",
        "color": "#DC2626",
        "marker": "s",
        "linestyle": "-",
        "discover_key": "ablation-no_cross_entity",
    },
    "no_state": {
        "label": "No State",
        "color": "#16A34A",
        "marker": "^",
        "linestyle": "-",
        "discover_key": "ablation-no_state",
    },
}

VARIANT_ORDER = ["full", "no_cross_entity", "no_state"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_history(path: str) -> dict:
    """Load a history.pt file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    history = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(history, dict):
        raise ValueError(f"Expected dict in {path}, got {type(history).__name__}")
    return history


def auto_discover_histories(base_dir: str) -> dict[str, str]:
    """Search *base_dir* for subdirectories matching ablation variant names.

    Returns a dict mapping variant key -> path to history.pt.
    """
    base = Path(base_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"Auto-discover base directory not found: {base}")

    found: dict[str, str] = {}
    for variant_key in VARIANT_ORDER:
        pattern = VARIANT_META[variant_key]["discover_key"]
        candidates = sorted(
            [d for d in base.iterdir() if d.is_dir() and pattern in d.name],
            key=lambda p: p.name,
        )
        if not candidates:
            raise FileNotFoundError(
                f"Could not find a subdirectory containing '{pattern}' in {base}"
            )
        # Take the last match (most recent if names are timestamped)
        history_path = candidates[-1] / "history.pt"
        if not history_path.exists():
            raise FileNotFoundError(
                f"Found directory {candidates[-1]} but no history.pt inside it"
            )
        found[variant_key] = str(history_path)
        print(f"  [auto-discover] {variant_key}: {history_path}")

    return found


def _extract_final_metrics(history: dict) -> dict:
    """Extract final-epoch and best metrics from a history dict."""
    val_auprc = history.get("val_auprc", [])
    test_auprc = history.get("test_auprc", [])
    val_f1 = history.get("val_f1", [])
    test_f1 = history.get("test_f1", [])

    return {
        "final_val_auprc": val_auprc[-1] if val_auprc else float("nan"),
        "final_test_auprc": test_auprc[-1] if test_auprc else float("nan"),
        "best_val_auprc": max(val_auprc) if val_auprc else float("nan"),
        "best_test_auprc": max(test_auprc) if test_auprc else float("nan"),
        "best_val_f1": max(val_f1) if val_f1 else float("nan"),
        "best_test_f1": max(test_f1) if test_f1 else float("nan"),
        "num_epochs": len(val_auprc),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _configure_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent styling to an axis."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(fontsize=11, framealpha=0.9, edgecolor="#cccccc")


def _plot_variant_curve(
    ax, epochs: list[int], values: list[float], variant_key: str, markevery: int = 1,
) -> None:
    """Plot a single variant's curve on the given axis."""
    meta = VARIANT_META[variant_key]
    ax.plot(
        epochs,
        values,
        color=meta["color"],
        linestyle=meta["linestyle"],
        marker=meta["marker"],
        markersize=5,
        markevery=markevery,
        linewidth=1.8,
        label=meta["label"],
        alpha=0.9,
    )


def _compute_markevery(num_epochs: int) -> int:
    """Choose a marker interval so plots aren't too cluttered."""
    if num_epochs <= 20:
        return 1
    elif num_epochs <= 50:
        return 2
    elif num_epochs <= 100:
        return 5
    else:
        return max(1, num_epochs // 20)


def plot_training_curves(
    histories: dict[str, dict], out_dir: str,
) -> None:
    """Create the key 2-panel figure: Val AUPRC & Test AUPRC vs Epoch."""
    fig, (ax_val, ax_test) = plt.subplots(1, 2, figsize=(12, 4.8))

    for variant_key in VARIANT_ORDER:
        h = histories[variant_key]
        val_auprc = h.get("val_auprc", [])
        test_auprc = h.get("test_auprc", [])
        num_epochs = max(len(val_auprc), len(test_auprc))
        if num_epochs == 0:
            continue
        markevery = _compute_markevery(num_epochs)

        if val_auprc:
            epochs_val = list(range(1, len(val_auprc) + 1))
            _plot_variant_curve(ax_val, epochs_val, val_auprc, variant_key, markevery)

        if test_auprc:
            epochs_test = list(range(1, len(test_auprc) + 1))
            _plot_variant_curve(ax_test, epochs_test, test_auprc, variant_key, markevery)

    _configure_ax(ax_val, "Val AUPRC vs Epoch", "Epoch", "AUPRC")
    _configure_ax(ax_test, "Test AUPRC vs Epoch", "Epoch", "AUPRC")

    # Force integer x-ticks
    for ax in (ax_val, ax_test):
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.suptitle(
        "Ablation Study: Training Curves",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    save_path = os.path.join(out_dir, "ablation_training_curves.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_full_curves(
    histories: dict[str, dict], out_dir: str,
) -> None:
    """Create the 4-panel figure: Train Loss, Val Loss, Val AUPRC, Test AUPRC."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax_tl, ax_vl), (ax_va, ax_ta) = axes

    panel_specs = [
        (ax_tl, "train_loss", "Train Loss vs Epoch", "Loss"),
        (ax_vl, "val_loss", "Val Loss vs Epoch", "Loss"),
        (ax_va, "val_auprc", "Val AUPRC vs Epoch", "AUPRC"),
        (ax_ta, "test_auprc", "Test AUPRC vs Epoch", "AUPRC"),
    ]

    for ax, key, title, ylabel in panel_specs:
        for variant_key in VARIANT_ORDER:
            h = histories[variant_key]
            values = h.get(key, [])
            if not values:
                continue
            epochs = list(range(1, len(values) + 1))
            markevery = _compute_markevery(len(values))
            _plot_variant_curve(ax, epochs, values, variant_key, markevery)

        _configure_ax(ax, title, "Epoch", ylabel)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.suptitle(
        "Ablation Study: Full Training Curves",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    save_path = os.path.join(out_dir, "ablation_full_curves.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def print_metrics_table(all_metrics: dict[str, dict]) -> None:
    """Print a human-readable comparison table to stdout."""
    header = f"{'Variant':<22} {'Val AUPRC':>10} {'Test AUPRC':>11} {'Val F1':>8} {'Test F1':>8}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for variant_key in VARIANT_ORDER:
        m = all_metrics[variant_key]
        label = VARIANT_META[variant_key]["label"]
        print(
            f"{label:<22} "
            f"{m['best_val_auprc']:>10.4f} "
            f"{m['best_test_auprc']:>11.4f} "
            f"{m['best_val_f1']:>8.4f} "
            f"{m['best_test_f1']:>8.4f}"
        )
    print(sep + "\n")


def _bold_best(values: list[float], fmt: str = ".4f") -> list[str]:
    """Return formatted strings; bold the maximum value."""
    best_val = max(values)
    result = []
    for v in values:
        s = f"{v:{fmt}}"
        if v == best_val:
            s = r"\textbf{" + s + "}"
        result.append(s)
    return result


def generate_latex_table(all_metrics: dict[str, dict], out_dir: str) -> None:
    """Generate and save a LaTeX table."""
    # Gather column values in variant order
    val_auprcs = [all_metrics[k]["best_val_auprc"] for k in VARIANT_ORDER]
    test_auprcs = [all_metrics[k]["best_test_auprc"] for k in VARIANT_ORDER]
    val_f1s = [all_metrics[k]["best_val_f1"] for k in VARIANT_ORDER]
    test_f1s = [all_metrics[k]["best_test_f1"] for k in VARIANT_ORDER]

    # Format with bolding
    fmt_val_auprc = _bold_best(val_auprcs)
    fmt_test_auprc = _bold_best(test_auprcs)
    fmt_val_f1 = _bold_best(val_f1s)
    fmt_test_f1 = _bold_best(test_f1s)

    rows = []
    for i, variant_key in enumerate(VARIANT_ORDER):
        label = VARIANT_META[variant_key]["label"]
        row = (
            f"        {label} & {fmt_val_auprc[i]} & {fmt_test_auprc[i]} "
            f"& {fmt_val_f1[i]} & {fmt_test_f1[i]} \\\\"
        )
        rows.append(row)

    latex = (
        r"\begin{table}[ht]" + "\n"
        r"    \centering" + "\n"
        r"    \caption{Ablation study on DARPA TC E3 THEIA.}" + "\n"
        r"    \label{tab:ablation}" + "\n"
        r"    \begin{tabular}{lcccc}" + "\n"
        r"        \toprule" + "\n"
        r"        Variant & Val AUPRC & Test AUPRC & Val F1 & Test F1 \\" + "\n"
        r"        \midrule" + "\n"
        + "\n".join(rows) + "\n"
        r"        \bottomrule" + "\n"
        r"    \end{tabular}" + "\n"
        r"\end{table}" + "\n"
    )

    save_path = os.path.join(out_dir, "ablation_table.tex")
    with open(save_path, "w") as f:
        f.write(latex)
    print(f"[saved] {save_path}")


def save_metrics_json(all_metrics: dict[str, dict], out_dir: str) -> None:
    """Save all metrics to a JSON file."""
    save_path = os.path.join(out_dir, "ablation_metrics.json")
    with open(save_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"[saved] {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2: Ablation Study with Training Curves",
    )

    # Explicit history paths
    parser.add_argument(
        "--full_history", type=str, default=None,
        help="Path to history.pt for the full (baseline) variant.",
    )
    parser.add_argument(
        "--no_cross_entity_history", type=str, default=None,
        help="Path to history.pt for the no-cross-entity variant.",
    )
    parser.add_argument(
        "--no_state_history", type=str, default=None,
        help="Path to history.pt for the no-state variant.",
    )

    # Auto-discover mode
    parser.add_argument(
        "--auto_discover", type=str, default=None,
        help="Base directory to auto-discover ablation run subdirectories.",
    )

    parser.add_argument(
        "--out_dir", type=str, default="results/experiment2",
        help="Output directory for plots, tables, and metrics.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve history paths -------------------------------------------------
    history_paths: dict[str, str] = {}

    if args.auto_discover:
        print(f"[auto-discover] Searching {args.auto_discover} ...")
        history_paths = auto_discover_histories(args.auto_discover)
    else:
        if args.full_history:
            history_paths["full"] = args.full_history
        if args.no_cross_entity_history:
            history_paths["no_cross_entity"] = args.no_cross_entity_history
        if args.no_state_history:
            history_paths["no_state"] = args.no_state_history

    missing = [k for k in VARIANT_ORDER if k not in history_paths]
    if missing:
        print(
            f"[error] Missing history paths for variant(s): {', '.join(missing)}.\n"
            f"Provide them explicitly or use --auto_discover.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load histories --------------------------------------------------------
    print("[loading] Loading history files ...")
    histories: dict[str, dict] = {}
    for variant_key in VARIANT_ORDER:
        path = history_paths[variant_key]
        print(f"  {variant_key}: {path}")
        histories[variant_key] = load_history(path)

    # Extract metrics -------------------------------------------------------
    all_metrics: dict[str, dict] = {}
    for variant_key in VARIANT_ORDER:
        all_metrics[variant_key] = _extract_final_metrics(histories[variant_key])

    # Create output directory -----------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    # Print comparison table ------------------------------------------------
    print_metrics_table(all_metrics)

    # Generate plots --------------------------------------------------------
    print("[plotting] Generating training curve plots ...")
    plot_training_curves(histories, args.out_dir)
    plot_full_curves(histories, args.out_dir)

    # Generate LaTeX table --------------------------------------------------
    print("[latex] Generating LaTeX table ...")
    generate_latex_table(all_metrics, args.out_dir)

    # Save metrics JSON -----------------------------------------------------
    print("[json] Saving metrics JSON ...")
    save_metrics_json(all_metrics, args.out_dir)

    print("\n[done] Experiment 2 ablation analysis complete.")


if __name__ == "__main__":
    main()
