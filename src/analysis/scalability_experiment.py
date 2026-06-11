"""
Experiment 4: Computational Scalability Profiling

Measures per-event throughput (events/second) and peak memory for HyperMamba
and GRU-TGN across varying chunk_size and d_model configurations.

Both models are O(N) in sequence length (sequential event processing, no
sequence-level attention). The comparison isolates the constant-factor
overhead of HyperMamba's AllSetAggregator (3-entity cross-attention per event)
vs GRU-TGN's simpler GRUCell update.

Results are anchored against the DARPA TC real-time ingestion rate (~10K
events/second) to ground the scalability claim in deployment viability.

Outputs:
    results/scalability/
    ├── profiling_results.csv          # Raw measurements
    ├── parameter_counts.csv           # Model parameter breakdown
    ├── throughput_vs_chunk_size.png   # Plot: events/s vs chunk_size
    ├── throughput_vs_d_model.png      # Plot: events/s vs d_model
    ├── memory_vs_d_model.png         # Plot: memory vs d_model
    └── summary.txt                    # Paper-ready summary table

Usage:
    python -m src.analysis.scalability_experiment
    python -m src.analysis.scalability_experiment --device cpu
    python -m src.analysis.scalability_experiment --mode inference  # forward only
"""

import argparse
import gc
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.model.hypermamba_full import HyperMambaFull
from src.baselines.gru_tgn import GRUTGNBaseline


# ---------------------------------------------------------------------------
# Real THEIA dataset dimensions (from shard 0)
# ---------------------------------------------------------------------------
REAL_N_CONT = 70
REAL_N_EVENT_TYPES = 19
REAL_N_ENTITIES = 2_736_362  # Full entity vocabulary

# For profiling, use a reduced entity count to avoid OOM on the state bank.
# The bank is (num_entities, d_model) — at d_model=512 and 2.7M entities,
# that's 5.2GB just for the bank buffer. We use 100K entities for profiling
# which is representative of per-event compute without the bank size dominating.
PROFILE_N_ENTITIES = 100_000

# DARPA TC ingestion rate (from KAIROS paper, Section 6.1)
DARPA_TC_INGESTION_RATE = 10_000  # events/second


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def make_synthetic_batch(chunk_size, n_cont, n_event_types, n_entities, device):
    """Generate one synthetic batch matching the real data interface.

    Returns tensors shaped (1, C, ...) as the DataLoader produces.
    """
    x_cont = torch.randn(1, chunk_size, n_cont, device=device)
    event_type = torch.randint(0, n_event_types, (1, chunk_size), device=device)
    # Entity IDs: random valid IDs, with ~10% invalid (slot 2 = obj2 often null)
    entity_ids = torch.randint(0, n_entities, (1, chunk_size, 3), device=device)
    # Make ~10% of obj2 slots invalid (-1)
    mask = torch.rand(1, chunk_size, device=device) < 0.1
    entity_ids[:, :, 2][mask] = -1
    timestamps = torch.arange(chunk_size, device=device, dtype=torch.float32).unsqueeze(0)

    return x_cont, event_type, entity_ids, timestamps


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(model_type, d_model, n_entities, device):
    """Instantiate a model with given d_model."""
    if model_type == "hypermamba":
        model = HyperMambaFull(
            num_entities=n_entities,
            n_cont_features=REAL_N_CONT,
            num_event_types=REAL_N_EVENT_TYPES,
            d_model=d_model,
        )
    elif model_type == "gru_tgn":
        model = GRUTGNBaseline(
            num_entities=n_entities,
            n_cont_features=REAL_N_CONT,
            num_event_types=REAL_N_EVENT_TYPES,
            d_model=d_model,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.to(device)


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Bank buffers (not parameters)
    buffers = sum(b.numel() for b in model.buffers())
    return {"total_params": total, "trainable_params": trainable, "buffer_elements": buffers}


def count_parameter_breakdown(model, model_type):
    """Detailed parameter breakdown by component."""
    breakdown = {}
    for name, param in model.named_parameters():
        # Group by top-level component
        component = name.split(".")[0]
        if component not in breakdown:
            breakdown[component] = 0
        breakdown[component] += param.numel()
    return breakdown


# ---------------------------------------------------------------------------
# Memory measurement
# ---------------------------------------------------------------------------

def measure_memory(model, batch, device, mode="inference"):
    """Measure peak memory for a single forward (+ optional backward) pass.

    Returns peak memory in MB.
    """
    x_cont, event_type, entity_ids, timestamps = batch

    # Force garbage collection
    gc.collect()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)

        if mode == "inference":
            with torch.no_grad():
                logits = model(x_cont, event_type, entity_ids, timestamps)
        else:
            logits = model(x_cont, event_type, entity_ids, timestamps)
            loss = logits.sum()
            loss.backward()

        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        model.detach_bank()
        return (peak - baseline) / (1024 * 1024)  # MB

    elif device.type == "mps":
        # MPS: use driver memory reporting
        torch.mps.synchronize()
        baseline = torch.mps.current_allocated_memory()

        if mode == "inference":
            with torch.no_grad():
                logits = model(x_cont, event_type, entity_ids, timestamps)
        else:
            logits = model(x_cont, event_type, entity_ids, timestamps)
            loss = logits.sum()
            loss.backward()

        torch.mps.synchronize()
        peak = torch.mps.current_allocated_memory()
        model.detach_bank()
        return max(0, (peak - baseline)) / (1024 * 1024)

    else:
        # CPU: estimate from model + activation size (no reliable peak API)
        import resource
        gc.collect()
        r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux, bytes on Mac

        if mode == "inference":
            with torch.no_grad():
                logits = model(x_cont, event_type, entity_ids, timestamps)
        else:
            logits = model(x_cont, event_type, entity_ids, timestamps)
            loss = logits.sum()
            loss.backward()

        gc.collect()
        r1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        model.detach_bank()

        # macOS reports in bytes, Linux in KB
        import platform
        if platform.system() == "Darwin":
            return max(0, (r1 - r0)) / (1024 * 1024)  # bytes → MB
        else:
            return max(0, (r1 - r0)) / 1024  # KB → MB


# ---------------------------------------------------------------------------
# Throughput measurement
# ---------------------------------------------------------------------------

def measure_throughput(model, chunk_size, n_entities, device, mode="inference",
                       warmup_iters=3, timed_iters=10):
    """Measure events/second for a given configuration.

    Args:
        mode: "inference" (forward only) or "training" (forward + backward + step)
    """
    model.eval() if mode == "inference" else model.train()
    model.reset_bank()

    batch = make_synthetic_batch(chunk_size, REAL_N_CONT, REAL_N_EVENT_TYPES,
                                 n_entities, device)

    optimizer = None
    if mode == "training":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Warmup: populate caches, JIT compile, etc.
    for _ in range(warmup_iters):
        if mode == "inference":
            with torch.no_grad():
                _ = model(*batch)
        else:
            optimizer.zero_grad()
            logits = model(*batch)
            loss = logits.sum()
            loss.backward()
            optimizer.step()
        model.detach_bank()

    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(timed_iters):
        if mode == "inference":
            with torch.no_grad():
                _ = model(*batch)
        else:
            optimizer.zero_grad()
            logits = model(*batch)
            loss = logits.sum()
            loss.backward()
            optimizer.step()
        model.detach_bank()

    # Synchronize after timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - t0
    total_events = chunk_size * timed_iters
    events_per_sec = total_events / elapsed
    ms_per_chunk = (elapsed / timed_iters) * 1000

    return {
        "events_per_sec": events_per_sec,
        "ms_per_chunk": ms_per_chunk,
        "elapsed_sec": elapsed,
        "timed_iters": timed_iters,
    }


# ---------------------------------------------------------------------------
# Main profiling sweep
# ---------------------------------------------------------------------------

def run_profiling(args):
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else
                           "mps" if torch.backends.mps.is_available() else "cpu"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Device: {device}")
    logging.info(f"Mode: {args.mode}")
    logging.info(f"Profile entities: {PROFILE_N_ENTITIES:,}")

    model_types = ["hypermamba", "gru_tgn"]
    chunk_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    d_models = [64, 128, 256, 512]

    # Fixed defaults for single-variable sweeps
    default_chunk = 4096
    default_d_model = 256

    all_results = []

    # -----------------------------------------------------------------------
    # Part 1: Parameter counts at each d_model
    # -----------------------------------------------------------------------
    logging.info("\n" + "=" * 60)
    logging.info("PARAMETER COUNTS")
    logging.info("=" * 60)

    param_rows = []
    for d in d_models:
        for mt in model_types:
            model = make_model(mt, d, PROFILE_N_ENTITIES, torch.device("cpu"))
            counts = count_parameters(model)
            breakdown = count_parameter_breakdown(model, mt)
            param_rows.append({
                "model": mt,
                "d_model": d,
                **counts,
                "breakdown": str(breakdown),
            })
            logging.info(f"  {mt:>12} d={d:>3}: {counts['trainable_params']:>10,} params, "
                         f"{counts['buffer_elements']:>12,} buffer elements")
            del model
            gc.collect()

    param_df = pd.DataFrame(param_rows)
    param_df.to_csv(out_dir / "parameter_counts.csv", index=False)

    # -----------------------------------------------------------------------
    # Part 2: Throughput vs chunk_size (d_model fixed at 256)
    # -----------------------------------------------------------------------
    logging.info("\n" + "=" * 60)
    logging.info(f"THROUGHPUT vs CHUNK SIZE (d_model={default_d_model})")
    logging.info("=" * 60)

    for mt in model_types:
        model = make_model(mt, default_d_model, PROFILE_N_ENTITIES, device)
        model.eval() if args.mode == "inference" else model.train()

        for cs in chunk_sizes:
            model.reset_bank()
            try:
                result = measure_throughput(model, cs, PROFILE_N_ENTITIES, device,
                                            mode=args.mode)
                # Memory measurement
                model.reset_bank()
                batch = make_synthetic_batch(cs, REAL_N_CONT, REAL_N_EVENT_TYPES,
                                             PROFILE_N_ENTITIES, device)
                mem_mb = measure_memory(model, batch, device, mode=args.mode)

                row = {
                    "model": mt,
                    "d_model": default_d_model,
                    "chunk_size": cs,
                    "sweep": "chunk_size",
                    "events_per_sec": result["events_per_sec"],
                    "ms_per_chunk": result["ms_per_chunk"],
                    "peak_memory_mb": mem_mb,
                }
                all_results.append(row)

                logging.info(f"  {mt:>12} chunk={cs:>5}: "
                             f"{result['events_per_sec']:>8,.0f} evt/s, "
                             f"{result['ms_per_chunk']:>7.1f} ms/chunk, "
                             f"{mem_mb:>6.1f} MB")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.warning(f"  {mt:>12} chunk={cs:>5}: OOM — skipped")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Part 3: Throughput vs d_model (chunk_size fixed at 4096)
    # -----------------------------------------------------------------------
    logging.info("\n" + "=" * 60)
    logging.info(f"THROUGHPUT vs D_MODEL (chunk_size={default_chunk})")
    logging.info("=" * 60)

    for mt in model_types:
        for d in d_models:
            try:
                model = make_model(mt, d, PROFILE_N_ENTITIES, device)
                model.eval() if args.mode == "inference" else model.train()
                model.reset_bank()

                result = measure_throughput(model, default_chunk, PROFILE_N_ENTITIES,
                                            device, mode=args.mode)

                model.reset_bank()
                batch = make_synthetic_batch(default_chunk, REAL_N_CONT, REAL_N_EVENT_TYPES,
                                             PROFILE_N_ENTITIES, device)
                mem_mb = measure_memory(model, batch, device, mode=args.mode)

                row = {
                    "model": mt,
                    "d_model": d,
                    "chunk_size": default_chunk,
                    "sweep": "d_model",
                    "events_per_sec": result["events_per_sec"],
                    "ms_per_chunk": result["ms_per_chunk"],
                    "peak_memory_mb": mem_mb,
                }
                all_results.append(row)

                logging.info(f"  {mt:>12} d={d:>3}: "
                             f"{result['events_per_sec']:>8,.0f} evt/s, "
                             f"{result['ms_per_chunk']:>7.1f} ms/chunk, "
                             f"{mem_mb:>6.1f} MB")

                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.warning(f"  {mt:>12} d={d:>3}: OOM — skipped")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    raise

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_dir / "profiling_results.csv", index=False)
    logging.info(f"\nResults saved to {out_dir / 'profiling_results.csv'}")

    # -----------------------------------------------------------------------
    # Generate plots and summary
    # -----------------------------------------------------------------------
    generate_plots(results_df, param_df, out_dir)
    generate_summary(results_df, param_df, out_dir, args.mode)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_plots(results_df, param_df, out_dir):
    """Generate all scalability comparison plots."""
    colors = {"hypermamba": "#2563eb", "gru_tgn": "#dc2626"}
    markers = {"hypermamba": "o", "gru_tgn": "s"}
    labels = {"hypermamba": "HyperMamba", "gru_tgn": "GRU-TGN"}

    # --- Plot 1: Throughput vs chunk_size ---
    chunk_data = results_df[results_df["sweep"] == "chunk_size"]
    if not chunk_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for mt in ["hypermamba", "gru_tgn"]:
            d = chunk_data[chunk_data["model"] == mt].sort_values("chunk_size")
            if not d.empty:
                ax.plot(d["chunk_size"], d["events_per_sec"],
                        marker=markers[mt], color=colors[mt], label=labels[mt],
                        linewidth=2, markersize=8)

        # DARPA TC reference line
        ax.axhline(y=DARPA_TC_INGESTION_RATE, color="gray", linestyle="--",
                    linewidth=1.5, alpha=0.7, label=f"DARPA TC rate ({DARPA_TC_INGESTION_RATE:,}/s)")

        ax.set_xlabel("Chunk Size (events per forward pass)", fontsize=12)
        ax.set_ylabel("Throughput (events/second)", fontsize=12)
        ax.set_title("Throughput vs Chunk Size (d_model=256)", fontsize=14)
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "throughput_vs_chunk_size.png", dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"  Plot: throughput_vs_chunk_size.png")

    # --- Plot 2: Throughput vs d_model ---
    dmodel_data = results_df[results_df["sweep"] == "d_model"]
    if not dmodel_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for mt in ["hypermamba", "gru_tgn"]:
            d = dmodel_data[dmodel_data["model"] == mt].sort_values("d_model")
            if not d.empty:
                ax.plot(d["d_model"], d["events_per_sec"],
                        marker=markers[mt], color=colors[mt], label=labels[mt],
                        linewidth=2, markersize=8)

        ax.axhline(y=DARPA_TC_INGESTION_RATE, color="gray", linestyle="--",
                    linewidth=1.5, alpha=0.7, label=f"DARPA TC rate ({DARPA_TC_INGESTION_RATE:,}/s)")

        ax.set_xlabel("Model Dimension (d_model)", fontsize=12)
        ax.set_ylabel("Throughput (events/second)", fontsize=12)
        ax.set_title("Throughput vs Model Capacity (chunk_size=4096)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "throughput_vs_d_model.png", dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"  Plot: throughput_vs_d_model.png")

    # --- Plot 3: Memory vs d_model ---
    if not dmodel_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for mt in ["hypermamba", "gru_tgn"]:
            d = dmodel_data[dmodel_data["model"] == mt].sort_values("d_model")
            if not d.empty:
                ax.plot(d["d_model"], d["peak_memory_mb"],
                        marker=markers[mt], color=colors[mt], label=labels[mt],
                        linewidth=2, markersize=8)

        ax.set_xlabel("Model Dimension (d_model)", fontsize=12)
        ax.set_ylabel("Peak Activation Memory (MB)", fontsize=12)
        ax.set_title("Memory vs Model Capacity (chunk_size=4096)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "memory_vs_d_model.png", dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"  Plot: memory_vs_d_model.png")

    # --- Plot 4: Parameter count comparison ---
    if not param_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        width = 0.35
        d_vals = sorted(param_df["d_model"].unique())
        x = np.arange(len(d_vals))

        for idx, mt in enumerate(["hypermamba", "gru_tgn"]):
            vals = []
            for d in d_vals:
                row = param_df[(param_df["model"] == mt) & (param_df["d_model"] == d)]
                vals.append(int(row["trainable_params"].iloc[0]) if len(row) > 0 else 0)
            offset = (idx - 0.5) * width
            bars = ax.bar(x + offset, [v / 1000 for v in vals], width,
                          label=labels[mt], color=colors[mt], alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v/1000:.0f}K", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("d_model", fontsize=12)
        ax.set_ylabel("Trainable Parameters (thousands)", fontsize=12)
        ax.set_title("Parameter Count Comparison", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(d_vals)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "parameter_counts.png", dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"  Plot: parameter_counts.png")


# ---------------------------------------------------------------------------
# Summary table (paper-ready)
# ---------------------------------------------------------------------------

def generate_summary(results_df, param_df, out_dir, mode):
    """Generate a paper-ready summary comparing the two models."""
    lines = []
    lines.append("=" * 72)
    lines.append("EXPERIMENT 4: COMPUTATIONAL SCALABILITY SUMMARY")
    lines.append(f"Mode: {mode}")
    lines.append("=" * 72)

    # Parameter comparison at d_model=256
    lines.append("\n--- Parameter Counts (d_model=256) ---")
    for mt in ["hypermamba", "gru_tgn"]:
        row = param_df[(param_df["model"] == mt) & (param_df["d_model"] == 256)]
        if len(row) > 0:
            r = row.iloc[0]
            lines.append(f"  {mt:>12}: {int(r['trainable_params']):>10,} trainable params")

    # Throughput at default config (chunk=4096, d=256)
    default = results_df[
        (results_df["chunk_size"] == 4096) & (results_df["d_model"] == 256)
    ]
    lines.append(f"\n--- Throughput at Default Config (chunk=4096, d_model=256) ---")
    throughputs = {}
    for mt in ["hypermamba", "gru_tgn"]:
        d = default[default["model"] == mt]
        if len(d) > 0:
            eps = d.iloc[0]["events_per_sec"]
            throughputs[mt] = eps
            lines.append(f"  {mt:>12}: {eps:>10,.0f} events/sec")

    if "hypermamba" in throughputs and "gru_tgn" in throughputs:
        overhead = (1 - throughputs["hypermamba"] / throughputs["gru_tgn"]) * 100
        lines.append(f"  {'overhead':>12}: {overhead:>+.1f}% (HyperMamba vs GRU-TGN)")
        lines.append(f"  {'DARPA TC':>12}: {DARPA_TC_INGESTION_RATE:>10,} events/sec (reference)")

        for mt, label in [("hypermamba", "HyperMamba"), ("gru_tgn", "GRU-TGN")]:
            ratio = throughputs[mt] / DARPA_TC_INGESTION_RATE
            lines.append(f"  {label:>12}: {ratio:.1f}x DARPA TC ingestion rate")

    # Throughput scaling summary
    lines.append(f"\n--- Throughput Scaling (d_model=256) ---")
    lines.append(f"  {'chunk':>8} | {'HyperMamba':>12} | {'GRU-TGN':>12} | {'Overhead':>10}")
    lines.append(f"  {'-'*8} | {'-'*12} | {'-'*12} | {'-'*10}")

    chunk_data = results_df[results_df["sweep"] == "chunk_size"]
    for cs in sorted(chunk_data["chunk_size"].unique()):
        hm = chunk_data[(chunk_data["model"] == "hypermamba") & (chunk_data["chunk_size"] == cs)]
        gt = chunk_data[(chunk_data["model"] == "gru_tgn") & (chunk_data["chunk_size"] == cs)]
        if len(hm) > 0 and len(gt) > 0:
            hm_eps = hm.iloc[0]["events_per_sec"]
            gt_eps = gt.iloc[0]["events_per_sec"]
            oh = (1 - hm_eps / gt_eps) * 100
            lines.append(f"  {cs:>8} | {hm_eps:>10,.0f}/s | {gt_eps:>10,.0f}/s | {oh:>+8.1f}%")

    # Memory scaling
    lines.append(f"\n--- Memory Scaling (chunk_size=4096) ---")
    lines.append(f"  {'d_model':>8} | {'HyperMamba':>12} | {'GRU-TGN':>12}")
    lines.append(f"  {'-'*8} | {'-'*12} | {'-'*12}")

    dmodel_data = results_df[results_df["sweep"] == "d_model"]
    for d in sorted(dmodel_data["d_model"].unique()):
        hm = dmodel_data[(dmodel_data["model"] == "hypermamba") & (dmodel_data["d_model"] == d)]
        gt = dmodel_data[(dmodel_data["model"] == "gru_tgn") & (dmodel_data["d_model"] == d)]
        if len(hm) > 0 and len(gt) > 0:
            lines.append(f"  {d:>8} | {hm.iloc[0]['peak_memory_mb']:>9.1f} MB | "
                         f"{gt.iloc[0]['peak_memory_mb']:>9.1f} MB")

    # Deployment viability
    lines.append(f"\n--- Deployment Viability ---")
    if "hypermamba" in throughputs:
        if throughputs["hypermamba"] >= DARPA_TC_INGESTION_RATE:
            lines.append(f"  ✓ HyperMamba sustains real-time processing at "
                         f"{throughputs['hypermamba']/DARPA_TC_INGESTION_RATE:.1f}x "
                         f"the DARPA TC ingestion rate")
        else:
            lines.append(f"  ✗ HyperMamba at {throughputs['hypermamba']:,.0f} evt/s is below "
                         f"the {DARPA_TC_INGESTION_RATE:,} evt/s DARPA TC rate")

    lines.append("=" * 72)

    summary_text = "\n".join(lines)
    logging.info("\n" + summary_text)

    with open(out_dir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(f"\nSummary saved to {out_dir / 'summary.txt'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: Computational Scalability Profiling")
    parser.add_argument("--mode", default="inference", choices=["inference", "training"],
                        help="Profile forward-only or forward+backward+step")
    parser.add_argument("--device", default=None,
                        help="Device (auto-detects cuda > mps > cpu)")
    parser.add_argument("--out_dir", default="results/scalability",
                        help="Output directory for results and plots")
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--timed_iters", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_profiling(args)


if __name__ == "__main__":
    main()
