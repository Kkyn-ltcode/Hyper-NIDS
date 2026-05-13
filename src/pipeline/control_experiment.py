"""
Subject-Randomization Control Experiment.

Evaluates whether the model learned genuine temporal patterns or
just memorized subject identity. Two controls:

1. shuffle_temporal: Randomly permute events within each window
   (breaks sequential ordering, keeps features/labels)
2. shuffle_labels: Randomly reassign labels across windows
   (breaks feature→label correlation)

If AUPRC drops significantly under shuffle_temporal, Mamba's
sequential processing adds value. If it stays high, the model
is using per-event features only.

Usage:
    python -m src.pipeline.control_experiment \
        --checkpoint checkpoints/thyn_v0/best.pt \
        --config configs/thyn_v0.yaml
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

from src.data.thyn_dataset import THyNDataset
from src.model.thyn import THyN
from src.pipeline.train import compute_metrics, masked_bce_loss


DATA_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "processed" / "darpa_tc_e3"
)


class ShuffledDataset(Dataset):
    """Wraps THyNDataset with optional temporal/label/entity shuffling."""

    def __init__(self, base_ds, mode="temporal", seed=42):
        """
        Args:
            base_ds: THyNDataset instance
            mode: "temporal"   — shuffle event order within windows
                  "labels"     — randomly reassign labels across windows
                  "entity_ids" — shuffle entity IDs within windows
        """
        self.base = base_ds
        self.mode = mode
        self.rng = np.random.RandomState(seed)

        if mode == "labels":
            self.label_perm = self.rng.permutation(len(base_ds))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        if self.mode == "labels":
            item = self.base[idx]
            donor = self.base[self.label_perm[idx]]
            item["y"] = donor["y"]
            return item

        elif self.mode == "temporal":
            item = self.base[idx]
            seq_len = item["seq_len"]
            if seq_len > 1:
                perm = torch.randperm(seq_len)
                item["X_cont"][:seq_len] = item["X_cont"][perm]
                item["event_type"][:seq_len] = item["event_type"][perm]
                item["entity_ids"][:seq_len] = item["entity_ids"][perm]
                item["y"][:seq_len] = item["y"][perm]
            return item

        elif self.mode == "entity_ids":
            item = self.base[idx]
            seq_len = item["seq_len"]
            if seq_len > 1:
                ent = item["entity_ids"][:seq_len]  # (L, 3)
                # Collect all unique entity IDs in this window
                unique_ids = torch.unique(ent[ent > 0])
                if len(unique_ids) > 1:
                    # Create random permutation mapping
                    perm = unique_ids[torch.randperm(len(unique_ids))]
                    id_map = {old.item(): new.item()
                              for old, new in zip(unique_ids, perm)}
                    # Apply mapping to entity_ids
                    for i in range(seq_len):
                        for j in range(3):
                            v = ent[i, j].item()
                            if v > 0 and v in id_map:
                                ent[i, j] = id_map[v]
                    item["entity_ids"][:seq_len] = ent
            return item

        return self.base[idx]


@torch.no_grad()
def evaluate_control(model, loader, pw_t, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits, all_labels = [], []

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        X_c = batch["X_cont"].to(device)
        et = batch["event_type"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        ent = batch["entity_ids"].to(device)

        logits = model(X_c, et, entity_ids=ent, mask=mask)
        loss = masked_bce_loss(logits, y, mask, pw_t)

        total_loss += loss.item()
        n_batches += 1
        real = mask.bool()
        all_logits.extend(logits[real].cpu().tolist())
        all_labels.extend(y[real].cpu().tolist())

    metrics = compute_metrics(all_logits, all_labels)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def run_experiment(name, ckpt_path, cfg, dataset, device):
    print(f"\n{'='*60}")
    print(f"CONTROL: {name}")
    print(f"{'='*60}")

    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]

    data_root = DATA_ROOT / dataset

    # Load checkpoint — prefer its saved config over YAML
    ckpt = torch.load(ckpt_path, map_location=device,
                      weights_only=False)
    state = ckpt["model_state"]

    # Use checkpoint's config if available (handles GRU/Mamba mismatch)
    ckpt_cfg = ckpt.get("config", {}).get("model", {})

    # Infer architecture from checkpoint
    proj_weight = state["input_proj.0.weight"]
    d_type = ckpt_cfg.get("d_type", mcfg.get("d_type", 16))
    n_cont = proj_weight.shape[1] - d_type
    num_etypes = state["type_emb.weight"].shape[0]
    encoder_type = ckpt_cfg.get("encoder_type", mcfg["encoder_type"])
    model_type = ckpt_cfg.get("model_type", mcfg["model_type"])

    print(f"  Checkpoint encoder: {encoder_type}")
    print(f"  Checkpoint model:   {model_type}")

    model = THyN(
        n_cont_features=n_cont,
        num_event_types=num_etypes,
        d_type=d_type,
        d_model=ckpt_cfg.get("d_model", mcfg["d_model"]),
        d_hidden=ckpt_cfg.get("d_hidden", mcfg["d_hidden"]),
        n_layers=ckpt_cfg.get("n_layers", mcfg["n_layers"]),
        dropout=0.0,
        model_type=model_type,
        encoder_type=encoder_type,
    ).to(device)
    model.load_state_dict(state)

    pw_t = torch.tensor([tcfg.get("pos_weight", 3.0)], device=device)

    # Build val dataset
    val_ds = THyNDataset(
        dcfg["val_shards"], data_root,
        max_seq_len=dcfg["max_seq_len"],
        label_type=dcfg.get("label_type", "broad"),
    )

    results = {}

    # 1. Normal (baseline)
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    m = evaluate_control(model, loader, pw_t, device)
    results["normal"] = m
    print(f"  Normal:    AUPRC={m['auprc']:.4f} AUROC={m['auroc']:.4f} "
          f"F1={m['f1']:.4f}")

    # 2. Temporal shuffle (3 seeds)
    for seed in [42, 123, 456]:
        ds = ShuffledDataset(val_ds, mode="temporal", seed=seed)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        m = evaluate_control(model, loader, pw_t, device)
        results[f"temporal_s{seed}"] = m
        print(f"  Temporal (seed={seed}): AUPRC={m['auprc']:.4f} "
              f"AUROC={m['auroc']:.4f} F1={m['f1']:.4f}")

    # 3. Entity-ID shuffle (3 seeds)
    for seed in [42, 123, 456]:
        ds = ShuffledDataset(val_ds, mode="entity_ids", seed=seed)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        m = evaluate_control(model, loader, pw_t, device)
        results[f"entity_s{seed}"] = m
        print(f"  Entity (seed={seed}):   AUPRC={m['auprc']:.4f} "
              f"AUROC={m['auroc']:.4f} F1={m['f1']:.4f}")

    # 4. Label shuffle (3 seeds)
    for seed in [42, 123, 456]:
        ds = ShuffledDataset(val_ds, mode="labels", seed=seed)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        m = evaluate_control(model, loader, pw_t, device)
        results[f"labels_s{seed}"] = m
        print(f"  Labels (seed={seed}):   AUPRC={m['auprc']:.4f} "
              f"AUROC={m['auroc']:.4f} F1={m['f1']:.4f}")

    # Summary
    normal_auprc = results["normal"]["auprc"]
    temporal_auprcs = [results[f"temporal_s{s}"]["auprc"]
                       for s in [42, 123, 456]]
    entity_auprcs = [results[f"entity_s{s}"]["auprc"]
                     for s in [42, 123, 456]]
    label_auprcs = [results[f"labels_s{s}"]["auprc"]
                    for s in [42, 123, 456]]

    print(f"\n  ── Summary ──")
    print(f"  Normal AUPRC:         {normal_auprc:.4f}")
    print(f"  Temporal shuffle:     {np.mean(temporal_auprcs):.4f} "
          f"± {np.std(temporal_auprcs):.4f} "
          f"(Δ = {normal_auprc - np.mean(temporal_auprcs):+.4f})")
    print(f"  Entity-ID shuffle:    {np.mean(entity_auprcs):.4f} "
          f"± {np.std(entity_auprcs):.4f} "
          f"(Δ = {normal_auprc - np.mean(entity_auprcs):+.4f})")
    print(f"  Label shuffle:        {np.mean(label_auprcs):.4f} "
          f"± {np.std(label_auprcs):.4f} "
          f"(Δ = {normal_auprc - np.mean(label_auprcs):+.4f})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="theia")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    results = run_experiment(
        cfg.get("name", "model"), args.checkpoint, cfg, args.dataset, device)

    # Save
    save_path = Path(args.checkpoint).parent / "control_results.pt"
    torch.save(results, save_path)
    print(f"\n  Saved to {save_path}")


if __name__ == "__main__":
    main()
