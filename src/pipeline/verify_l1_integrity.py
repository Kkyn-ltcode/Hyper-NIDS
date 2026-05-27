"""
L1* Full Stack Integrity Check

Verifies every layer of the pipeline to ensure L1* experiment correctness.
Run on the CUDA machine where data is available.

Usage:
    python -m src.pipeline.verify_l1_integrity --dataset theia
"""

import argparse
import gc
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

DATA_ROOT = Path("data/processed/darpa_tc_e3")

TRAIN_SHARDS = {"theia": list(range(7)), "trace": list(range(5))}
VAL_SHARDS   = {"theia": [7],           "trace": [5]}
TEST_SHARDS  = {"theia": [8, 9],        "trace": [6]}


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_pass(msg):
    print(f"  ✅ {msg}")


def check_fail(msg):
    print(f"  ❌ {msg}")


def check_warn(msg):
    print(f"  ⚠️  {msg}")


def check_info(msg):
    print(f"  ℹ️  {msg}")


# ============================================================
# CHECK 1: L1* Label Correctness
# ============================================================
def check_l1_labels(dataset, data_root):
    section("CHECK 1: L1* Label Correctness")

    labeled_dir = data_root / "labeled"
    train_shards = TRAIN_SHARDS[dataset]
    val_shards = VAL_SHARDS[dataset]
    test_shards = TEST_SHARDS[dataset]

    # 1a. Verify label_l1 column exists in all shards
    all_shards = sorted(train_shards + val_shards + test_shards)
    for sid in all_shards:
        path = labeled_dir / f"labeled_shard{sid}.parquet"
        cols = pd.read_parquet(path, columns=[]).columns.tolist()
        if "label_l1" not in cols:
            check_fail(f"Shard {sid}: 'label_l1' column MISSING. Available: {cols}")
            return
    check_pass("label_l1 column exists in all shards")

    # 1b. Val/Test: label_l1 MUST equal label_broad
    for split_name, shard_list in [("val", val_shards), ("test", test_shards)]:
        for sid in shard_list:
            df = pd.read_parquet(
                labeled_dir / f"labeled_shard{sid}.parquet",
                columns=["label_broad", "label_l1"])
            diff = (df["label_broad"] != df["label_l1"]).sum()
            if diff > 0:
                check_fail(f"Shard {sid} ({split_name}): label_l1 differs from label_broad "
                           f"in {diff:,} events! L1* should NOT modify val/test labels.")
                return
            else:
                check_pass(f"Shard {sid} ({split_name}): label_l1 == label_broad (all {len(df):,} events)")
            del df

    # 1c. Train: label_l1 should have FEWER attacks than label_broad
    total_broad_atk = 0
    total_l1_atk = 0
    total_events = 0
    neutralized_subjects = set()

    for sid in train_shards:
        df = pd.read_parquet(
            labeled_dir / f"labeled_shard{sid}.parquet",
            columns=["subject_uuid", "label_broad", "label_l1"])
        broad_atk = (df["label_broad"] == 1).sum()
        l1_atk = (df["label_l1"] == 1).sum()
        neutralized = (df["label_broad"] == 1) & (df["label_l1"] == 0)
        n_neutralized = neutralized.sum()

        total_broad_atk += broad_atk
        total_l1_atk += l1_atk
        total_events += len(df)

        # Track which subjects were neutralized
        if n_neutralized > 0:
            neut_subjs = df[neutralized]["subject_uuid"].unique()
            neutralized_subjects.update(neut_subjs)

        check_info(f"Shard {sid} (train): broad_atk={broad_atk:,}, l1_atk={l1_atk:,}, "
                   f"neutralized={n_neutralized:,}")
        del df; gc.collect()

    if total_l1_atk < total_broad_atk:
        check_pass(f"Train L1* has fewer attacks than broad: "
                   f"{total_l1_atk:,} vs {total_broad_atk:,} "
                   f"(neutralized {total_broad_atk - total_l1_atk:,})")
    else:
        check_fail(f"Train L1* should have fewer attacks! "
                   f"l1={total_l1_atk:,}, broad={total_broad_atk:,}")

    check_info(f"Neutralized subjects: {len(neutralized_subjects):,}")

    # 1d. Verify neutralized subjects are ALL from the entry process
    subj_df = pd.read_parquet(data_root / "subjects.parquet")
    neut_basenames = set()
    for uuid in neutralized_subjects:
        row = subj_df[subj_df["uuid"] == uuid]
        if not row.empty:
            path = row.iloc[0].get("process_path", "")
            if not path and "cmd_line" in row.columns:
                path = row.iloc[0]["cmd_line"].split()[0] if str(row.iloc[0]["cmd_line"]).strip() else ""
            bn = str(path).rstrip("/").rsplit("/", 1)[-1].lower() if path else "<NONE>"
            neut_basenames.add(bn)

    check_info(f"Neutralized basenames: {neut_basenames}")
    if len(neut_basenames) == 1:
        check_pass(f"All neutralized subjects come from ONE binary: {neut_basenames}")
    elif len(neut_basenames) == 0:
        check_fail("No neutralized subjects found!")
    else:
        check_warn(f"Neutralized subjects come from MULTIPLE binaries: {neut_basenames}")

    del subj_df; gc.collect()

    # 1e. Critical: Are any neutralized basenames present as ATTACK in test?
    test_atk_basenames = set()
    subj_df = pd.read_parquet(data_root / "subjects.parquet")
    uuid_to_bn = {}
    for _, row in subj_df.iterrows():
        path = row.get("process_path", "")
        if not path and "cmd_line" in row.keys():
            path = str(row["cmd_line"]).split()[0] if str(row["cmd_line"]).strip() else ""
        bn = str(path).rstrip("/").rsplit("/", 1)[-1].lower() if path else ""
        uuid_to_bn[row["uuid"]] = bn
    del subj_df; gc.collect()

    for sid in test_shards:
        df = pd.read_parquet(
            labeled_dir / f"labeled_shard{sid}.parquet",
            columns=["subject_uuid", "label_broad"])
        atk_uuids = df[df["label_broad"] == 1]["subject_uuid"].unique()
        for u in atk_uuids:
            bn = uuid_to_bn.get(u, "")
            if bn:
                test_atk_basenames.add(bn)
        del df

    overlap = neut_basenames & test_atk_basenames
    if overlap:
        check_pass(f"Neutralized binary '{overlap}' IS present as attack in test. "
                   f"This is the L1* novel-binary scenario.")
    else:
        check_warn(f"Neutralized binary is NOT in test attacks. "
                   f"Test attacks: {test_atk_basenames}")

    check_info(f"Test attack basenames: {test_atk_basenames}")


# ============================================================
# CHECK 2: Feature Consistency (L0 vs L1*)
# ============================================================
def check_feature_consistency(dataset, data_root):
    section("CHECK 2: Feature Consistency (L0 vs L1*)")

    features_dir = data_root / "features_norm"

    # Features should be IDENTICAL between L0 and L1* (only labels change)
    # Verify by loading the same shard's features
    test_shard = TRAIN_SHARDS[dataset][0]
    d = np.load(features_dir / f"thyne_shard{test_shard}.npz")
    X = d["X"]
    y_broad = d["y_broad"]

    check_info(f"Shard {test_shard}: X.shape={X.shape}, y_broad.shape={y_broad.shape}")

    # Check for NaN/Inf
    if np.isnan(X).any():
        check_fail(f"Features contain NaN values!")
    else:
        check_pass("No NaN values in features")

    if np.isinf(X).any():
        check_fail(f"Features contain Inf values!")
    else:
        check_pass("No Inf values in features")

    # Verify train stats were computed from train only
    scaler_path = features_dir / "scaler_params.npz"
    if scaler_path.exists():
        scaler = np.load(scaler_path)
        if "train_shards" in scaler:
            train_shards_used = scaler["train_shards"]
            check_info(f"Scaler trained on shards: {train_shards_used}")
        if "mean" in scaler:
            check_info(f"Scaler mean range: [{scaler['mean'].min():.4f}, {scaler['mean'].max():.4f}]")
        check_pass("Scaler params file exists")
    else:
        check_warn("No scaler_params.npz found — verify normalization manually")

    # Verify test shard features are NOT zero-mean (should be shifted by train stats)
    test_sid = TEST_SHARDS[dataset][0]
    d_test = np.load(features_dir / f"thyne_shard{test_sid}.npz")
    X_test = d_test["X"]
    # Continuous features start after event type columns
    feat_names_path = data_root / "features" / "feature_names.txt"
    if not feat_names_path.exists():
        feat_names_path = features_dir / "feature_names.txt"
    all_names = feat_names_path.read_text().strip().split("\n")
    n_etype = sum(1 for n in all_names if n.startswith("etype_"))
    cont_features = X_test[:, n_etype:]

    test_means = cont_features.mean(axis=0)
    near_zero = np.abs(test_means) < 0.01
    n_near_zero = near_zero.sum()
    check_info(f"Test shard {test_sid}: {n_near_zero}/{len(test_means)} "
               f"continuous features have mean near 0")
    if n_near_zero == len(test_means):
        check_warn("ALL test features have mean ~0. Possible normalization leak "
                   "(test stats used instead of train stats)")
    else:
        check_pass("Test features are NOT zero-centered (normalized with train stats)")


# ============================================================
# CHECK 3: DataLoader Label Loading
# ============================================================
def check_dataloader_labels(dataset, data_root):
    section("CHECK 3: DataLoader Label Loading")

    from src.data.thyn_dataset import THyNDataset

    # Load train set with L1* labels
    print("  Loading train dataset with label_type='l1'...")
    train_l1 = THyNDataset(
        TRAIN_SHARDS[dataset], data_root,
        max_seq_len=512, label_type="l1", verbose=False)

    # Load train set with broad labels
    print("  Loading train dataset with label_type='broad'...")
    train_broad = THyNDataset(
        TRAIN_SHARDS[dataset], data_root,
        max_seq_len=512, label_type="broad", verbose=False)

    # 3a. Features must be identical
    if np.array_equal(train_l1.X_cont, train_broad.X_cont):
        check_pass("Train features are IDENTICAL between L0 and L1*")
    else:
        check_fail("Train features DIFFER between L0 and L1*! This should NOT happen.")

    # 3b. Event types must be identical
    if np.array_equal(train_l1.event_type, train_broad.event_type):
        check_pass("Train event types are IDENTICAL between L0 and L1*")
    else:
        check_fail("Train event types DIFFER between L0 and L1*!")

    # 3c. Labels must differ (L1* has fewer attacks)
    l1_atk = (train_l1.y == 1).sum()
    broad_atk = (train_broad.y == 1).sum()
    if l1_atk < broad_atk:
        check_pass(f"L1* train has fewer attacks: {l1_atk:,} vs {broad_atk:,}")
    else:
        check_fail(f"L1* should have fewer attacks! l1={l1_atk:,}, broad={broad_atk:,}")

    # 3d. Entity IDs must be identical
    if np.array_equal(train_l1.entity_ids, train_broad.entity_ids):
        check_pass("Train entity IDs are IDENTICAL between L0 and L1*")
    else:
        check_fail("Train entity IDs DIFFER between L0 and L1*!")

    # 3e. Windows must be identical (same subject sequences)
    if train_l1.windows == train_broad.windows:
        check_pass("Train windows are IDENTICAL between L0 and L1*")
    else:
        check_fail("Train windows DIFFER between L0 and L1*!")

    del train_l1, train_broad; gc.collect()

    # 3f. Val set: L1* config should still load BROAD labels for evaluation
    # This is handled in train.py, but verify the dataset itself
    print("  Loading val dataset with label_type='broad'...")
    val_broad = THyNDataset(
        VAL_SHARDS[dataset], data_root,
        max_seq_len=512, label_type="broad", verbose=False)
    val_atk = (val_broad.y == 1).sum()
    val_total = len(val_broad.y)
    check_info(f"Val (broad): {val_atk:,} attack / {val_total:,} total "
               f"({100*val_atk/val_total:.2f}%)")

    del val_broad; gc.collect()


# ============================================================
# CHECK 4: Temporal Shuffle Correctness
# ============================================================
def check_shuffle_correctness(dataset, data_root):
    section("CHECK 4: Temporal Shuffle Correctness")

    from src.data.thyn_dataset import THyNDataset
    from src.pipeline.audit_temporal import reorder_batch

    ds = THyNDataset(
        VAL_SHARDS[dataset], data_root,
        max_seq_len=512, label_type="broad", verbose=False)
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    # Get a sequence with enough events
    for i in range(batch["mask"].shape[0]):
        real_len = int(batch["mask"][i].sum().item())
        if real_len >= 10:
            break

    normal_et = batch["event_type"][i, :real_len].clone()
    normal_xc = batch["X_cont"][i, :real_len, 0].clone()  # first feature

    # Shuffle
    import copy
    batch_shuf = {k: v.clone() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    batch_shuf = reorder_batch(batch_shuf, order="shuffle")
    shuf_et = batch_shuf["event_type"][i, :real_len]
    shuf_xc = batch_shuf["X_cont"][i, :real_len, 0]

    # Reverse
    batch_rev = {k: v.clone() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    batch_rev = reorder_batch(batch_rev, order="reverse")
    rev_et = batch_rev["event_type"][i, :real_len]

    # Verify shuffle changed order
    if torch.equal(normal_et, shuf_et):
        check_warn("Shuffled event types are IDENTICAL to normal (could be coincidence, "
                   "but very unlikely for len>=10)")
    else:
        check_pass(f"Shuffle changes event order (seq_len={real_len})")

    # Verify reverse is actually reversed
    expected_rev = normal_et.flip(0)
    if torch.equal(rev_et, expected_rev):
        check_pass("Reverse produces exact reversal of event types")
    else:
        check_fail("Reverse does NOT produce exact reversal!")
        print(f"    Normal:   {normal_et[:10].tolist()}")
        print(f"    Reversed: {rev_et[:10].tolist()}")
        print(f"    Expected: {expected_rev[:10].tolist()}")

    # Verify features are also shuffled (not just event types)
    if torch.equal(normal_xc, shuf_xc):
        check_warn("Shuffled features IDENTICAL to normal (possible bug)")
    else:
        check_pass("Shuffle also reorders continuous features")

    # Verify shuffle preserves the same set of events (bag is identical)
    normal_sorted, _ = normal_et.sort()
    shuf_sorted, _ = shuf_et.sort()
    if torch.equal(normal_sorted, shuf_sorted):
        check_pass("Shuffle preserves bag-of-events (same multiset)")
    else:
        check_fail("Shuffle changed the BAG of events! This is a bug.")

    # Verify labels are also reordered consistently
    normal_y = batch["y"][i, :real_len]
    shuf_y = batch_shuf["y"][i, :real_len]
    n_y_sorted, _ = normal_y.sort()
    s_y_sorted, _ = shuf_y.sort()
    if torch.equal(n_y_sorted, s_y_sorted):
        check_pass("Shuffle preserves label distribution")
    else:
        check_fail("Shuffle changed label distribution! Bug in reorder_batch.")

    del ds; gc.collect()


# ============================================================
# CHECK 5: Train/Val/Test Subject Leakage
# ============================================================
def check_subject_leakage(dataset, data_root):
    section("CHECK 5: Train/Val/Test Subject Leakage")

    labeled_dir = data_root / "labeled"

    train_subjects = set()
    val_subjects = set()
    test_subjects = set()

    for sid in TRAIN_SHARDS[dataset]:
        df = pd.read_parquet(
            labeled_dir / f"labeled_shard{sid}.parquet",
            columns=["subject_uuid"])
        train_subjects.update(df["subject_uuid"].unique())
        del df

    for sid in VAL_SHARDS[dataset]:
        df = pd.read_parquet(
            labeled_dir / f"labeled_shard{sid}.parquet",
            columns=["subject_uuid"])
        val_subjects.update(df["subject_uuid"].unique())
        del df

    for sid in TEST_SHARDS[dataset]:
        df = pd.read_parquet(
            labeled_dir / f"labeled_shard{sid}.parquet",
            columns=["subject_uuid"])
        test_subjects.update(df["subject_uuid"].unique())
        del df

    train_val_overlap = train_subjects & val_subjects
    train_test_overlap = train_subjects & test_subjects
    val_test_overlap = val_subjects & test_subjects

    check_info(f"Unique subjects: train={len(train_subjects):,}, "
               f"val={len(val_subjects):,}, test={len(test_subjects):,}")

    if train_val_overlap:
        check_warn(f"Train/Val overlap: {len(train_val_overlap):,} subjects "
                   f"(expected in DARPA TC — same processes span shards)")
    else:
        check_pass("No train/val subject overlap")

    if train_test_overlap:
        check_warn(f"Train/Test overlap: {len(train_test_overlap):,} subjects")
    else:
        check_pass("No train/test subject overlap")

    # More importantly: are ATTACK subjects shared?
    train_atk_subjects = set()
    test_atk_subjects = set()

    for sid in TRAIN_SHARDS[dataset]:
        df = pd.read_parquet(
            labeled_dir / f"labeled_shard{sid}.parquet",
            columns=["subject_uuid", "label_broad"])
        atk = df[df["label_broad"] == 1]["subject_uuid"].unique()
        train_atk_subjects.update(atk)
        del df

    for sid in TEST_SHARDS[dataset]:
        df = pd.read_parquet(
            labeled_dir / f"labeled_shard{sid}.parquet",
            columns=["subject_uuid", "label_broad"])
        atk = df[df["label_broad"] == 1]["subject_uuid"].unique()
        test_atk_subjects.update(atk)
        del df

    atk_overlap = train_atk_subjects & test_atk_subjects
    check_info(f"Train attack subjects: {len(train_atk_subjects):,}")
    check_info(f"Test attack subjects: {len(test_atk_subjects):,}")
    if atk_overlap:
        check_warn(f"Attack subject UUID overlap between train and test: "
                   f"{len(atk_overlap):,} subjects. This means the SAME process "
                   f"instance spans both splits (long-running attack process).")
    else:
        check_pass("No attack subject UUID overlap between train and test")


# ============================================================
# CHECK 6: Checkpoint Sanity
# ============================================================
def check_checkpoint_sanity(dataset, data_root):
    section("CHECK 6: Checkpoint Sanity")

    import yaml
    from src.data.thyn_dataset import THyNDataset
    from src.model.thyn import THyN
    from src.pipeline.train import compute_metrics

    ckpt_dir = Path("checkpoints")

    for config_name in [f"{dataset}_thyn", f"{dataset}_baseline_a",
                        f"{dataset}_l1_thyn", f"{dataset}_l1_baseline_a"]:
        ckpt_path = ckpt_dir / config_name / "best.pt"
        config_path = Path("configs") / f"{config_name}.yaml"

        if not ckpt_path.exists():
            check_warn(f"{config_name}: checkpoint not found at {ckpt_path}")
            continue
        if not config_path.exists():
            check_warn(f"{config_name}: config not found at {config_path}")
            continue

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        dcfg = cfg["data"]
        mcfg = cfg["model"]
        label_type = dcfg.get("label_type", "broad")

        # Load val set with BROAD labels (always for evaluation)
        val_ds = THyNDataset(
            dcfg["val_shards"], data_root,
            max_seq_len=dcfg["max_seq_len"],
            label_type="broad",
            verbose=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = THyN(
            n_cont_features=val_ds.n_cont_features,
            num_event_types=val_ds.num_event_types,
            d_type=mcfg.get("d_type", 16),
            d_model=mcfg["d_model"],
            d_hidden=mcfg["d_hidden"],
            n_layers=mcfg["n_layers"],
            dropout=mcfg["dropout"],
            model_type=mcfg["model_type"],
            encoder_type=mcfg["encoder_type"],
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Quick evaluation
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                X_c = batch["X_cont"].to(device).clamp(-20, 20)
                et = batch["event_type"].to(device)
                mask = batch["mask"].to(device)
                ent = batch["entity_ids"].to(device)
                y = batch["y"].to(device)

                logits = model(X_c, et, entity_ids=ent, mask=mask)
                real = mask.bool()
                all_logits.extend(logits[real].cpu().tolist())
                all_labels.extend(y[real].cpu().tolist())

        metrics = compute_metrics(all_logits, all_labels)
        check_info(f"{config_name}: AUPRC={metrics['auprc']:.4f}, "
                   f"F1={metrics['best_f1']:.4f} (train_labels={label_type})")

        # Verify checkpoint metadata if available
        if "epoch" in ckpt:
            check_info(f"  Best epoch: {ckpt['epoch']}")
        if "val_auprc" in ckpt:
            logged = ckpt["val_auprc"]
            computed = metrics["auprc"]
            if abs(logged - computed) < 0.01:
                check_pass(f"  Logged AUPRC ({logged:.4f}) matches recomputed ({computed:.4f})")
            else:
                check_fail(f"  Logged AUPRC ({logged:.4f}) != recomputed ({computed:.4f})! "
                           f"Possible evaluation bug.")

        del model, val_ds, val_loader; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="L1* Full Stack Integrity Check")
    parser.add_argument("--dataset", default="theia", choices=["theia", "trace"])
    parser.add_argument("--skip-ckpt", action="store_true",
                        help="Skip checkpoint verification (Layer 6)")
    args = parser.parse_args()

    data_root = DATA_ROOT / args.dataset

    print("=" * 60)
    print(f"  L1* FULL STACK INTEGRITY CHECK — {args.dataset.upper()}")
    print("=" * 60)

    check_l1_labels(args.dataset, data_root)
    check_feature_consistency(args.dataset, data_root)
    check_dataloader_labels(args.dataset, data_root)
    check_shuffle_correctness(args.dataset, data_root)
    check_subject_leakage(args.dataset, data_root)
    if not args.skip_ckpt:
        check_checkpoint_sanity(args.dataset, data_root)

    section("DONE")
    print("  Review all ✅/❌/⚠️ above to confirm pipeline integrity.\n")


if __name__ == "__main__":
    main()
