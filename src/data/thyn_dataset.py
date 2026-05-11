"""
THyN PyTorch Dataset.

Indexes by (subject_id, window_start) pairs. Each item returns a
fixed-length window of a subject's event sequence with:
- Continuous features (hour, he_size, type_rarity, etc.)
- Event type index (for nn.Embedding)
- Entity IDs, labels, mask

Usage:
    from src.data.thyn_dataset import THyNDataset
    ds = THyNDataset([0,1,2,3,4,5,6], data_root, max_seq_len=512)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

EXCLUDED_UUIDS = {"00000000-0000-0000-0000-000000000000"}


class THyNDataset(Dataset):
    """
    Window-based dataset for temporal hypergraph classification.

    Returns dict with keys:
        X_cont:       (max_seq_len, n_cont_features) float32
        event_type:   (max_seq_len,) int64 — event type index (0-padded)
        y:            (max_seq_len,) int64 — labels (-1 = padding)
        entity_ids:   (max_seq_len, 3) int64 — [subj, obj, obj2]
        mask:         (max_seq_len,) float32 — 1=real, 0=pad
        subject_id:   int
        seq_len:      int
    """

    def __init__(
        self,
        shard_ids: list[int],
        data_root: str | Path,
        max_seq_len: int = 512,
        stride: int | None = None,
        min_seq_len: int = 2,
        label_type: str = "broad",  # "broad", "narrow", "ioc", "crossprocess"
    ):
        data_root = Path(data_root)
        features_dir = data_root / "features_norm"
        labeled_dir = data_root / "labeled"
        graph_dir = data_root / "graph"

        self.max_seq_len = max_seq_len
        stride = stride or max_seq_len

        # Load entity vocabulary
        vocab = np.load(graph_dir / "entity_vocab.npz", allow_pickle=True)
        uuid_to_id = {str(u): i for i, u in enumerate(vocab["uuids"])}
        num_entities = int(vocab["num_entities"])

        # Load shard offsets
        off = np.load(graph_dir / "shard_offsets.npz")
        shard_starts = {int(s): int(st) for s, st in zip(off["shard_idx"], off["start"])}
        shard_ends = {int(s): int(e) for s, e in zip(off["shard_idx"], off["end"])}

        # Build global→local mapping
        shard_ranges = []
        local_pos = 0
        for sid in shard_ids:
            gs, ge = shard_starts[sid], shard_ends[sid]
            shard_ranges.append((gs, ge, local_pos - gs))
            local_pos += (ge - gs)
        self._shard_ranges = shard_ranges
        self._valid_min = shard_starts[shard_ids[0]]
        self._valid_max = shard_ends[shard_ids[-1]]

        # --- Identify feature columns ---
        feat_names_path = data_root / "features" / "feature_names.txt"
        if not feat_names_path.exists():
            feat_names_path = features_dir / "feature_names.txt"
        all_feat_names = feat_names_path.read_text().strip().split("\n")
        etype_cols = [i for i, n in enumerate(all_feat_names)
                      if n.startswith("etype_")]
        cont_cols = [i for i, n in enumerate(all_feat_names)
                     if not n.startswith("etype_")]
        self.etype_names = [all_feat_names[i] for i in etype_cols]
        self.cont_names = [all_feat_names[i] for i in cont_cols]
        self.num_event_types = len(etype_cols) + 1  # +1 for unknown/padding
        self.n_cont_features = len(cont_cols)

        print(f"  Feature split: {len(etype_cols)} event types, "
              f"{len(cont_cols)} continuous")

        # --- Load features + labels ---
        self.label_type = label_type
        label_col = f"label_{label_type}"
        print(f"  Loading features for shards {shard_ids} "
              f"(labels={label_type})...")

        Xs, ys = [], []
        for sid in shard_ids:
            d = np.load(features_dir / f"thyne_shard{sid}.npz")
            Xs.append(d["X"])

            if label_type == "broad":
                ys.append(d["y_broad"])
            else:
                # Load narrow/ioc labels from labeled parquets
                ldf = pd.read_parquet(
                    labeled_dir / f"labeled_shard{sid}.parquet",
                    columns=[label_col])
                y_arr = ldf[label_col].values.astype(np.int64)
                if (y_arr == -1).all():
                    raise ValueError(
                        f"label_{label_type} is all -1 in shard {sid}. "
                        f"Run: python -m src.pipeline.relabel --dataset ...")
                ys.append(y_arr)
                del ldf

        # Align feature dimensions
        max_cols = max(x.shape[1] for x in Xs)
        for i in range(len(Xs)):
            if Xs[i].shape[1] < max_cols:
                pad_width = max_cols - Xs[i].shape[1]
                Xs[i] = np.pad(Xs[i], ((0, 0), (0, pad_width)),
                               constant_values=0.0)

        X_all = np.concatenate(Xs)
        self.y = np.concatenate(ys).astype(np.int64)
        n_atk = int((self.y == 1).sum())
        print(f"    Label distribution: {n_atk:,} attack / "
              f"{len(self.y) - n_atk:,} benign "
              f"({100*n_atk/len(self.y):.3f}%)")
        del Xs, ys; gc.collect()

        # Split into event type index + continuous features
        n_etype = len(etype_cols)
        etype_onehot = X_all[:, :n_etype]
        # argmax gives event type index; add 1 so 0 = padding token
        self.event_type = etype_onehot.argmax(axis=1).astype(np.int64) + 1
        # For rows where no etype is set (all zeros), assign 0 (unknown)
        no_etype = etype_onehot.sum(axis=1) == 0
        self.event_type[no_etype] = 0

        self.X_cont = X_all[:, n_etype:].astype(np.float32)
        del X_all, etype_onehot; gc.collect()

        # Also keep raw feature count for backward compat
        self.n_features = self.n_cont_features
        print(f"    Continuous: {self.X_cont.shape}, "
              f"Event types: {self.num_event_types}, "
              f"Labels: {self.y.shape}")

        # --- Load entity IDs ---
        print(f"  Loading entity IDs...")
        ent_parts = []
        for sid in shard_ids:
            df = pd.read_parquet(
                labeled_dir / f"labeled_shard{sid}.parquet",
                columns=["subject_uuid", "predicate_object_uuid",
                          "predicate_object2_uuid"],
            )
            sub = df["subject_uuid"].map(uuid_to_id).fillna(-1).astype(np.int64).values
            obj = df["predicate_object_uuid"].map(uuid_to_id).fillna(-1).astype(np.int64).values
            obj2 = df["predicate_object2_uuid"].map(uuid_to_id).fillna(-1).astype(np.int64).values
            ent_parts.append(np.stack([sub, obj, obj2], axis=1))
            del df; gc.collect()
        self.entity_ids = np.concatenate(ent_parts)
        del ent_parts; gc.collect()
        print(f"    Entity IDs: {self.entity_ids.shape}")

        # --- Build per-subject filtered sequences + windows ---
        print(f"  Building subject windows (max_seq_len={max_seq_len})...")
        seq_data = np.load(graph_dir / "subject_sequences.npz")
        all_he = seq_data["he_ids"]
        seq_offset = seq_data["offset"]

        filtered_chunks = []
        filtered_offsets = [0]
        self.windows = []

        for subj_id in range(num_entities):
            s, e = seq_offset[subj_id], seq_offset[subj_id + 1]
            if s == e:
                filtered_offsets.append(filtered_offsets[-1])
                continue

            subj_he = all_he[s:e]
            mask = (subj_he >= self._valid_min) & (subj_he < self._valid_max)
            valid_he = subj_he[mask]

            if len(valid_he) < min_seq_len:
                filtered_offsets.append(filtered_offsets[-1])
                continue

            local_idx = self._global_to_local_vec(valid_he)
            filtered_chunks.append(local_idx)

            n = len(local_idx)
            base = filtered_offsets[-1]
            filtered_offsets.append(base + n)

            for w_start in range(0, n, stride):
                w_end = min(w_start + max_seq_len, n)
                if w_end - w_start < min_seq_len:
                    continue
                self.windows.append((subj_id, base + w_start, base + w_end))

        self.filtered_idx = (np.concatenate(filtered_chunks)
                             if filtered_chunks
                             else np.array([], dtype=np.int64))
        self.filtered_offset = np.array(filtered_offsets, dtype=np.int64)
        del seq_data, all_he, seq_offset, filtered_chunks
        gc.collect()

        n_subjects = (np.diff(self.filtered_offset) > 0).sum()
        print(f"    Subjects with events: {n_subjects:,}")
        print(f"    Windows: {len(self.windows):,}")
        print(f"    Total events in split: {len(self.X_cont):,}")

    def _global_to_local_vec(self, he_ids: np.ndarray) -> np.ndarray:
        result = np.full_like(he_ids, -1)
        for gs, ge, off in self._shard_ranges:
            mask = (he_ids >= gs) & (he_ids < ge)
            result[mask] = he_ids[mask] + off
        return result

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        subj_id, w_start, w_end = self.windows[idx]
        local_indices = self.filtered_idx[w_start:w_end]
        seq_len = len(local_indices)

        X_cont = self.X_cont[local_indices]
        etype = self.event_type[local_indices]
        y = self.y[local_indices]
        ent = self.entity_ids[local_indices]

        # Pad
        pad = self.max_seq_len - seq_len
        if pad > 0:
            X_cont = np.pad(X_cont, ((0, pad), (0, 0)))
            etype = np.pad(etype, (0, pad), constant_values=0)
            y = np.pad(y, (0, pad), constant_values=-1)
            ent = np.pad(ent, ((0, pad), (0, 0)), constant_values=-1)

        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:seq_len] = 1.0

        return {
            "X_cont": torch.from_numpy(X_cont.copy()).float(),
            "event_type": torch.from_numpy(etype.copy()).long(),
            "y": torch.from_numpy(y.copy()).long(),
            "entity_ids": torch.from_numpy(ent.copy()).long(),
            "mask": torch.from_numpy(mask),
            "subject_id": subj_id,
            "seq_len": seq_len,
        }
