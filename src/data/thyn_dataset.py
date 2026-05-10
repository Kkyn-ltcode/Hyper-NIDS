"""
THyN PyTorch Dataset.

Indexes by (subject_id, window_start) pairs. Each item returns a
fixed-length window of a subject's event sequence with features,
labels, entity IDs, and a padding mask.

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

    Each item is a fixed-length window from one subject's chronological
    event sequence. Long subjects produce multiple windows; short ones
    are padded.

    Returns dict with keys:
        X:          (max_seq_len, n_features) float32
        y:          (max_seq_len,) int64 — labels (-1 = padding)
        entity_ids: (max_seq_len, 3) int64 — [subj, obj, obj2] per event
        mask:       (max_seq_len,) float32 — 1=real, 0=pad
        subject_id: int
        seq_len:    int — actual (unpadded) length
    """

    def __init__(
        self,
        shard_ids: list[int],
        data_root: str | Path,
        max_seq_len: int = 512,
        stride: int | None = None,
        min_seq_len: int = 2,
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
        shard_ranges = []  # (global_start, global_end, local_offset)
        local_pos = 0
        for sid in shard_ids:
            gs, ge = shard_starts[sid], shard_ends[sid]
            shard_ranges.append((gs, ge, local_pos - gs))
            local_pos += (ge - gs)
        self._shard_ranges = shard_ranges
        self._valid_min = shard_starts[shard_ids[0]]
        self._valid_max = shard_ends[shard_ids[-1]]

        # --- Load features + labels ---
        print(f"  Loading features for shards {shard_ids}...")
        Xs, ys = [], []
        for sid in shard_ids:
            d = np.load(features_dir / f"thyne_shard{sid}.npz")
            Xs.append(d["X"])
            ys.append(d["y_broad"])

        # Align feature dimensions (event type one-hot may vary per shard)
        max_cols = max(x.shape[1] for x in Xs)
        for i in range(len(Xs)):
            if Xs[i].shape[1] < max_cols:
                pad_width = max_cols - Xs[i].shape[1]
                Xs[i] = np.pad(Xs[i], ((0, 0), (0, pad_width)),
                               constant_values=0.0)

        self.X = np.concatenate(Xs)
        self.y = np.concatenate(ys).astype(np.int64)
        del Xs, ys; gc.collect()
        self.n_features = self.X.shape[1]
        print(f"    Features: {self.X.shape}, labels: {self.y.shape}")

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
        self.windows = []  # (subject_id, w_start, w_end)

        for subj_id in range(num_entities):
            s, e = seq_offset[subj_id], seq_offset[subj_id + 1]
            if s == e:
                filtered_offsets.append(filtered_offsets[-1])
                continue

            subj_he = all_he[s:e]
            # Filter to events in this split's shards
            mask = (subj_he >= self._valid_min) & (subj_he < self._valid_max)
            valid_he = subj_he[mask]

            if len(valid_he) < min_seq_len:
                filtered_offsets.append(filtered_offsets[-1])
                continue

            # Convert global HE IDs to local feature indices
            local_idx = self._global_to_local_vec(valid_he)
            filtered_chunks.append(local_idx)

            n = len(local_idx)
            base = filtered_offsets[-1]
            filtered_offsets.append(base + n)

            # Create windows
            for w_start in range(0, n, stride):
                w_end = min(w_start + max_seq_len, n)
                if w_end - w_start < min_seq_len:
                    continue
                self.windows.append((subj_id, base + w_start, base + w_end))

        self.filtered_idx = np.concatenate(filtered_chunks) if filtered_chunks else np.array([], dtype=np.int64)
        self.filtered_offset = np.array(filtered_offsets, dtype=np.int64)
        del seq_data, all_he, seq_offset, filtered_chunks
        gc.collect()

        n_subjects = (np.diff(self.filtered_offset) > 0).sum()
        print(f"    Subjects with events: {n_subjects:,}")
        print(f"    Windows: {len(self.windows):,}")
        print(f"    Total events in split: {len(self.X):,}")

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

        X = self.X[local_indices]
        y = self.y[local_indices]
        ent = self.entity_ids[local_indices]

        # Pad
        pad = self.max_seq_len - seq_len
        if pad > 0:
            X = np.pad(X, ((0, pad), (0, 0)))
            y = np.pad(y, (0, pad), constant_values=-1)
            ent = np.pad(ent, ((0, pad), (0, 0)), constant_values=-1)

        mask = np.zeros(self.max_seq_len, dtype=np.float32)
        mask[:seq_len] = 1.0

        return {
            "X": torch.from_numpy(X.copy()).float(),
            "y": torch.from_numpy(y.copy()).long(),
            "entity_ids": torch.from_numpy(ent.copy()).long(),
            "mask": torch.from_numpy(mask),
            "subject_id": subj_id,
            "seq_len": seq_len,
        }
