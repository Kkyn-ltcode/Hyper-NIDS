import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ChronoDataset(Dataset):
    """
    Chronological dataset for global temporal processing.
    
    Returns dict with keys:
        X_cont:       (chunk_size, n_cont_features) float32
        event_type:   (chunk_size,) int64
        y:            (chunk_size,) int64
        entity_ids:   (chunk_size, 3) int64 — [subj, obj, obj2]
        mask:         (chunk_size,) float32 — 1=real
    """
    
    def __init__(
        self,
        shard_ids: list[int],
        data_root: str | Path,
        chunk_size: int = 1024,
        label_type: str = "broad",
        verbose: bool = True,
    ):
        data_root = Path(data_root)
        features_dir = data_root / "features_norm"
        labeled_dir = data_root / "labeled"
        graph_dir = data_root / "graph"
        
        self.chunk_size = chunk_size
        
        # Load entity vocabulary to get total entities
        vocab = np.load(graph_dir / "entity_vocab.npz", allow_pickle=True)
        uuid_to_id = {str(u): i for i, u in enumerate(vocab["uuids"])}
        self.num_entities = int(vocab["num_entities"])
        
        # --- Identify feature columns ---
        feat_names_path = data_root / "features" / "feature_names.txt"
        if not feat_names_path.exists():
            feat_names_path = features_dir / "feature_names.txt"
        all_feat_names = feat_names_path.read_text().strip().split("\n")
        etype_cols = [i for i, n in enumerate(all_feat_names) if n.startswith("etype_")]
        cont_cols = [i for i, n in enumerate(all_feat_names) if not n.startswith("etype_")]
        
        self.num_event_types = len(etype_cols) + 1  # +1 for unknown/padding
        self.n_cont_features = len(cont_cols)
        
        if verbose:
            print(f"  Feature split: {len(etype_cols)} event types, {len(cont_cols)} continuous")
            
        # --- Load features + labels ---
        label_col = f"label_{label_type}"
        if verbose:
            print(f"  Loading features for shards {shard_ids} (labels={label_type})...")
            
        Xs, ys = [], []
        for sid in shard_ids:
            d = np.load(features_dir / f"thyne_shard{sid}.npz")
            Xs.append(d["X"])
            
            if label_type == "broad":
                ys.append(d["y_broad"])
            else:
                ldf = pd.read_parquet(
                    labeled_dir / f"labeled_shard{sid}.parquet",
                    columns=[label_col])
                ys.append(ldf[label_col].values.astype(np.int64))
                del ldf
                
        # Align feature dimensions
        max_cols = max(x.shape[1] for x in Xs)
        for i in range(len(Xs)):
            if Xs[i].shape[1] < max_cols:
                pad_width = max_cols - Xs[i].shape[1]
                Xs[i] = np.pad(Xs[i], ((0, 0), (0, pad_width)), constant_values=0.0)
                
        X_all = np.concatenate(Xs)
        self.y = np.concatenate(ys).astype(np.int64)
        del Xs, ys; gc.collect()
        
        n_etype = len(etype_cols)
        etype_onehot = X_all[:, :n_etype]
        self.event_type = etype_onehot.argmax(axis=1).astype(np.int64) + 1
        no_etype = etype_onehot.sum(axis=1) == 0
        self.event_type[no_etype] = 0
        
        self.X_cont = X_all[:, n_etype:].astype(np.float32)
        del X_all, etype_onehot; gc.collect()
        
        # --- Load entity IDs and timestamps ---
        if verbose: print(f"  Loading entity IDs and timestamps...")
        ent_parts = []
        ts_parts = []
        for sid in shard_ids:
            df = pd.read_parquet(
                labeled_dir / f"labeled_shard{sid}.parquet",
                columns=["subject_uuid", "predicate_object_uuid", "predicate_object2_uuid", "timestamp_nanos"],
            )
            sub = df["subject_uuid"].map(uuid_to_id).fillna(-1).astype(np.int64).values
            obj = df["predicate_object_uuid"].map(uuid_to_id).fillna(-1).astype(np.int64).values
            obj2 = df["predicate_object2_uuid"].map(uuid_to_id).fillna(-1).astype(np.int64).values
            ent_parts.append(np.stack([sub, obj, obj2], axis=1))
            
            # Timestamp (nanoseconds). Convert to seconds as float64 for precision, then to float32
            ts_sec = (df["timestamp_nanos"].values.astype(np.float64) / 1e9).astype(np.float32)
            ts_parts.append(ts_sec)
            del df; gc.collect()
            
        self.entity_ids = np.concatenate(ent_parts)
        self.timestamps = np.concatenate(ts_parts)
        del ent_parts, ts_parts; gc.collect()
        
        self.total_events = len(self.X_cont)
        self.num_chunks = self.total_events // self.chunk_size
        
        if verbose:
            print(f"  Total events: {self.total_events:,}")
            print(f"  Total chunks (size {self.chunk_size}): {self.num_chunks:,}")
            
    def __len__(self):
        return self.num_chunks
        
    def __getitem__(self, idx):
        start = idx * self.chunk_size
        end = start + self.chunk_size
        
        X_c = self.X_cont[start:end]
        et = self.event_type[start:end]
        y = self.y[start:end]
        ent = self.entity_ids[start:end]
        ts = self.timestamps[start:end]
        mask = np.ones(self.chunk_size, dtype=np.float32)
        
        return {
            "X_cont": torch.from_numpy(X_c.copy()).float(),
            "event_type": torch.from_numpy(et.copy()).long(),
            "y": torch.from_numpy(y.copy()).long(),
            "entity_ids": torch.from_numpy(ent.copy()).long(),
            "timestamp": torch.from_numpy(ts.copy()).float(),
            "mask": torch.from_numpy(mask),
        }
