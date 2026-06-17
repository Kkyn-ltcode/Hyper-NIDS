"""
Node Aggregation Module.

Bridges the gap between HyperMamba's integer entity IDs and DARPA TC UUIDs,
and aggregates event-level logits to node-level anomaly scores.
Also handles loading PIDSMaker's native ground truth for fair comparison.
"""

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_entity_vocab(data_root: str | Path) -> tuple[dict[int, str], dict[str, int]]:
    """
    Load HyperMamba's entity_vocab.npz to map between integer IDs and UUIDs.
    
    Returns:
        id_to_uuid: {integer_id: "uuid_string"}
        uuid_to_id: {"uuid_string": integer_id}
    """
    data_root = Path(data_root)
    vocab_path = data_root / "graph" / "entity_vocab.npz"
    
    if not vocab_path.exists():
        raise FileNotFoundError(f"entity_vocab.npz not found at {vocab_path}")
    
    logger.info(f"Loading entity vocabulary from {vocab_path}")
    vocab = np.load(vocab_path, allow_pickle=True)
    uuids = vocab["uuids"]
    
    # integer ID is just the array index
    id_to_uuid = {i: str(u) for i, u in enumerate(uuids)}
    uuid_to_id = {str(u): i for i, u in enumerate(uuids)}
    
    logger.info(f"  Loaded {len(id_to_uuid):,} entities")
    return id_to_uuid, uuid_to_id


def aggregate_to_nodes(
    event_scores: np.ndarray,
    entity_ids: np.ndarray,
    id_to_uuid: dict[int, str],
    method: str = "max",
) -> dict[str, float]:
    """
    Aggregate per-event scores to per-node scores using integer entity IDs.
    
    Args:
        event_scores: (N,) float array of anomaly scores.
        entity_ids:   (N, 3) int array of [subject_id, object_id, object2_id].
        id_to_uuid:   Mapping from integer ID to UUID string.
        method:       Aggregation method ('max', 'mean', 'sum', 'p99').
        
    Returns:
        {uuid_string: aggregated_score}
    """
    assert event_scores.shape[0] == entity_ids.shape[0], \
        f"Score/ID length mismatch: {event_scores.shape[0]} vs {entity_ids.shape[0]}"
    
    from collections import defaultdict
    node_scores: dict[str, list[float]] = defaultdict(list)
    
    n_unmapped = 0
    for i in range(len(event_scores)):
        score = float(event_scores[i])
        for col_idx in range(entity_ids.shape[1]):
            eid = int(entity_ids[i, col_idx])
            if eid >= 0:  # -1 is padding/null
                uuid = id_to_uuid.get(eid)
                if uuid:
                    node_scores[uuid].append(score)
                else:
                    n_unmapped += 1
    
    if n_unmapped > 0:
        logger.warning(f"  {n_unmapped:,} entity IDs had no UUID mapping")
    
    result = {}
    for uuid, scores in node_scores.items():
        arr = np.array(scores)
        if method == "max":
            result[uuid] = float(arr.max())
        elif method == "mean":
            result[uuid] = float(arr.mean())
        elif method == "sum":
            result[uuid] = float(arr.sum())
        elif method == "p99":
            result[uuid] = float(np.percentile(arr, 99))
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
            
    logger.info(f"  Aggregated {len(event_scores):,} events to {len(result):,} nodes (method={method})")
    return result


def load_pidsmaker_gt(gt_dir: str | Path) -> dict[str, int]:
    """
    Load PIDSMaker's native ground truth format (no header, CSV).
    
    Format: UUID, {description_dict}, pidsmaker_index_id
    
    Args:
        gt_dir: Directory containing GT CSVs (e.g., Ground_Truth/orthrus/E3-THEIA/)
        
    Returns:
        {uuid_string: 1} for all malicious nodes.
    """
    gt_dir = Path(gt_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"PIDSMaker GT directory not found: {gt_dir}")
        
    csv_files = glob.glob(str(gt_dir / "*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {gt_dir}")
        return {}
        
    malicious_uuids = set()
    for fpath in csv_files:
        try:
            with open(fpath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # First column is the UUID
                    uuid = line.split(",", 1)[0].strip()
                    malicious_uuids.add(uuid)
        except Exception as e:
            logger.error(f"Error reading {fpath}: {e}")
            
    logger.info(f"Loaded PIDSMaker GT: {len(malicious_uuids)} malicious nodes from {len(csv_files)} files")
    return {u: 1 for u in malicious_uuids}
