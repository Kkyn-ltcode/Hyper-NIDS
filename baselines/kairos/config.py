"""
KAIROS baseline configuration for HyperMamba-NIDS comparison.

Adapted from:
  https://github.com/ProvenanceAnalytics/kairos/blob/main/DARPA/CADETS_E3/config.py

Key differences from original KAIROS:
  - No PostgreSQL dependency (we read from Parquet)
  - Uses our shard-based train/test split instead of day-based
  - Evaluation uses event-level AUPRC for THyN comparison
"""

from pathlib import Path

# ============================================================
# Paths
# ============================================================

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "darpa_tc_e3"

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifact"
GRAPHS_DIR = ARTIFACT_DIR / "graphs"
MODELS_DIR = ARTIFACT_DIR / "models"

# ============================================================
# Dataset-specific configurations
# ============================================================

DATASET_CONFIGS = {
    "theia": {
        "data_dir": DATA_ROOT / "theia",
        "train_shards": list(range(0, 7)),   # shards 0-6
        "val_shards": [7],                    # shard 7
        "test_shards": [8, 9],                # shards 8-9
        "include_edge_type": [
            "EVENT_WRITE",
            "EVENT_READ",
            "EVENT_OPEN",
            "EVENT_EXECUTE",
            "EVENT_SENDMSG",
            "EVENT_RECVMSG",
            "EVENT_RECVFROM",
        ],
        "edge_reversed": [
            "EVENT_RECVFROM",
            "EVENT_RECVMSG",
        ],
    },
    "trace": {
        "data_dir": DATA_ROOT / "trace",
        "train_shards": list(range(0, 5)),
        "val_shards": [5],
        "test_shards": [6],
        "include_edge_type": [
            "EVENT_WRITE",
            "EVENT_READ",
            "EVENT_OPEN",
            "EVENT_EXECUTE",
            "EVENT_SENDMSG",
            "EVENT_RECVMSG",
            "EVENT_RECVFROM",
        ],
        "edge_reversed": [
            "EVENT_RECVFROM",
            "EVENT_RECVMSG",
        ],
    },
}

# ============================================================
# Edge type → ID mapping (auto-generated from include_edge_type)
# ============================================================

def build_rel2id(include_edge_type):
    """Build bidirectional edge type ↔ integer mapping."""
    rel2id = {}
    for idx, etype in enumerate(include_edge_type, start=1):
        rel2id[idx] = etype
        rel2id[etype] = idx
    return rel2id

# ============================================================
# Model hyperparameters (KAIROS defaults)
# ============================================================

NODE_EMBEDDING_DIM = 16
NODE_STATE_DIM = 100
NEIGHBOR_SIZE = 20
EDGE_DIM = 100
TIME_DIM = 100

# ============================================================
# Training hyperparameters
# ============================================================

BATCH_SIZE = 1024
LR = 5e-5
EPS = 1e-8
WEIGHT_DECAY = 0.01
EPOCH_NUM = 50

# Time window for anomaly detection (15 minutes in nanoseconds)
TIME_WINDOW_SIZE = 60_000_000_000 * 15
