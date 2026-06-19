"""
PIDSMaker Output Parser for Experiment 1 Comparison.

Reads PIDSMaker edge-level CSV outputs and converts them into the
standardized {uuid: anomaly_score} dict consumed by our metrics.py.

PIDSMaker CSV format (edge-level):
    loss,srcnode,dstnode,time,edge_type
    
Where srcnode/dstnode are PIDSMaker integer `index_id` values that must
be mapped back to UUIDs via the PostgreSQL node tables (or their cached
pickle export).

File path pattern:
    {task_path}/edge_losses/{split}/model_epoch_{epoch}/{time_interval}.csv

Usage:
    from src.baselines.parse_pidsmaker import load_pidsmaker_scores
    
    node_scores = load_pidsmaker_scores(
        edge_losses_dir="/path/to/edge_losses/test/model_epoch_0",
        nid_to_uuid_path="/path/to/nid_to_uuid.pkl",
        aggregation="max",
    )
"""

import glob
import logging
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================
# ID Mapping Loaders
# ============================================================

def load_nid_to_uuid_from_pickle(pkl_path: str | Path) -> dict[int, str]:
    """
    Load PIDSMaker's integer index_id → UUID mapping from a pickle file.
    
    PIDSMaker can export a mapping via its PostgreSQL database queries:
        nid2uuid = {index_id: node_uuid}
    
    If you've exported this from the database, load it here.
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"NID→UUID mapping not found at {pkl_path}. "
            f"Export it from PIDSMaker's PostgreSQL database using:\n"
            f"  from pidsmaker.utils.labelling import get_uuid2nids\n"
            f"  uuid2nids, nid2uuid = get_uuid2nids(cursor)\n"
            f"  pickle.dump(nid2uuid, open('nid_to_uuid.pkl', 'wb'))"
        )
    
    with open(pkl_path, "rb") as f:
        mapping = pickle.load(f)
    
    logger.info(f"Loaded NID→UUID mapping: {len(mapping):,} entries from {pkl_path}")
    return mapping


def load_nid_to_uuid_from_csv(csv_path: str | Path) -> dict[int, str]:
    """
    Load NID → UUID mapping from a CSV file with columns: index_id, node_uuid.
    
    Alternative to pickle loading — useful if exported via SQL query directly.
    """
    df = pd.read_csv(csv_path)
    mapping = dict(zip(df["index_id"].astype(int), df["node_uuid"].astype(str)))
    logger.info(f"Loaded NID→UUID mapping: {len(mapping):,} entries from {csv_path}")
    return mapping


# ============================================================
# Edge Loss CSV Loading
# ============================================================

def load_edge_losses_from_dir(
    edge_losses_dir: str | Path,
) -> pd.DataFrame:
    """
    Load all edge-level loss CSVs from a PIDSMaker output directory.
    
    Concatenates all time-window CSVs into a single DataFrame.
    
    Args:
        edge_losses_dir: Path to e.g. edge_losses/test/model_epoch_0/
    
    Returns:
        DataFrame with columns: loss, srcnode, dstnode, time, edge_type
    """
    edge_losses_dir = Path(edge_losses_dir)
    
    if not edge_losses_dir.exists():
        raise FileNotFoundError(f"Edge losses directory not found: {edge_losses_dir}")
    
    csv_files = sorted(glob.glob(str(edge_losses_dir / "*.csv")))
    
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {edge_losses_dir}. "
            f"Expected PIDSMaker edge loss CSVs with format: "
            f"loss,srcnode,dstnode,time,edge_type"
        )
    
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Validate expected columns
            expected_cols = {"loss", "srcnode", "dstnode"}
            if not expected_cols.issubset(set(df.columns)):
                logger.warning(
                    f"Skipping {csv_file}: missing columns {expected_cols - set(df.columns)}")
                continue
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error reading {csv_file}: {e}")
            continue
    
    if not dfs:
        raise ValueError(f"No valid edge loss CSVs loaded from {edge_losses_dir}")
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined):,} edges from {len(csv_files)} CSV files")
    
    return combined


# ============================================================
# Score Aggregation (Edge → Node)
# ============================================================

def aggregate_edge_losses_to_node_scores(
    edge_df: pd.DataFrame,
    nid_to_uuid: dict[int, str],
    aggregation: str = "max",
    use_dst_node: bool = True,
) -> dict[str, float]:
    """
    Aggregate per-edge anomaly scores to per-node (UUID) scores.
    
    This replicates PIDSMaker's internal node_evaluation.py logic:
    - For each edge, assign the loss to the source node
    - Optionally also to the destination node
    - Aggregate all losses per node via max/mean/p90
    
    Args:
        edge_df:       DataFrame with columns: loss, srcnode, dstnode
        nid_to_uuid:   Mapping from PIDSMaker integer ID → UUID string
        aggregation:   'max', 'mean', 'sum', 'p90', 'p99'
        use_dst_node:  If True, also assign each edge's loss to dstnode.
    
    Returns:
        {uuid: aggregated_anomaly_score}
    """
    node_losses: dict[str, list[float]] = defaultdict(list)
    
    unmapped_src = 0
    unmapped_dst = 0
    
    for _, row in edge_df.iterrows():
        loss_val = float(row["loss"])
        src_nid = int(row["srcnode"])
        dst_nid = int(row["dstnode"])
        
        # Map src
        src_uuid = nid_to_uuid.get(src_nid)
        if src_uuid:
            node_losses[src_uuid].append(loss_val)
        else:
            unmapped_src += 1
        
        # Map dst
        if use_dst_node:
            dst_uuid = nid_to_uuid.get(dst_nid)
            if dst_uuid:
                node_losses[dst_uuid].append(loss_val)
            else:
                unmapped_dst += 1
    
    if unmapped_src > 0:
        logger.warning(f"  {unmapped_src:,} edges had unmapped srcnode IDs")
    if use_dst_node and unmapped_dst > 0:
        logger.warning(f"  {unmapped_dst:,} edges had unmapped dstnode IDs")
    
    # Aggregate
    result = {}
    for uuid, losses in node_losses.items():
        arr = np.array(losses)
        if aggregation == "max":
            result[uuid] = float(arr.max())
        elif aggregation == "mean":
            result[uuid] = float(arr.mean())
        elif aggregation == "sum":
            result[uuid] = float(arr.sum())
        elif aggregation == "p90":
            result[uuid] = float(np.percentile(arr, 90))
        elif aggregation == "p99":
            result[uuid] = float(np.percentile(arr, 99))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    logger.info(f"  Aggregated to {len(result):,} unique nodes "
                f"(method={aggregation})")
    
    return result


def aggregate_edge_losses_vectorized(
    edge_df: pd.DataFrame,
    nid_to_uuid: dict[int, str],
    aggregation: str = "max",
    use_dst_node: bool = True,
) -> dict[str, float]:
    """
    Vectorized version of aggregate_edge_losses_to_node_scores.
    Much faster for large datasets (millions of edges).
    """
    # Map src node IDs to UUIDs
    src_uuids = edge_df["srcnode"].map(nid_to_uuid)
    
    # Build a long-form DataFrame: (uuid, loss)
    records = [pd.DataFrame({"uuid": src_uuids, "loss": edge_df["loss"]})]
    
    if use_dst_node:
        dst_uuids = edge_df["dstnode"].map(nid_to_uuid)
        records.append(pd.DataFrame({"uuid": dst_uuids, "loss": edge_df["loss"]}))
    
    long_df = pd.concat(records, ignore_index=True)
    # Drop unmapped entries
    n_unmapped = long_df["uuid"].isna().sum()
    if n_unmapped > 0:
        logger.warning(f"  {n_unmapped:,} edge-node pairs had unmapped IDs")
    long_df = long_df.dropna(subset=["uuid"])
    
    # Aggregate
    if aggregation == "max":
        agg = long_df.groupby("uuid")["loss"].max()
    elif aggregation == "mean":
        agg = long_df.groupby("uuid")["loss"].mean()
    elif aggregation == "sum":
        agg = long_df.groupby("uuid")["loss"].sum()
    elif aggregation == "p90":
        agg = long_df.groupby("uuid")["loss"].quantile(0.90)
    elif aggregation == "p99":
        agg = long_df.groupby("uuid")["loss"].quantile(0.99)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    result = agg.to_dict()
    logger.info(f"  Aggregated to {len(result):,} unique nodes "
                f"(method={aggregation}, vectorized)")
    return result


# ============================================================
# High-Level API
# ============================================================

def load_pidsmaker_scores(
    edge_losses_dir: str | Path,
    nid_to_uuid: dict[int, str] | str | Path,
    aggregation: str = "max",
    use_dst_node: bool = True,
    vectorized: bool = True,
) -> dict[str, float]:
    """
    End-to-end loader: reads PIDSMaker CSV outputs → returns node-level scores.
    
    Args:
        edge_losses_dir:  Path to edge_losses/test/model_epoch_N/
        nid_to_uuid:      Either a pre-loaded dict or path to a pickle/CSV file.
        aggregation:      'max', 'mean', 'sum', 'p90', 'p99'
        use_dst_node:     If True, count edge losses toward destination nodes too.
        vectorized:       Use vectorized aggregation (faster for large datasets).
    
    Returns:
        {uuid_string: anomaly_score}
    """
    # Load NID→UUID mapping if path is given
    if isinstance(nid_to_uuid, (str, Path)):
        nid_path = Path(nid_to_uuid)
        if nid_path.suffix == ".pkl":
            nid_to_uuid = load_nid_to_uuid_from_pickle(nid_path)
        elif nid_path.suffix == ".csv":
            nid_to_uuid = load_nid_to_uuid_from_csv(nid_path)
        else:
            raise ValueError(f"Unsupported mapping file format: {nid_path.suffix}")
    
    # Load edge losses
    edge_df = load_edge_losses_from_dir(edge_losses_dir)
    
    # Aggregate to node scores
    if vectorized:
        return aggregate_edge_losses_vectorized(
            edge_df, nid_to_uuid, aggregation, use_dst_node)
    else:
        return aggregate_edge_losses_to_node_scores(
            edge_df, nid_to_uuid, aggregation, use_dst_node)


def load_pidsmaker_ground_truth(
    gt_csv_path: str | Path,
) -> dict[str, int]:
    """
    Load PIDSMaker's ground truth CSV → {uuid: label} dict.
    
    PIDSMaker GT format: node_uuid,node_labels[,extra]
    
    Args:
        gt_csv_path: Path to the ground truth CSV.
    
    Returns:
        {uuid: 1} for all malicious nodes (benign nodes are not listed).
    """
    df = pd.read_csv(gt_csv_path)
    
    # PIDSMaker GT CSVs typically have node_uuid and node_labels columns
    uuid_col = "node_uuid" if "node_uuid" in df.columns else df.columns[0]
    label_col = "node_labels" if "node_labels" in df.columns else df.columns[1]
    
    gt = {}
    for _, row in df.iterrows():
        gt[str(row[uuid_col])] = int(row[label_col])
    
    logger.info(f"Loaded PIDSMaker ground truth: {len(gt)} entries, "
                f"{sum(v == 1 for v in gt.values())} malicious")
    return gt


def match_pidsmaker_edges_to_our_labels(
    edge_df: pd.DataFrame,
    data_root: str | Path,
    test_shards: list[int],
    label_type: str = "crossprocess",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match PIDSMaker edge losses to our event labels by exact nanosecond timestamp.
    
    This allows us to evaluate PIDSMaker baselines at the event-level using
    our exact ground truth labels, ensuring a 100% fair apples-to-apples
    comparison on the exact same events.
    
    Args:
        edge_df:     DataFrame from load_edge_losses_from_dir()
        data_root:   Path to processed dataset (e.g., data/processed/darpa_tc_e3/theia)
        test_shards: List of test shard IDs (e.g., [8, 9])
        label_type:  Which of our labels to use (default: 'crossprocess')
        
    Returns:
        (matched_scores, matched_labels) — Parallel 1D numpy arrays.
    """
    data_root = Path(data_root)
    labeled_dir = data_root / "labeled"
    label_col = f"label_{label_type}"
    
    # 1. Load all test events from our parquets
    logger.info(f"Loading our test events from shards {test_shards}...")
    our_events_dfs = []
    for sid in test_shards:
        parquet_path = labeled_dir / f"labeled_shard{sid}.parquet"
        df = pd.read_parquet(parquet_path, columns=["timestamp_nanos", label_col])
        our_events_dfs.append(df)
        
    our_df = pd.concat(our_events_dfs, ignore_index=True)
    # Some timestamps might have multiple events (e.g., hyperedges from the same log).
    # Group by timestamp and take the max label (if any event at this nanosecond is malicious, the timestamp is malicious)
    our_ts_labels = our_df.groupby("timestamp_nanos")[label_col].max().to_dict()
    
    logger.info(f"  Loaded {len(our_df):,} events, {len(our_ts_labels):,} unique nanosecond timestamps")
    
    # 2. Match PIDSMaker edges to our timestamps
    matched_scores = []
    matched_labels = []
    
    n_unmatched = 0
    n_edges = len(edge_df)
    
    # PIDSMaker 'time' column should be nanoseconds (int)
    for _, row in edge_df.iterrows():
        ts_nanos = int(row["time"])
        score = float(row["loss"])
        
        if ts_nanos in our_ts_labels:
            matched_scores.append(score)
            matched_labels.append(our_ts_labels[ts_nanos])
        else:
            n_unmatched += 1
            
    matched_scores_arr = np.array(matched_scores, dtype=np.float32)
    matched_labels_arr = np.array(matched_labels, dtype=np.int64)
    
    match_rate = len(matched_scores) / max(n_edges, 1) * 100
    logger.info(f"  Matched {len(matched_scores):,} / {n_edges:,} PIDSMaker edges ({match_rate:.1f}%)")
    
    if n_unmatched > 0:
        logger.warning(f"  {n_unmatched:,} PIDSMaker edges could not be matched to our test events by timestamp.")
        if match_rate < 10.0:
            logger.error("  Match rate is critically low! Check if PIDSMaker aggregated timestamps or used different test window.")
            
    return matched_scores_arr, matched_labels_arr
