# HyperMamba-NIDS Pipeline

## Quick Start

```bash
# Full pipeline (Theia)
make pipeline-theia

# Full pipeline (TRACE)  
make pipeline-trace

# Or run from a specific stage
python -m src.pipeline.run --dataset theia --start-from features
```

## Pipeline Stages

```
download → ingest → relabel → features → normalize → build_graph → build_sequences
```

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| **download** | `src.data.download_darpa_tc` | DARPA TC archives | `data/raw/{dataset}/*.json` |
| **ingest** | `src.pipeline.batch_ingest` | Raw JSON | `data/processed/{dataset}/shards/` + `labeled/` |
| **relabel** | `src.pipeline.relabel` | `labeled/*.parquet` | Updated labels (narrow, IoC, crossprocess) |
| **features** | `src.pipeline.batch_features` | `labeled/*.parquet` | `features/*.npz` + `feature_names.txt` |
| **normalize** | `src.pipeline.normalize` | `features/*.npz` | `features_norm/*.npz` + `norm_stats.npz` |
| **graph** | `src.pipeline.build_graph` | `labeled/*.parquet` | `graph/incidence.npz` + `entity_vocab.json` |
| **sequences** | `src.pipeline.build_sequences` | `graph/` + `labeled/` | `graph/subject_sequences.npz` |

## Data Splits

### Theia (25 shards, from 4 archives: 1r, 3, 5m, 6r)

**IMPORTANT**: Shard IDs are NOT in chronological order! The parser numbered
them by archive filename order, not by timestamp.

- **Train**: shards [0–8] — April 3–5 (42M events)
- **Validation**: shards [9, 10] — April 5 late (3.2M events)
- **Test**: shards [11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 24, 14, 15, 16] — April 9–13 (60.5M events, **chronological order!**)

Note: April 6–8 have no data in the DARPA TC dataset.

### TRACE (7 shards)
- **Train**: shards 0–4
- **Validation**: shard 5
- **Test**: shard 6

Splits are defined during normalization (`--train-shards`) and enforced during training via `SHARD_CONFIG`.

## Label Types

| Label | Description | Theia Attack % | TRACE Attack % |
|-------|-------------|---------------|---------------|
| `label_broad` | Entire process tree of entry processes | ~24% | ~71% |
| `label_narrow` | Only during known attack time windows | ~0% | varies |
| `label_ioc` | IoC-matched objects (IPs, files) | ~0% | varies |
| `label_crossprocess` | Child processes only (excl. entry) | ~1% | ~0.1% |

## Training

```bash
# THyN (hypergraph + Mamba)
torchrun --nproc_per_node=4 -m src.pipeline.train --config configs/thyn_v0.yaml

# Baseline A (pairwise + Mamba)
torchrun --nproc_per_node=4 -m src.pipeline.train --config configs/baseline_a.yaml
```

## Evaluation

```bash
# Control experiments (temporal, entity-ID, label shuffle)
python -m src.pipeline.control_experiment \
    --checkpoint ckpts/thyn_v0/best.pt \
    --config configs/thyn_v0.yaml
```

## Output Structure

```
data/processed/darpa_tc_e3/{dataset}/
├── shards/
│   ├── events_shard{i}.parquet
│   ├── subjects_shard{i}.parquet
│   └── objects_shard{i}.parquet
├── labeled/
│   └── labeled_shard{i}.parquet
├── features/
│   ├── thyne_shard{i}.npz
│   └── feature_names.txt
├── features_norm/
│   ├── thyne_shard{i}.npz
│   ├── norm_stats.npz
│   └── feature_names.txt
├── graph/
│   ├── incidence.npz
│   ├── entity_vocab.npz
│   ├── shard_offsets.npz
│   ├── hyperedge_labels.npz
│   ├── hyperedge_metadata.npz
│   └── subject_sequences.npz
├── subjects.parquet
└── objects.parquet
```
