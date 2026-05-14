.PHONY: pipeline-theia pipeline-trace pipeline-trace-1 \
        ingest parse relabel features normalize graph sequences \
        train-thyn train-baseline control-experiment \
        kairos-convert kairos-train kairos-eval help

PYTHON ?= python
DATASET ?= theia
TRAIN_SHARDS ?= 0-6
GPUS ?= 4

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'

# ============================================================
# Full pipelines
# ============================================================

pipeline-theia:  ## Run full pipeline for Theia
	$(PYTHON) -m src.pipeline.run --dataset theia --train-shards 0-6

pipeline-trace:  ## Run full pipeline for TRACE
	$(PYTHON) -m src.pipeline.run --dataset trace --train-shards 0-4

pipeline-trace-1:  ## Run full pipeline for TRACE
	$(PYTHON) -m src.pipeline.run --dataset trace-1 --train-shards 0-4

# ============================================================
# Individual stages
# ============================================================

parse:  ## Parse JSON shards → Parquet
	$(PYTHON) -m src.data.darpa_tc_parser --shards all --dataset $(DATASET)

ingest:  ## Parse + label (DATASET=theia|trace)
	$(PYTHON) -m src.pipeline.batch_ingest --dataset $(DATASET)

relabel:  ## Recompute narrow/IoC/crossprocess labels
	$(PYTHON) -m src.pipeline.relabel --dataset $(DATASET)

features:  ## Extract per-event features
	$(PYTHON) -m src.pipeline.batch_features --dataset $(DATASET)

_TRAIN_SHARDS = $(if $(filter trace,$(DATASET)),0-4,$(TRAIN_SHARDS))
normalize:  ## Z-score normalization
	$(PYTHON) -m src.pipeline.normalize --dataset $(DATASET) --train-shards $(_TRAIN_SHARDS)

graph:  ## Build incidence matrix
	$(PYTHON) -m src.pipeline.build_graph --dataset $(DATASET)

sequences:  ## Build subject sequences
	$(PYTHON) -m src.pipeline.build_sequences --dataset $(DATASET)

# ============================================================
# Training
# ============================================================

train-thyn:  ## Train THyN (4 GPU)
	torchrun --nproc_per_node=$(GPUS) -m src.pipeline.train --config configs/thyn_v0.yaml

train-baseline:  ## Train Baseline A (4 GPU)
	torchrun --nproc_per_node=$(GPUS) -m src.pipeline.train --config configs/baseline_a.yaml

# ============================================================
# Evaluation
# ============================================================

control-experiment:  ## Run control experiment (temporal/entity/label shuffle)
	$(PYTHON) -m src.pipeline.control_experiment \
		--checkpoint checkpoints/thyn_v0/best.pt \
		--config configs/thyn_v0.yaml
	$(PYTHON) -m src.pipeline.control_experiment \
		--checkpoint checkpoints/baseline_a/best.pt \
		--config configs/baseline_a.yaml

# ============================================================
# KAIROS Baseline
# ============================================================

kairos-convert:  ## Convert Parquet → KAIROS TemporalData
	$(PYTHON) -m baselines.kairos.data_converter --dataset $(DATASET)

kairos-train:  ## Train KAIROS baseline
	$(PYTHON) -m baselines.kairos.train --dataset $(DATASET)

kairos-eval:  ## Evaluate KAIROS baseline
	$(PYTHON) -m baselines.kairos.evaluate --dataset $(DATASET) --split test

