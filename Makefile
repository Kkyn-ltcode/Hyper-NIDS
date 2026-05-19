.PHONY: pipeline-theia pipeline-trace pipeline-trace-1 \
        ingest parse relabel features normalize graph sequences \
        train-thyn train-baseline control-experiment \
        l1-relabel train-thyn-l1 train-baseline-l1 \
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
	torchrun --nproc_per_node=$(GPUS) -m src.pipeline.train --config configs/theia_thyn_v0.yaml

train-baseline:  ## Train Baseline A (4 GPU)
	torchrun --nproc_per_node=$(GPUS) -m src.pipeline.train --config configs/theia_baseline_a.yaml

train-thyn-trace:  ## Train THyN on TRACE (4 GPU)
	torchrun --nproc_per_node=$(GPUS) -m src.pipeline.train --config configs/trace_thyn.yaml --dataset trace

train-baseline-trace:  ## Train Baseline A on TRACE (4 GPU)
	torchrun --nproc_per_node=$(GPUS) -m src.pipeline.train --config configs/trace_baseline_a.yaml --dataset trace

# ---- L1** Novel-Binary Experiment ----

l1-relabel:  ## Add L1** labels (neutralize Firefox in training)
	$(PYTHON) -m src.pipeline.novel_binary_relabel --dataset $(DATASET)

train-thyn-l1:  ## Train THyN on L1** labels
	torchrun --nproc_per_node=$(GPUS) -m src.pipeline.train --config configs/theia_l1_thyn.yaml

train-baseline-l1:  ## Train Baseline A on L1** labels
	torchrun --nproc_per_node=$(GPUS) -m src.pipeline.train --config configs/theia_l1_baseline_a.yaml

# ============================================================
# Evaluation
# ============================================================

control-experiment:  ## Run control experiment (temporal/entity/label shuffle)
	$(PYTHON) -m src.pipeline.control_experiment \
		--checkpoint checkpoints/thyn_v0/best.pt \
		--config configs/theia_thyn_v0.yaml
	$(PYTHON) -m src.pipeline.control_experiment \
		--checkpoint checkpoints/baseline_a/best.pt \
		--config configs/theia_baseline_a.yaml

# ============================================================
# KAIROS Baseline
# ============================================================

kairos-convert:  ## Convert Parquet → KAIROS TemporalData
	$(PYTHON) -m baselines.kairos.data_converter --dataset $(DATASET)

kairos-train:  ## Train KAIROS baseline
	$(PYTHON) -m baselines.kairos.train --dataset $(DATASET)

kairos-eval:  ## Evaluate KAIROS baseline (unsupervised)
	$(PYTHON) -m baselines.kairos.evaluate --dataset $(DATASET) --split val
	$(PYTHON) -m baselines.kairos.evaluate --dataset $(DATASET) --split test

kairos-extract:  ## Extract KAIROS embeddings for supervised head
	$(PYTHON) -m baselines.kairos.extract_embeddings --dataset $(DATASET)

kairos-supervised:  ## Train & evaluate supervised KAIROS head
	$(PYTHON) -m baselines.kairos.supervised_head --dataset $(DATASET)

kairos-full:  ## Full KAIROS pipeline (convert → train → eval → supervised)
	$(MAKE) kairos-convert DATASET=$(DATASET)
	$(MAKE) kairos-train DATASET=$(DATASET)
	$(MAKE) kairos-eval DATASET=$(DATASET)
	$(MAKE) kairos-extract DATASET=$(DATASET)
	$(MAKE) kairos-supervised DATASET=$(DATASET)
