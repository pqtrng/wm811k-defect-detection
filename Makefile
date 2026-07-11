.PHONY: help install notebook mlflow-server train evaluate gradcam validate

UV ?= uv
MLFLOW_HOST ?= 0.0.0.0
MLFLOW_PORT ?= 5000
MLFLOW_BACKEND_STORE_URI ?= sqlite:///$(CURDIR)/mlflow.db
MLFLOW_ARTIFACT_ROOT ?= $(CURDIR)/mlruns
CONFIG ?= configs/default.yaml
MODEL ?= resnet18
CHECKPOINT ?= models/resnet18-aug_best.pt
ARGS ?=

help:
	@printf "Available targets:\n"
	@printf "  make install         Install dependencies with uv\n"
	@printf "  make notebook        Start Jupyter Lab\n"
	@printf "  make mlflow-server   Start the MLflow tracking server\n"
	@printf "  make train           Train a model (MODEL=resnet18 ARGS='--augment')\n"
	@printf "  make evaluate        Evaluate a checkpoint (CHECKPOINT=models/...pt)\n"
	@printf "  make gradcam         Generate Grad-CAM grids from a checkpoint\n"
	@printf "  make validate        Run data quality gates + die-preservation report\n"

install:
	$(UV) sync

notebook:
	$(UV) run jupyter lab

mlflow-server:
	$(UV) run mlflow server \
		--backend-store-uri "$(MLFLOW_BACKEND_STORE_URI)" \
		--default-artifact-root "$(MLFLOW_ARTIFACT_ROOT)" \
		--host "$(MLFLOW_HOST)" \
		--port "$(MLFLOW_PORT)"

train:
	$(UV) run python -m wm811k.train --config $(CONFIG) --model $(MODEL) $(ARGS)

evaluate:
	$(UV) run python -m wm811k.evaluate --config $(CONFIG) --model $(MODEL) --checkpoint $(CHECKPOINT) $(ARGS)

gradcam:
	$(UV) run python -m wm811k.gradcam --config $(CONFIG) --model $(MODEL) --checkpoint $(CHECKPOINT)

validate:
	$(UV) run python -m wm811k.validate --config $(CONFIG)