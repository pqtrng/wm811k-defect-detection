.PHONY: help install notebook mlflow-server train evaluate gradcam validate serve test lint silver gold verify-gold

UV ?= uv
MLFLOW_HOST ?= 0.0.0.0
MLFLOW_PORT ?= 5000
SERVE_HOST ?= 0.0.0.0
SERVE_PORT ?= 8000
MLFLOW_BACKEND_STORE_URI ?= sqlite:///$(CURDIR)/mlflow.db
MLFLOW_ARTIFACT_ROOT ?= $(CURDIR)/mlruns
CONFIG ?= configs/default.yaml
MODEL ?= resnet18
CHECKPOINT ?= models/resnet18-aug_best.pt
ARGS ?=

# Pick the right PyTorch build for THIS machine automatically:
#   - NVIDIA GPU present (nvidia-smi on PATH) -> CUDA wheels  (cu126)
#   - otherwise (e.g. macOS / no GPU)         -> CPU wheels   (cpu)
# Override anytime, e.g.:  make test TORCH_EXTRA=cpu
TORCH_EXTRA ?= $(shell command -v nvidia-smi >/dev/null 2>&1 && echo cu126 || echo cpu)

# Every `uv run` goes through RUN so it requests the same extra. This keeps
# torch installed and stops uv from uninstalling/reinstalling it between targets.
RUN := $(UV) run --extra $(TORCH_EXTRA)

help:
	@printf "Available targets:\n"
	@printf "  make install         Install dependencies with uv\n"
	@printf "  make notebook        Start Jupyter Lab\n"
	@printf "  make mlflow-server   Start the MLflow tracking server\n"
	@printf "  make train           Train a model (MODEL=resnet18 ARGS='--augment')\n"
	@printf "  make evaluate        Evaluate a checkpoint (CHECKPOINT=models/...pt)\n"
	@printf "  make gradcam         Generate Grad-CAM grids from a checkpoint\n"
	@printf "  make validate        Run data quality gates + die-preservation report\n"
	@printf "  make serve           Run the FastAPI server locally (SERVE_HOST/PORT)\n"
	@printf "  make silver          Build the silver layer from bronze\n"
	@printf "  make gold            Build the gold layer from silver\n"
	@printf "  make verify-gold     Rebuild gold from silver and compare (gate)\n"
	@printf "\n"
	@printf "PyTorch build auto-detected for this machine: TORCH_EXTRA=$(TORCH_EXTRA)\n"
	@printf "  Override with e.g.: make install TORCH_EXTRA=cpu\n"

install:
	$(UV) sync --extra $(TORCH_EXTRA)

notebook:
	$(RUN) jupyter lab

mlflow-server:
	$(RUN) mlflow server \
		--backend-store-uri "$(MLFLOW_BACKEND_STORE_URI)" \
		--default-artifact-root "$(MLFLOW_ARTIFACT_ROOT)" \
		--host "$(MLFLOW_HOST)" \
		--port "$(MLFLOW_PORT)"

train:
	$(RUN) python -m wm811k.train --config $(CONFIG) --model $(MODEL) $(ARGS)

evaluate:
	$(RUN) python -m wm811k.evaluate --config $(CONFIG) --model $(MODEL) --checkpoint $(CHECKPOINT) $(ARGS)

gradcam:
	$(RUN) python -m wm811k.gradcam --config $(CONFIG) --model $(MODEL) --checkpoint $(CHECKPOINT)

validate:
	$(RUN) python -m wm811k.validate --config $(CONFIG)

serve:
	$(RUN) python -m wm811k.serve --config $(CONFIG) --model $(MODEL) --checkpoint $(CHECKPOINT) --host $(SERVE_HOST) --port $(SERVE_PORT)

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check src tests
	$(RUN) yamllint .github/ configs/

silver:
	$(RUN) python -m wm811k.pipeline silver --config $(CONFIG)

gold:
	$(RUN) python -m wm811k.pipeline gold --config $(CONFIG)

verify-gold:
	$(RUN) python -m wm811k.pipeline verify-gold --config $(CONFIG)
