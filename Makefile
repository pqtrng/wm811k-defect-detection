.PHONY: help install notebook mlflow-server

UV ?= uv
MLFLOW_HOST ?= 0.0.0.0
MLFLOW_PORT ?= 5000
MLFLOW_BACKEND_STORE_URI ?= sqlite:///$(CURDIR)/mlflow.db
MLFLOW_ARTIFACT_ROOT ?= $(CURDIR)/mlruns

help:
	@printf "Available targets:\n"
	@printf "  make install         Install dependencies with uv\n"
	@printf "  make notebook        Start Jupyter Lab\n"
	@printf "  make mlflow-server   Start the MLflow tracking server\n"

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
