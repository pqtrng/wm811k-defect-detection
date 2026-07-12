"""Config loading for wm811k pipeline.

Ported from the hardcoded globals in notebooks/03_train.ipynb (SEED, BATCH, LR, EPOCHS, DATA_DIR, LABELS) into a single YAML-driven, type-hinted Config. No values were changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PathConfig:
    """Filesystem paths, resolved to absolute paths relative to project root.
    Medallion data layers:
        - bronze_dir: raw data, immutable (LSWMD.pkl, LSWMD_clean.pkl)
        - silver_dir: one cleaned table (wafer.parquet), filtered + resized, not split
        - gold_dir: train/val/test.parquet the models consume
    """

    data_dir: Path
    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path
    models_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters ported verbatim from notebooks"""

    batch_size: int
    lr: float
    epochs: int
    early_stopping_patience: int


@dataclass(frozen=True)
class MLFlowConfig:
    """MLflow tracking configuration."""

    tracking_uri: str
    experiment_name: str


@dataclass(frozen=True)
class Config:
    """Top level pipeline configuration."""

    seed: int
    labels: list[str] = field(default_factory=list)
    paths: PathConfig = None  # type: ignore[assignment]
    training: TrainingConfig = None  # type: ignore[assignment]
    mlflow: MLFlowConfig = None  # type: ignore[assignment]

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    @property
    def label2idx(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.labels)}


def load_config(path: str | Path) -> Config:
    """Load a Config from a YAML file.
    The YAML's parent directory (e.g. `configs/`) is expected to sit directly under the project root, so `project_root = path.parent.parent`. All relative paths in the `paths` section are resolved against that root.
    """
    config_path = Path(path).resolve()
    project_root = config_path.parent.parent

    with config_path.open("r") as f:
        raw = yaml.safe_load(f)

    paths_raw = raw["paths"]
    paths = PathConfig(
        data_dir=(project_root / paths_raw["data_dir"]).resolve(),
        bronze_dir=(project_root / paths_raw["bronze_dir"]).resolve(),
        silver_dir=(project_root / paths_raw["silver_dir"]).resolve(),
        gold_dir=(project_root / paths_raw["gold_dir"]).resolve(),
        models_dir=(project_root / paths_raw["models_dir"]).resolve(),
        figures_dir=(project_root / paths_raw["figures_dir"]).resolve(),
    )

    training = TrainingConfig(
        **raw["training"],
    )

    mlflow_raw = raw["mlflow"]
    tracking_uri = mlflow_raw["tracking_uri"]
    if tracking_uri == "auto":
        db_path = project_root / "mlflow.db"
        tracking_uri = f"sqlite:///{db_path}"

    mlflow_cfg = MLFlowConfig(
        tracking_uri=tracking_uri, experiment_name=mlflow_raw["experiment_name"]
    )

    return Config(
        seed=raw["seed"],
        labels=raw["labels"],
        paths=paths,
        training=training,
        mlflow=mlflow_cfg,
    )
