"""Tests for wm811k.config -- YAML loading and path resolution.

The riskiest logic: project_root = config_path.parent.parent, i.e. paths
resolve relative to the directory ABOVE configs/. Move the YAML file and
every relative path silently shifts -- this test pins the rule down.
"""

from __future__ import annotations

import yaml
from wm811k.config import load_config


def _write_yaml(tmp_path, tracking_uri="auto"):
    raw = {
        "seed": 7,
        "labels": ["A", "B"],
        "paths": {
            "data_dir": "data",
            "processed_dir": "data/processed",
            "models_dir": "models",
            "figures_dir": "docs/figures",
        },
        "training": {
            "batch_size": 4, "lr": 0.01, "epochs": 1,
            "early_stopping_patience": 1,
        },
        "mlflow": {"tracking_uri": tracking_uri, "experiment_name": "t"},
    }
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    path = configs_dir / "test.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_paths_resolve_against_project_root(tmp_path):
    config = load_config(_write_yaml(tmp_path))
    # project_root = configs/.. = tmp_path
    assert config.paths.processed_dir == (tmp_path / "data/processed").resolve()
    assert config.seed == 7
    assert config.num_classes == 2
    assert config.label2idx == {"A": 0, "B": 1}


def test_tracking_uri_auto_becomes_sqlite_at_root(tmp_path):
    config = load_config(_write_yaml(tmp_path, tracking_uri="auto"))
    assert config.mlflow.tracking_uri == f"sqlite:///{tmp_path / 'mlflow.db'}"


def test_tracking_uri_explicit_passthrough(tmp_path):
    config = load_config(_write_yaml(tmp_path, tracking_uri="http://localhost:5000"))
    assert config.mlflow.tracking_uri == "http://localhost:5000"
