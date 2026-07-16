"""Tests for wm811k.serve -- the FastAPI serving layer (T10).

Design principles (same as conftest.py):
- API CONTRACT, never model quality. The model here is a random-init `cnn`
  built at test time, so predictions are meaningless -- we assert response
  SHAPE, status codes, and the {0,1,2}/64x64 input contract, never which class
  comes back. Model quality is verified by the evaluate CLI on real data.
- Self-contained: no checkpoint canonical, no real data, no GPU. Builds a tiny
  config YAML + a random state_dict on disk under tmp_path, so this runs on CI.
- serve.py reads its checkpoint/config/model from module-level vars set at
  import time. We monkeypatch those vars DIRECTLY (not just env) so the choice
  is independent of whether the module was already imported by another test,
  then enter TestClient as a context manager so lifespan actually loads.

"""

from __future__ import annotations
from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from wm811k import serve
from wm811k.models import build_model

LABELS = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]


def _make_client(tmp_path, monkeypatch, model_name: str):
    """Build a TestClient backed by a tiny on-disk config + a random checkpoint
    of `model_name`. Shared by the cnn and resnet18 fixtures below."""
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    config_path = configs_dir / "test.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "labels": LABELS,
                "paths": {
                    "data_dir": "data",
                    "bronze_dir": "data/bronze",
                    "silver_dir": "data/silver",
                    "gold_dir": "data/gold",
                    "models_dir": "models",
                    "figures_dir": "docs/figures",
                    "mlruns_dir": "mlruns",
                },
                "training": {
                    "batch_size": 8,
                    "lr": 1e-3,
                    "epochs": 1,
                    "early_stopping_patience": 2,
                },
                "mlflow": {
                    "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
                    "experiment_name": "wm811k-tests",
                },
            }
        )
    )

    checkpoint_path = tmp_path / f"random_{model_name}.pt"
    model = build_model(model_name, num_classes=len(LABELS))
    torch.save(model.state_dict(), checkpoint_path)

    monkeypatch.setattr(serve, "_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(serve, "_MODEL_NAME", model_name)
    monkeypatch.setattr(serve, "_CHECKPOINT", str(checkpoint_path))
    monkeypatch.setattr(serve, "_DEVICE", "cpu")

    return TestClient(serve.app)


@pytest.fixture
def serve_client(tmp_path, monkeypatch):
    """Default client: a random `cnn`. Fast; used for everything except the
    Grad-CAM success path (cnn has no layer4)."""
    with _make_client(tmp_path, monkeypatch, "cnn") as client:
        yield client


@pytest.fixture
def serve_client_resnet(tmp_path, monkeypatch):
    """Client backed by a random `resnet18` -- the only model with a layer4,
    so the only one whose Grad-CAM path can succeed."""
    with _make_client(tmp_path, monkeypatch, "resnet18") as client:
        yield client


def _flat_wafer(value: int = 0) -> list[int]:
    """A valid flat 4096 wafer, all one die value in {0,1,2}."""
    return [value] * 4096


def test_health_reports_loaded_model(serve_client):
    resp = serve_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["device"] == "cpu"
    assert body["num_classes"] == 8


def test_predict_returns_full_distribution(serve_client):
    resp = serve_client.post("/predict", json={"wafer": _flat_wafer(1)})
    assert resp.status_code == 200
    body = resp.json()
    # Contract: a valid class name, an index, and a full 8-way distribution.
    assert body["predicted_class"] in LABELS
    assert 0 <= body["class_index"] < 8
    assert len(body["probabilities"]) == 8
    assert set(body["probabilities"]) == set(LABELS)
    # Softmax must sum to ~1 (float tolerance).
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_accepts_square_64(serve_client):
    square = np.zeros((64, 64), dtype=int).tolist()
    resp = serve_client.post("/predict", json={"wafer": square})
    assert resp.status_code == 200
    assert resp.json()["predicted_class"] in LABELS


def test_predict_gradcam_returns_64x64_map(serve_client_resnet):
    resp = serve_client_resnet.post(
        "/predict", params={"gradcam": "true"}, json={"wafer": _flat_wafer(2)}
    )
    assert resp.status_code == 200
    body = resp.json()
    cam = np.array(body["gradcam"])
    assert cam.shape == (64, 64)
    # Grad-CAM output is normalized to [0, 1].
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0
    assert body["predicted_class"] in LABELS


def test_gradcam_unsupported_model_returns_501(serve_client):
    # The plain CNN has no layer4; asking for Grad-CAM must fail cleanly (501),
    # not crash with an AttributeError (500).
    resp = serve_client.post(
        "/predict", params={"gradcam": "true"}, json={"wafer": _flat_wafer(2)}
    )
    assert resp.status_code == 501


def test_predict_rejects_bad_value(serve_client):
    # Die value 3 violates the {0,1,2} contract -> 422 from check_wafer_grid.
    resp = serve_client.post("/predict", json={"wafer": _flat_wafer(3)})
    assert resp.status_code == 422


def test_predict_rejects_bad_shape(serve_client):
    # 100 elements is neither (4096,) nor (64,64) -> 422.
    resp = serve_client.post("/predict", json={"wafer": [0] * 100})
    assert resp.status_code == 422
