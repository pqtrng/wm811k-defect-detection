"""FastAPI serving for the WM-811K defect classifier (T10).
Single-node by design (no Kubernetes): one process, one model, loaded once at
startup. The SAME WaferResNet18 serves both endpoints -- /predict for inference
and /predict?gradcam=true for explanations -- because Grad-CAM must hook the
model's real `layer4` (a plain nn.Module attribute). We deliberately do NOT load
the model through the MLflow registry here: the registry's production loader
returns a torch.export GraphModule whose `layer4` is gone, so Grad-CAM cannot
hook it. The registry decides WHICH checkpoint is production; serving just loads that .pt file. One
checkpoint, reused by predict, gradcam, and (later) the ONNX browser demo.

Run:
    uv run python -m wm811k.serve --checkpoint models/resnet18-aug_best.pt
Or via Docker: uvicorn wm811k.serve:app  (app is importable at module scope).

Input contract: RAW wafer maps with die values in {0, 1, 2}. Do NOT normalize
before sending -- the model divides by 2.0 inside forward(). Validation reuses
check_wafer_grid (one rule set, two doors): the same shape + value contract the
training pipeline's Parquet gates enforce.
"""
from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from wm811k.config import load_config
from wm811k.gradcam import GradCAM
from wm811k.models import build_model
from wm811k.validation import check_wafer_grid

# --- Startup configuration ---------------------------------------------------
# Read once at import time so both `python -m wm811k.serve` (which parses argv in
# __main__ and re-exports these) and `uvicorn wm811k.serve:app` (which imports
# the module directly, no argv) get a working default. Env vars let Docker
# override without argv, mirroring the Makefile's CHECKPOINT/CONFIG/MODEL names.
_CONFIG_PATH = os.environ.get("WM811K_CONFIG", "configs/default.yaml")
_MODEL_NAME = os.environ.get("WM811K_MODEL", "resnet18")
_CHECKPOINT = os.environ.get("WM811K_CHECKPOINT", "models/resnet18-aug_best.pt")
_DEVICE = os.environ.get("WM811K_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu"
)


def _load_model(checkpoint: str, model_name: str, num_classes: int, device: str):
    """Build the architecture and load a .pt state_dict onto `device`.

    Mirrors the load pattern in engine.py / gradcam.py exactly:
    build_model(...) then load_state_dict(torch.load(..., map_location=device)).
    Returns an eval-mode model with a real `layer4` for Grad-CAM to hook.
    """
    model = build_model(model_name, num_classes=num_classes)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# --- App state ---------------------------------------------------------------
# Populated in the lifespan handler; a plain dict keeps it simple and testable.
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model ONCE at startup (fail-fast: a missing checkpoint crashes
    the boot, not the first request). Config gives us the label order so we can
    map a class index back to a name."""
    config = load_config(_CONFIG_PATH)
    model = _load_model(_CHECKPOINT, _MODEL_NAME, config.num_classes, _DEVICE)
    _state["model"] = model
    _state["labels"] = config.labels
    _state["device"] = _DEVICE
    yield
    _state.clear()


app = FastAPI(title="WM-811K Defect Classifier", lifespan=lifespan)


# --- Request / response schemas ----------------------------------------------
class PredictRequest(BaseModel):
    """One wafer map. `wafer` is a flat 4096 list OR a nested 64x64 list of ints
    in {0, 1, 2}. Values are validated (and kept RAW) by check_wafer_grid."""

    wafer: list = Field(
        ...,
        description="Flat 4096 or nested 64x64 wafer map; die values in {0,1,2}.",
    )


def _to_tensor(grid: np.ndarray, device: str) -> torch.Tensor:
    """(64,64) raw float32 -> (1,1,64,64) tensor on `device`. NO normalization:
    the model's forward() owns the /2.0 step, so we feed raw {0,1,2}."""
    return torch.from_numpy(grid).unsqueeze(0).unsqueeze(0).to(device)


@app.get("/health")
def health():
    """Liveness + what actually loaded. Exposes the real device so we can verify
    (not assume) whether the checkpoint landed on cpu or cuda -- the T9 export
    warning made this worth surfacing explicitly."""
    return {
        "status": "ok" if "model" in _state else "no_model",
        "model_loaded": "model" in _state,
        "device": _state.get("device"),
        "num_classes": len(_state.get("labels", [])),
    }


@app.post("/predict")
def predict(req: PredictRequest, gradcam: bool = Query(default=False)):
    """Classify one wafer. With ?gradcam=true, also return a 64x64 CAM (values
    in [0,1]) explaining the prediction. Both paths use the SAME model, so the
    Grad-CAM hook on layer4 works."""
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="model not loaded")
    # One rule set, two doors: same contract as the training pipeline.
    try:
        grid = check_wafer_grid(req.wafer)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    model = _state["model"]
    labels = _state["labels"]
    device = _state["device"]
    x = _to_tensor(grid, device)
    if gradcam:
        # Grad-CAM hooks the model's layer4; only resnet18 has one. A model
        # without it (e.g. the plain CNN) cannot be explained this way -- say so
        # clearly (501) instead of crashing with an AttributeError (500).
        if not hasattr(model, "layer4"):
            raise HTTPException(
                status_code=501,
                detail=(
                    "Grad-CAM requires a model with a layer4 (e.g. resnet18); "
                    "the loaded model does not expose one"
                ),
            )
        # Grad-CAM needs gradients, so no torch.no_grad() here. It also returns
        # the predicted class, so we get prediction + explanation in one pass.
        with GradCAM(model, target_layer=model.layer4) as cam_engine:
            cam, pred = cam_engine.generate(x)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().tolist()
        return {
            "predicted_class": labels[pred],
            "class_index": pred,
            "probabilities": {labels[i]: probs[i] for i in range(len(labels))},
            "gradcam": cam.tolist(),
        }
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        pred = int(logits.argmax(dim=1).item())
    return {
        "predicted_class": labels[pred],
        "class_index": pred,
        "probabilities": {labels[i]: probs[i] for i in range(len(labels))},
    }


def main() -> None:
    """Entry point for `python -m wm811k.serve`, mirroring train.py/evaluate.py.
    Sets the module-level config from argv, then hands off to uvicorn."""
    parser = argparse.ArgumentParser(description="Serve the WM-811K classifier.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--checkpoint", default="models/resnet18-aug_best.pt")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Push argv into the env the module reads at import, so uvicorn's import of
    # `app` sees the same values. Set before importing uvicorn's runner.
    os.environ["WM811K_CONFIG"] = args.config
    os.environ["WM811K_MODEL"] = args.model
    os.environ["WM811K_CHECKPOINT"] = args.checkpoint

    import uvicorn

    uvicorn.run("wm811k.serve:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
