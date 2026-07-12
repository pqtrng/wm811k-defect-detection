"""Shared synthetic fixtures for the wm811k test suite.

Design principles:
- NO real data. CI runners have no access to WM-811K; unit tests must be
  fast (<60s total) and self-contained. Tests verify the pipeline CONTRACT
  (shapes, dtypes, value domains, determinism) -- never model quality.
  Model quality lives in the evaluate CLI against real data.
- Factory fixtures over fixed fixtures: validation tests must be able to
  construct *invalid* splits (die value 3, missing class, null label),
  so builders return callables instead of one blessed DataFrame.
- Synthetic wafers satisfy the exact processed-split contract:
  column `wafer` = flattened uint8 array of shape (4096,) with values
  in {0, 1, 2} (0 = off-wafer, 1 = good die, 2 = defect die);
  column `label` = one of the 8 defect classes.
- Each class gets a geometrically distinct defect pattern so a model can
  memorize the tiny set -- required by the 5-step overfit smoke test.
"""
import numpy as np
import pandas as pd
import pytest
from wm811k.config import TrainingConfig, MLFlowConfig, Config, PathConfig

LABELS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]
SIDE = 64


def make_wafer(label: str, rng: np.random.Generator) -> np.ndarray:
    """One synthetic 64x64 wafer, unit8 {0,1,2}, class-specific defect geometry.
    Not meant to look like the real WM-811K maps -- only to (a) statisfy the data contract exactly, and (b) be geometrically seperatable across classes.
    `rng` adds per-sample jitter so samples within a class are not identical.
    """
    center = (SIDE - 1) / 2.0
    yy, xx = np.mgrid[0:SIDE, 0:SIDE]
    radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    disk = radius <= SIDE * 0.48

    wafer = np.where(disk, 1, 0).astype(np.uint8)
    if label == "Center":
        defect = radius < rng.integers(5, 9)
    elif label == "Donut":
        inner = rng.integers(10, 14)
        defect = (radius > inner) & (radius < inner + 6)
    elif label == "Edge-Loc":
        angle = np.arctan2(yy - center, xx - center)
        a0 = rng.uniform(-np.pi, np.pi)
        delta = np.abs(np.angle(np.exp(1j * (angle - a0))))
        defect = (radius > SIDE * 0.36) & (delta < 0.5)
    elif label == "Edge-Ring":
        defect = radius > SIDE * 0.42
    elif label == "Loc":
        oy, ox = rng.integers(-12, 13, size=2)
        defect = np.sqrt((yy - center - oy) ** 2 + (xx - center - ox) ** 2) < 6
    elif label == "Near-full":
        defect = rng.random((SIDE, SIDE)) < 0.85
    elif label == "Random":
        defect = rng.random((SIDE, SIDE)) < 0.10
    elif label == "Scratch":
        offset = rng.integers(-10, 11)
        defect = np.abs((yy - center) - (xx - center) - offset) < 1.5
    else:
        raise ValueError(f"Unknown label: {label}")

    wafer[defect & disk] = 2
    return wafer


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG for wafer synthesis -- fixed seed so fixtures are stable run-to-run"""
    return np.random.default_rng(42)


@pytest.fixture
def labels() -> list[str]:
    """Canonical defect labels, returned as a fresh list for each test."""
    return list(LABELS)


@pytest.fixture
def make_split_df(rng):
    """Factory: build one valid processed-split DataFrame.
    Returns a callable so tests control size/classes and can mutate the result BEFORE writing to Parquet (negative tests corrupt on purpose).
    """

    def _make(n_per_class: int = 4, labels: list[str] | None = None) -> pd.DataFrame:
        labels = LABELS if labels is None else labels
        rows = [
            {"wafer": make_wafer(label, rng).reshape(-1), "label": label} for label in labels for _ in
            range(n_per_class)
        ]
        return pd.DataFrame(rows)

    return _make


@pytest.fixture
def write_parquet(tmp_path):
    """Factory:write a split DataFrame to Parquet under tmp_path, return the path
    Round-trips through pyarrow exactly like the real pipeline, so WaferDataset / validate_split in tests read what they red in production
    """

    def _write(df: pd.DataFrame, name: str = "split.parquet"):
        path = tmp_path / name
        df.to_parquet(path, engine="pyarrow", index=False)
        return path

    return _write


@pytest.fixture
def tiny_config(tmp_path, make_split_df) -> Config:
    """A complete frozen Config over a tiny on-disk processed dir.
    Writes train/val/test Parquets (8 classes x 4 samples each) into tmp_path/processed and points all paths at tmp_path, so build_loaders and the engine can run end-to-end without the repo's configs/ or data/
    """
    processed = tmp_path / "processed"
    processed.mkdir()
    for split in ("train", "val", "test"):
        make_split_df(n_per_class=4).to_parquet(
            processed / f"{split}.parquet", engine="pyarrow", index=False
        )

    return Config(
        seed=42,
        labels=list(LABELS),
        paths=PathConfig(
            data_dir=tmp_path / "data",
            processed_dir=processed,
            models_dir=tmp_path / "models",
            figures_dir=tmp_path / "figures",
        ),
        training=TrainingConfig(
            batch_size=8, lr=1e-3, epochs=1, early_stopping_patience=2
        ),
        mlflow=MLFlowConfig(
            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
            experiment_name="wm811k-tests",
        )
    )
