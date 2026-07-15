"""Pandera quality gates for the processed Parquet data layer.

These are DATA gates, not unit tests: they validate what is actually on disk
against the pipeline contract, and are meant to fail loudly (non-zero exit via
the validate CLI) when an upstream change silently corrupts the data.

Contract for each processed split (train/val/test Parquet):
- column `wafer`: flattened uint8 array of shape (4096,) = 64x64,
  values a subset of {0, 1, 2} (0 = off-wafer, 1 = good die, 2 = defect die)
- column `label`: one of the 8 defect classes, no nulls

Cross-split contract:
- row counts reconcile: train + val + test == 25,519 (fixed by preprocessing)
- every class present in every split (stratified split guarantee)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

# Pipeline constants - the preprocessing contract, not tunables.
EXPECTED_TOTAL = 25_519  # 8-class subset of WM-811K after dropping `none`/unlabeled
WAFER_FLAT_SHAPE = (4096,)  # 64 * 64
ALLOWED_DIE_VALUES = frozenset({0, 1, 2})


def _wafer_shapes_ok(s: pd.Series) -> pd.Series:
    return s.map(lambda a: np.array(a).shape == WAFER_FLAT_SHAPE)


def _wafer_values_ok(s: pd.Series) -> pd.Series:
    return s.map(lambda a: bool(np.isin(np.asarray(a), list(ALLOWED_DIE_VALUES)).all()))


def build_split_schema(labels: list[str]) -> pa.DataFrameSchema:
    """Schema for one processed Parquet split"""
    return pa.DataFrameSchema(
        {
            "wafer": pa.Column(
                dtype=object,
                nullable=False,
                checks=[
                    pa.Check(_wafer_shapes_ok, error=f"wafer shape != {WAFER_FLAT_SHAPE}"),
                    pa.Check(_wafer_values_ok, error="die value not in {0, 1, 2}")]),
            "label": pa.Column(
                dtype=str,
                nullable=False,
                checks=pa.Check.isin(labels, error="Unknown defect label")
            )
        },
        strict=False
    )


def validate_split(parquet_path: str | Path, labels: list[str]) -> pd.DataFrame:
    """Validate one processed Parquet split; raises pandera.errors.SchemaErrors on violation"""
    df = pd.read_parquet(path=parquet_path, engine="pyarrow")
    return build_split_schema(labels).validate(df, lazy=True)


def validate_cross_split(splits: dict[str, pd.DataFrame], labels: list[str]) -> list[str]:
    """Cross-split contract checks. Returns a list of violation messages (empty = OK)"""
    errors: list[str] = []
    total = sum(len(df) for df in splits.values())
    if total != EXPECTED_TOTAL:
        errors.append(
            f"row counts do not reconcile: "
            f"{' + '.join(f'{k}={len(v)}' for k, v in splits.items())} "
            f"= {total}, expected {EXPECTED_TOTAL}"
        )

    for name, df in splits.items():
        missing = set(labels) - set(df["label"].unique())
        if missing:
            errors.append(f"split '{name}' is missing class(es): {sorted(missing)}")

    return errors


# --- Single-grid gate: reused by the serving layer (T10) ---------------------
# The Parquet gates above validate a whole split on disk. The API validates one
# request at a time, but the RULE must be identical -- one rule set, two doors --
# so a wafer the model was trained on and a wafer the API accepts can never
# diverge. check_wafer_grid is that shared door for a single sample.

WAFER_SIDE = 64  # 64 x 64; WAFER_FLAT_SHAPE == (WAFER_SIDE * WAFER_SIDE,)


def check_wafer_grid(grid: object) -> np.ndarray:
    """Validate ONE wafer map coming from an API request; return it as float32.

    Accepts either a flat sequence of 4096 values or a 64x64 nested sequence.
    Applies the SAME shape + die-value contract as the Parquet split schema
    (WAFER_FLAT_SHAPE, ALLOWED_DIE_VALUES), so serving cannot accept a wafer the
    training pipeline would have rejected.

    Args:
        grid: a wafer map -- flat length-4096 or nested 64x64 -- of ints in
            {0, 1, 2} (0 = off-wafer, 1 = good die, 2 = defect die).

    Returns:
        np.ndarray of shape (64, 64), dtype float32, RAW values in {0, 1, 2}.
        Caller must NOT normalize: the model divides by 2.0 inside forward().

    Raises:
        ValueError: wrong shape or a value outside {0, 1, 2}. The message is
            safe to surface to an API client (no internals leaked).
    """
    try:
        arr = np.asarray(grid)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"wafer is not array-like: {exc}") from exc

    # Accept flat (4096,) or square (64, 64); reject everything else.
    if arr.shape == WAFER_FLAT_SHAPE:
        arr = arr.reshape(WAFER_SIDE, WAFER_SIDE)
    elif arr.shape != (WAFER_SIDE, WAFER_SIDE):
        raise ValueError(
            f"wafer shape must be {WAFER_FLAT_SHAPE} (flat) or "
            f"({WAFER_SIDE}, {WAFER_SIDE}) (square), got {arr.shape}"
        )

    # Same die-value contract as _wafer_values_ok on the Parquet gate.
    if not np.isin(arr, list(ALLOWED_DIE_VALUES)).all():
        bad = sorted(set(np.unique(arr).tolist()) - ALLOWED_DIE_VALUES)
        raise ValueError(
            f"die value(s) not in {sorted(ALLOWED_DIE_VALUES)}: {bad}"
        )
    
    return arr.astype(np.float32)
