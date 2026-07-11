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
