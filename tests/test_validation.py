"""Tests for wm811k.validation -- the Pandera data gates.

Every gate gets a negative test: a gate that has never been seen failing
is a gate that has never been proven. The lazy=True test is promoted to a permanent regression test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.errors as pa_errors
import pytest
from wm811k.validation import (
    EXPECTED_TOTAL,
    validate_split,
    validate_cross_split, check_wafer_grid
)


def test_valid_split_passes(make_split_df, write_parquet, labels):
    path = write_parquet(make_split_df())
    validated = validate_split(path, labels)  # must not raise
    assert len(validated) == 8 * 4


def test_rejects_bad_die_value(make_split_df, write_parquet, labels):
    df = make_split_df()
    df.loc[0, "wafer"][100] = 3  # in-place: {0,1,2} contract violated
    path = write_parquet(df)
    with pytest.raises(pa_errors.SchemaErrors):
        validate_split(path, labels)


def test_rejects_unknown_label(make_split_df, write_parquet, labels):
    df = make_split_df()
    df.loc[0, "label"] = "Banana"
    path = write_parquet(df)
    with pytest.raises(pa_errors.SchemaErrors):
        validate_split(path, labels)


def test_rejects_null_label(make_split_df, write_parquet, labels):
    df = make_split_df()
    df.loc[0, "label"] = None
    path = write_parquet(df)
    with pytest.raises(pa_errors.SchemaErrors):
        validate_split(path, labels)


def test_rejects_wrong_wafer_shape(make_split_df, write_parquet, labels):
    df = make_split_df()
    df.at[0, "wafer"] = df.at[0, "wafer"][:100]  # (100,) != (4096,)
    path = write_parquet(df)
    with pytest.raises(pa_errors.SchemaErrors):
        validate_split(path, labels)


def test_lazy_collects_violations_across_columns(make_split_df, write_parquet, labels):
    """lazy=True must gather ALL violations in one run, not stop at the first.

    Corrupt wafer (die value 3) AND label (Banana) in the same DataFrame;
    failure_cases must report both columns. If someone later drops lazy=True,
    validation stops at the first column and this test goes red.
    """
    df = make_split_df()
    df.loc[0, "wafer"][100] = 3
    df.loc[1, "label"] = "Banana"
    path = write_parquet(df)

    with pytest.raises(pa_errors.SchemaErrors) as exc_info:
        validate_split(path, labels)

    failed_columns = set(exc_info.value.failure_cases["column"])
    assert {"wafer", "label"} <= failed_columns


def _label_only_df(n_rows: int, labels: list[str]) -> pd.DataFrame:
    """Lightweight split stand-in: validate_cross_split only reads len() and
    the `label` column, so no wafers needed -- 25,519 rows stays cheap."""
    cycled = np.resize(np.array(labels, dtype=object), n_rows)
    return pd.DataFrame({"label": cycled})


def test_cross_split_ok(labels):
    n_train = EXPECTED_TOTAL - 2 * 2_000
    splits = {
        "train": _label_only_df(n_train, labels),
        "val": _label_only_df(2_000, labels),
        "test": _label_only_df(2_000, labels),
    }
    assert validate_cross_split(splits, labels) == []


def test_cross_split_detects_bad_total(labels):
    splits = {
        "train": _label_only_df(100, labels),
        "val": _label_only_df(50, labels),
        "test": _label_only_df(50, labels),
    }
    errors = validate_cross_split(splits, labels)
    assert any("do not reconcile" in e for e in errors)


def test_cross_split_detects_missing_class(labels):
    n_train = EXPECTED_TOTAL - 2 * 2_000
    splits = {
        "train": _label_only_df(n_train, labels),
        "val": _label_only_df(2_000, labels[:-1]),  # Scratch missing from val
        "test": _label_only_df(2_000, labels),
    }
    errors = validate_cross_split(splits, labels)
    assert any("val" in e and "Scratch" in e for e in errors)


# --- check_wafer_grid: the single-grid serving gate (T10) --------------------
# Same {0,1,2} + shape contract as the Parquet gates above, but for one wafer
# arriving at the API. Every rejection path gets a negative test; a gate never
# seen failing is a gate never proven.
def test_check_wafer_grid_accepts_flat_4096():
    flat = ([0, 1, 2] * 1365) + [0]  # length 4096
    out = check_wafer_grid(flat)
    assert out.shape == (64, 64)


def test_check_wafer_grid_accepts_square_64():
    out = check_wafer_grid(np.zeros((64, 64), dtype=int))
    assert out.shape == (64, 64)


def test_check_wafer_grid_returns_float32():
    out = check_wafer_grid(np.ones((64, 64), dtype=int))
    assert out.dtype == np.float32


def test_check_wafer_grid_keeps_raw_values():
    # The model divides by 2.0 inside forward(); the gate must NOT pre-normalize.
    out = check_wafer_grid(np.full((64, 64), 2, dtype=int))
    assert out.max() == 2.0


def test_check_wafer_grid_rejects_bad_value():
    with pytest.raises(ValueError, match="die value"):
        check_wafer_grid(np.full((64, 64), 3, dtype=int))


def test_check_wafer_grid_rejects_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        check_wafer_grid(np.zeros((10, 10), dtype=int))
