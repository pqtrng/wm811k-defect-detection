"""Die-preservation quality metric
The preprocessing resizes variable-size wafer maps to 64x64 with
cv2.INTER_NEAREST -- mandatory to keep die values discrete, but lossy:
nearest-neighbor sampling can drop (or duplicate) sparse defect dies.
This module quantifies that loss on the real data instead of hand-waving it.

Definition -- density-normalized preservation rate, per wafer:

    rate = (defect_dies_after / 4096) / (defect_dies_before / (H * W))

i.e. "how much of the wafer's defect-signal DENSITY survives the resize".
1.0 = perfectly preserved, 0.0 = the defect pattern vanished entirely.
Raw before/after counts are not comparable across wafer sizes (a 300x300
wafer loses ~22x the dies of everything else just from resolution), which is
why the rate is density-normalized.

The resize function below is the EXACT transform from 02_preprocessing.ipynb
(cv2.resize on uint8, INTER_NEAREST) -- measuring a lookalike would measure
the wrong thing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('Agg')

RESIZE = 64


def resize_wafer(wm, size: int = RESIZE) -> np.ndarray:
    """Verbatim port of the resize used in 02_preprocessing.ipynb"""
    arr = np.asarray(wm, dtype=np.uint8)
    return cv2.resize(arr, (size, size), interpolation=cv2.INTER_NEAREST)


def flatten_label(value: object) -> str:
    """Extract a scalar label from WM-811K's nested label arrays.
    Public because the Medallion pipeline (pipeline.build_silver) reuses the
    exact same label-flattening as the quality report -- one rule, one source.
    Note: 02_preprocessing.ipynb used np.asarray(x).flatten() (a copy); this
    uses .ravel() (a view when possible). Values are identical -- both take the
    first element of the flattened array -- only the copy-vs-view differs.
    """
    values = np.asarray(value, dtype=object).ravel()
    return str(values[0]) if values.size else ""


def _normalized_labels(raw_df: pd.DataFrame) -> pd.Series:
    """Return labels from either processed or clean raw WM-811K data."""
    if "label" in raw_df:
        return raw_df["label"].map(flatten_label)
    if "failureType" in raw_df:
        return raw_df["failureType"].map(flatten_label)

    available = ", ".join(map(str, raw_df.columns))
    raise ValueError(
        "raw dataframe must contain a 'label' or 'failureType' column "
        f"(found: {available})"
    )


def die_preservation_report(raw_df: pd.DataFrame, labels: list[str], figures_dir: str | Path,
                            worst_k: int = 10) -> dict:
    """Compute per-wafer die preservation across the resize, print a summary table, save a histogram, and return the stats dict (used by tests)
    Args:
        raw_df: clean raw dataframe with `waferMap` (2D array) and either
            `failureType` (the nested source labels) or a flat `label` column.
        labels: the 8 defect classes; rows outside them are excluded (mirror preprocessing).
        figures_dir: where die_preservation.png is written.
        worst_k: how many worst offenders to print.
    """
    normalized_labels = _normalized_labels(raw_df)
    keep = normalized_labels.isin(labels)
    df = raw_df.loc[keep, ["waferMap"]].copy()
    df["label"] = normalized_labels.loc[keep].to_numpy()
    df = df.reset_index(drop=True)

    records = []
    for i, row in enumerate(df.itertuples(index=False)):
        arr = np.asarray(row.waferMap, dtype=np.uint8)
        before = int((arr == 2).sum())
        if before == 0:
            continue  # no defect dies to preserve; excluded from the rate
        after = int((resize_wafer(arr) == 2).sum())
        h, w = arr.shape
        rate = (after / (RESIZE * RESIZE)) / (before / (h * w))
        records.append(
            {"idx": i, "label": row.label, "dims": f"{h}x{w}",
             "before": before, "after": after, "rate": rate}
        )
    rep = pd.DataFrame.from_records(records)

    stats = {
        "n_wafers": len(rep),
        "n_excluded_zero_defect": len(df) - len(rep),
        "mean_rate": float(rep["rate"].mean()),
        "median_rate": float(rep["rate"].median()),
        "p5": float(rep["rate"].quantile(0.05)),
        "p25": float(rep["rate"].quantile(0.25)),
        "p75": float(rep["rate"].quantile(0.75)),
        "p95": float(rep["rate"].quantile(0.95)),
        "pct_total_loss": float((rep["rate"] == 0).mean() * 100),
    }

    print("\n=== Die-preservation across 64x64 INTER_NEAREST resize ===")
    print(f"wafers measured: {stats['n_wafers']:,} "
          f"(excluded, zero defect dies: {stats['n_excluded_zero_defect']:,})")
    print(f"density-normalized preservation rate: "
          f"mean={stats['mean_rate']:.3f} median={stats['median_rate']:.3f}")
    print(f"percentiles: p5={stats['p5']:.3f} p25={stats['p25']:.3f} "
          f"p75={stats['p75']:.3f} p95={stats['p95']:.3f}")
    print(f"wafers with TOTAL signal loss (rate == 0): {stats['pct_total_loss']:.2f}%")

    print(f"\nworst {worst_k} offenders:")
    worst = rep.nsmallest(worst_k, "rate")
    print(worst.to_string(index=False,
                          formatters={"rate": lambda r: f"{r:.3f}"}))

    per_class = rep.groupby("label")["rate"].agg(["mean", "median", "count"])
    print("\nper-class preservation:")
    print(per_class.round(3).to_string())

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "die_preservation.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(rep["rate"], bins=60, color="#3b6ea5")
    axes[0].axvline(1.0, color="red", ls="--", lw=1, label="perfect preservation")
    axes[0].set_xlabel("density-normalized preservation rate")
    axes[0].set_ylabel("wafers")
    axes[0].set_title("Defect-die preservation across resize")
    axes[0].legend()
    # Right panel: per-class DISTRIBUTION, not means -- the means are all ~1.0
    # (a bar chart shows eight identical bars); the story is in the tails,
    # where Loc/Scratch outliers dip to ~0.67.
    order = rep.groupby("label")["rate"].quantile(0.05).sort_values().index.tolist()
    data = [rep.loc[rep["label"] == c, "rate"] for c in order]
    axes[1].boxplot(
        data, vert=False, tick_labels=order,
        flierprops={"markersize": 2, "alpha": 0.4},
    )
    axes[1].axvline(1.0, color="red", ls="--", lw=1)
    axes[1].set_xlabel("preservation rate (per-wafer distribution)")
    axes[1].set_title("Per class")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {fig_path}")

    return stats
