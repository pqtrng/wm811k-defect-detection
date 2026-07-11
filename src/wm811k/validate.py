"""CLI quality gate: python -m wm811k.validate
    1. Pandera schema validation on the three processed Parquets splits + cross-splut reconciliation. Any violation => non-zero exit (this is the gate a pipeline run must not pass through)
    2. Die-preservation report on the raw data (report, not a gate: it needs data/LSWMD_clean.pkl and is skipped with a warning if absent)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
from wm811k.config import load_config
from wm811k.quality import die_preservation_report
from wm811k.validation import validate_cross_split, validate_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the processed data layer.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--raw", default=None,
        help="Path to LSWMD_clean.pkl (default: <data_dir>/LSWMD_clean.pkl)",
    )
    parser.add_argument("--skip-quality", action="store_true",
                        help="Run only the schema gates")
    args = parser.parse_args()

    config = load_config(args.config)
    failed = False

    # --- Gate 1: per-split schemas ---
    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        path = config.paths.processed_dir / f"{name}.parquet"
        try:
            splits[name] = validate_split(path, config.labels)
            print(f"[PASS] schema: {name} ({len(splits[name]):,} rows)")
        except (pa.errors.SchemaErrors, pa.errors.SchemaError) as e:
            failed = True
            cases = getattr(e, "failure_cases", None)
            detail = cases.head(20) if cases is not None else e
            print(f"[FAIL] schema: {name}\n{detail}")
        except FileNotFoundError:
            failed = True
            print(f"[FAIL] schema: {name} — file not found: {path}")

    # --- Gate 2: cross-split reconciliation ---
    if len(splits) == 3:
        errors = validate_cross_split(splits, config.labels)
        if errors:
            failed = True
            for msg in errors:
                print(f"[FAIL] cross-split: {msg}")
        else:
            print("[PASS] cross-split: counts reconcile, all classes in all splits")

    # --- Report: die preservation (not a gate) ---
    if not args.skip_quality:
        raw_path = Path(args.raw) if args.raw else config.paths.data_dir / "LSWMD_clean.pkl"
        if raw_path.exists():
            raw_df = pd.read_pickle(raw_path)
            die_preservation_report(raw_df, config.labels, config.paths.figures_dir)
        else:
            print(f"[SKIP] die-preservation report — raw file not found: {raw_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
