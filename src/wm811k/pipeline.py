"""Medallion data pipeline for WM-811K: bronze -> silver -> gold.

Layers (see PathConfig):
- bronze/  immutable raw pickles (LSWMD.pkl, LSWMD_clean.pkl). Never written here.
- silver/  wafers.parquet: 8-class subset, resized 64x64 INTER_NEAREST,
           flattened uint8 -- NOT split. One cleaned table.
- gold/    train/val/test.parquet: stratified 70/15/15 split the models consume.

The split is a MODELING decision (ratios, seed), not a property of the data,
so it lives in build_gold(), separate from silver. That separation is what
lets verify_gold() deterministically rebuild gold from silver and compare.

All transform logic is ported VERBATIM from notebooks/02_preprocessing.ipynb:
- flatten_label / resize_wafer are reused from wm811k.quality (one source of truth).
- whitelist of 8 DEFECT_CLASSES, INTER_NEAREST resize, uint8 flatten to (4096,),
  2-stage stratified split (TEST_SIZE=0.15, val_relative=VAL_SIZE/(1-TEST_SIZE),
  random_state=seed on BOTH stages).
Differences from the notebook are limited to structure (functions + a CLI),
never to values or ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from wm811k.config import Config, load_config
from wm811k.quality import flatten_label, resize_wafer

# Preprocessing contract, ported verbatim from 02_preprocessing.ipynb.
DEFECT_CLASSES = [
    "Edge-Ring", "Edge-Loc", "Center", "Loc",
    "Scratch", "Random", "Donut", "Near-full",
]
TEST_SIZE = 0.15
VAL_SIZE = 0.15
CLEAN_PKL_NAME = "LSWMD_clean.pkl"
SILVER_NAME = "wafers.parquet"
GOLD_SPLITS = ("train", "val", "test")


def _to_parquet_df(split_df: pd.DataFrame) -> pd.DataFrame:
    """Build the 4-column output frame from a split (verbatim from notebook cell 7).

    Columns: wafer (flattened uint8 (4096,)), label, lotName, waferIndex.
    lotName/waferIndex are carried through so gold rows are individually
    identifiable -- they are the fingerprint keys verify_gold relies on.
    """
    return pd.DataFrame(
        {
            "wafer": split_df["wafer_64"].apply(lambda a: a.astype(np.uint8).flatten()),
            "label": split_df["label"].values,
            "lotName": split_df["lotName"].values,
            "waferIndex": split_df["waferIndex"].values,
        }
    )


def build_silver(config: Config) -> Path:
    """bronze/LSWMD_clean.pkl -> silver/wafers.parquet (filtered + resized, NOT split).

    Verbatim port of notebook cells 1-5 + 7, collapsed into a single table.
    """
    clean_pkl = config.paths.bronze_dir / CLEAN_PKL_NAME
    if not clean_pkl.exists():
        raise FileNotFoundError(
            f"bronze clean pickle not found: {clean_pkl}. "
            "Place LSWMD_clean.pkl under the bronze layer first."
        )

    df = pd.read_pickle(clean_pkl)
    print(f"Loaded bronze: {df.shape}")

    df["label"] = df["failureType"].apply(flatten_label)

    before = len(df)
    df = df[df["label"].isin(DEFECT_CLASSES)].reset_index(drop=True)
    after = len(df)
    print(f"Filtered to 8 classes: {before} -> {after} rows ({before - after} dropped)")
    assert set(df["label"].unique()) == set(DEFECT_CLASSES), "Unexpected labels remain!"

    df["wafer_64"] = df["waferMap"].apply(resize_wafer)
    print(f"Resized all {len(df)} wafers to 64x64 (INTER_NEAREST)")

    out_df = _to_parquet_df(df)

    silver_dir = config.paths.silver_dir
    silver_dir.mkdir(parents=True, exist_ok=True)
    out_path = silver_dir / SILVER_NAME
    out_df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Wrote {out_path} ({len(out_df):,} rows)")
    return out_path


def _split_silver(silver_df: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """2-stage stratified split, verbatim from notebook cell 6.

    Operates on the silver frame directly (already 4-column, already resized),
    so _to_parquet_df is NOT reapplied -- silver rows pass straight through.
    """
    df_trainval, df_test = train_test_split(
        silver_df, test_size=TEST_SIZE, stratify=silver_df["label"], random_state=seed
    )
    val_relative = VAL_SIZE / (1 - TEST_SIZE)
    df_train, df_val = train_test_split(
        df_trainval, test_size=val_relative, stratify=df_trainval["label"],
        random_state=seed,
    )
    return {"train": df_train, "val": df_val, "test": df_test}


def build_gold(config: Config, gold_dir: Path | None = None) -> dict[str, Path]:
    """silver/wafers.parquet -> gold/{train,val,test}.parquet (stratified 70/15/15).

    gold_dir override lets verify_gold rebuild into a temp dir without touching
    the real gold layer.
    """
    silver_path = config.paths.silver_dir / SILVER_NAME
    if not silver_path.exists():
        raise FileNotFoundError(
            f"silver table not found: {silver_path}. Run `pipeline silver` first."
        )

    silver_df = pd.read_parquet(silver_path, engine="pyarrow")
    splits = _split_silver(silver_df, seed=config.seed)

    target_dir = gold_dir if gold_dir is not None else config.paths.gold_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for name in GOLD_SPLITS:
        out_path = target_dir / f"{name}.parquet"
        splits[name].to_parquet(
            out_path, engine="pyarrow", compression="snappy", index=False
        )
        written[name] = out_path
        print(f"Wrote {out_path} ({len(splits[name]):,} rows)")
    return written


def _fingerprint_split(df: pd.DataFrame) -> list[str]:
    """Order-independent row-level fingerprint of a gold split.

    Each row hashed over (wafer bytes, label, lotName, waferIndex); the SORTED
    list of per-row hashes identifies the split regardless of row order. Two
    splits with the same rows in a different order fingerprint identically;
    any changed/added/dropped/reassigned row changes the fingerprint.
    """
    hashes: list[str] = []
    for row in df.itertuples(index=False):
        h = hashlib.sha1()
        h.update(np.asarray(row.wafer, dtype=np.uint8).tobytes())
        h.update(str(row.label).encode())
        h.update(str(row.lotName).encode())
        h.update(str(row.waferIndex).encode())
        hashes.append(h.hexdigest())
    hashes.sort()
    return hashes


def verify_gold(config: Config) -> bool:
    """Rebuild gold from silver into a temp dir and compare row-level fingerprints
    against the gold currently on disk, per split.

    Returns True on full match. On MISMATCH: prints the offending split(s) and
    returns False WITHOUT modifying anything -- the caller must stop and discuss,
    never auto-heal the on-disk gold to match.
    """
    gold_dir = config.paths.gold_dir
    for name in GOLD_SPLITS:
        p = gold_dir / f"{name}.parquet"
        if not p.exists():
            print(f"[FAIL] verify-gold: on-disk gold missing {name}.parquet ({p})")
            return False

    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = build_gold(config, gold_dir=Path(tmp))

        all_match = True
        for name in GOLD_SPLITS:
            on_disk = pd.read_parquet(gold_dir / f"{name}.parquet", engine="pyarrow")
            fresh = pd.read_parquet(rebuilt[name], engine="pyarrow")

            fp_disk = _fingerprint_split(on_disk)
            fp_fresh = _fingerprint_split(fresh)

            if fp_disk == fp_fresh:
                print(f"[MATCH] verify-gold: {name} ({len(on_disk):,} rows)")
            else:
                all_match = False
                print(
                    f"[MISMATCH] verify-gold: {name} "
                    f"(on-disk {len(on_disk):,} rows, rebuilt {len(fresh):,} rows)"
                )

    if not all_match:
        print(
            "\nverify-gold FAILED. The on-disk gold does not match a fresh "
            "deterministic rebuild from silver. STOP -- do not regenerate gold "
            "to force a match. Investigate why silver or the split logic drifted."
        )
    return all_match


def main() -> None:
    parser = argparse.ArgumentParser(description="Medallion data pipeline (bronze/silver/gold).")
    parser.add_argument(
        "stage", choices=["silver", "gold", "all", "verify-gold"],
        help="silver: build silver from bronze; gold: build gold from silver; "
             "all: silver then gold; verify-gold: rebuild-and-compare gate",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.stage == "silver":
        build_silver(config)
    elif args.stage == "gold":
        build_gold(config)
    elif args.stage == "all":
        build_silver(config)
        build_gold(config)
    elif args.stage == "verify-gold":
        ok = verify_gold(config)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
