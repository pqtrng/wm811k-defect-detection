"""CLI entry point: train a wafer defect classifier.

Usage:
    uv run python -m wm811k.train --config configs/default.yaml --model resnet18 --augment
    uv run python -m wm811k.train --model resnet18 --augment --epochs 2   # smoke test

The config file is the source of truth for hyperparameters; --epochs is the
only override, provided for cheap end-to-end smoke tests before a full run.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from wm811k.config import load_config
from wm811k.data import build_loaders
from wm811k.engine import fit
from wm811k.models import build_model
from wm811k.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a wafer defect classifier.")
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to the YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--model", default="resnet18", choices=["cnn", "resnet18"],
        help="Model architecture to train (default: resnet18)",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable domain-safe augmentation (rot90/flip) on the train split",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="MLflow run name (default: '<model>' or '<model>-aug')",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override config epochs (for smoke tests only)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        # Config dataclasses are frozen: build a modified copy instead of mutating.
        config = replace(config, training=replace(config.training, epochs=args.epochs))

    run_name = args.run_name or (f"{args.model}-aug" if args.augment else args.model)

    set_seed(config.seed)
    train_loader, val_loader, _ = build_loaders(config, augment=args.augment)
    model = build_model(args.model, num_classes=config.num_classes)

    best_path = fit(
        model=model,
        run_name=run_name,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
