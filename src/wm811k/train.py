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
from wm811k.engine import evaluate_and_report, fit, log_and_register
from wm811k.models import build_model
from wm811k.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a wafer defect classifier.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML config (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=["cnn", "resnet18"],
        help="Model architecture to train (default: resnet18)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable domain-safe augmentation (rot90/flip) on the train split",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name (default: '<model>' or '<model>-aug')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override config epochs (for smoke tests only)",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="After training, log the best model into the run and register a new "
        "registry version. Does NOT promote to @production (manual step: "
        "`python -m wm811k.registry promote`).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        # Config dataclasses are frozen: build a modified copy instead of mutating.
        config = replace(config, training=replace(config.training, epochs=args.epochs))

    run_name = args.run_name or (f"{args.model}-aug" if args.augment else args.model)

    set_seed(config.seed)
    train_loader, val_loader, test_loader = build_loaders(config, augment=args.augment)
    model = build_model(args.model, num_classes=config.num_classes)

    best_path, run_id = fit(
        model=model,
        run_name=run_name,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    print(f"Best checkpoint: {best_path}")
    print(f"Run id: {run_id}")
    
    if args.register:
        # Test metrics + model artifact go into the SAME run fit() created, so the
        # registered version carries its own per-class metrics (what compare reads).
        # Evaluate on the RAW test split (augment is train-only; loaders already
        # built test without augmentation).
        evaluate_and_report(
            best_path=best_path,
            model=model,
            loader=test_loader,
            title=run_name,
            config=config,
            run_id=run_id,
        )
        version = log_and_register(
            run_id=run_id,
            model=model,
            best_path=best_path,
            config=config,
            register=True,
        )
        print(f"Registered version: {version}")
        print(
            "Review per-class deltas before promoting:\n"
            f"  python -m wm811k.registry compare --candidate {version}\n"
            f"  python -m wm811k.registry promote --version {version}"
        )


if __name__ == "__main__":
    main()
