"""CLI entry point: evaluate a trained checkpoint with per-class reporting.

Usage:
    uv run python -m wm811k.evaluate --checkpoint models/resnet18-aug_best.pt \
        --model resnet18 --split test
"""

from __future__ import annotations

import argparse

from wm811k.config import load_config
from wm811k.data import build_loaders
from wm811k.engine import evaluate_and_report
from wm811k.models import build_model
from wm811k.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint: per-class report + confusion matrix."
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to the YAML config (default: configs/default.yaml)",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt state_dict")
    parser.add_argument(
        "--model", default="resnet18", choices=["cnn", "resnet18"],
        help="Architecture the checkpoint was trained with (default: resnet18)",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Which split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--title", default=None,
        help="Report title / figure name (default: '<model> (<split>)')",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    # augment=False always: evaluation must run on raw, untransformed data.
    train_loader, val_loader, test_loader = build_loaders(config, augment=False)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    model = build_model(args.model, num_classes=config.num_classes)
    title = args.title or f"{args.model} ({args.split})"

    cm_path = evaluate_and_report(
        best_path=args.checkpoint,
        model=model,
        loader=loader,
        title=title,
        config=config,
    )
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()
