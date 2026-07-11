"""Training and evaluation engine for wm811k pipeline.

Ported from notebooks/03_train.ipynb: train_one_epoch, evaluate,
run_experiment (renamed fit), evaluate_and_report.

Deliberate, experimentally validated decisions (do not "fix" these):
- Checkpoint selection uses BEST VAL_LOSS, never val_macro_f1.
  A macro-F1-based checkpoint was tried and gave 0.751 vs 0.831 -- the F1
  signal is too noisy on small minority classes for stable selection.
- ReduceLROnPlateau also monitors val_loss for the same stability reason.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on servers/CI
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from wm811k.config import Config


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # enable Dropout + BatchNorm use batch stats
    running_loss = 0.0
    n_samples = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()  # clear gradients from previous step
        logits = model(xb)  # forward
        loss = criterion(logits, yb)  # how wrong
        loss.backward()  # backprop: compute gradients
        optimizer.step()  # nudge parameters

        running_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
    return running_loss / n_samples


@torch.no_grad()  # no gradients needed for eval -> faster, less memory
def evaluate(model, loader, criterion, device):
    model.eval()  # disables Dropout + BatchNorm uses running stats
    running_loss = 0.0
    n_samples = 0
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        running_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(yb.cpu())

    avg_loss = running_loss / n_samples
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return avg_loss, preds, labels


def fit(
        model: nn.Module,
        run_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: torch.device | None = None,
) -> Path:
    """Train with MLflow logging, LR scheduling, and early stopping.
    Returns the path to the best checkpoint selected by val_loss.

    Ported from the notebook's run_experiment; hyperparameters now come from
    config instead of module globals.
    """
    device = device or _default_device()
    epochs = config.training.epochs
    lr = config.training.lr
    es_patience = config.training.early_stopping_patience

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.5, patience=3
    )
    best_val_loss = float("inf")
    epochs_no_improve = 0
    config.paths.models_dir.mkdir(parents=True, exist_ok=True)
    best_path = config.paths.models_dir / f"{run_name}_best.pt"
    run_params = {
        "model": run_name,
        "epochs": epochs,
        "batch_size": config.training.batch_size,
        "lr": lr,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "sampler": "WeightedRandomSampler",
        "scheduler": "ReduceLROnPlateau(factor=0.5, patience=3)",
        "early_stopping_patience": es_patience,
        "seed": config.seed,
        "img_size": 64,
        "num_classes": config.num_classes,
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(run_params)
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model=model, loader=train_loader, criterion=criterion,
                optimizer=optimizer, device=device,
            )
            val_loss, val_preds, val_labels = evaluate(
                model=model, loader=val_loader, criterion=criterion, device=device,
            )
            val_macro_f1 = f1_score(y_true=val_labels, y_pred=val_preds, average="macro")

            scheduler.step(metrics=val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss,
                 "val_macro_f1": val_macro_f1, "lr": current_lr},
                step=epoch,
            )

            print(
                f"epoch {epoch:2d} | train {train_loss:.4f} | val_loss {val_loss:.4f}"
                f" | val_macro-F1: {val_macro_f1:.4f} | lr: {current_lr:.2e}"
            )

            # track the best + early stopping counter
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_path)
                mlflow.log_artifact(str(best_path))
                print(f"\t-> saved best (val_loss {val_loss:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= es_patience:
                    print(
                        f"\t-> early stopping at epoch {epoch}"
                        f" (no improvement for {epochs_no_improve} epochs)"
                    )
                    break
    print(f"Done! Best val_loss: {best_val_loss:.4f}")
    print(f"MLflow run id: {run.info.run_id}")
    return best_path


def evaluate_and_report(
        best_path: str | Path,
        model: nn.Module,
        loader: DataLoader,
        title: str,
        config: Config,
        device: torch.device | None = None,
) -> Path:
    """Load best checkpoint, run per-class report + confusion matrix on loader.

    Never reports aggregate accuracy alone: per-class precision/recall/F1 and
    the confusion matrix are the point (class imbalance hides regressions).
    Logs test metrics to MLflow and saves the confusion matrix under
    config.paths.figures_dir.
    """
    device = device or _default_device()
    criterion = nn.CrossEntropyLoss()

    model = model.to(device=device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_preds, test_labels = evaluate(
        model=model, loader=loader, criterion=criterion, device=device
    )
    test_macro_f1 = f1_score(y_true=test_labels, y_pred=test_preds, average="macro")
    print(f"Test loss for {title}: {test_loss:.4f}")
    print(
        classification_report(
            y_true=test_labels, y_pred=test_preds,
            target_names=config.labels, digits=3,
        )
    )

    cm = confusion_matrix(y_true=test_labels, y_pred=test_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.labels, yticklabels=config.labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{title} - Test Confusion Matrix")
    plt.tight_layout()

    safe_title = "_".join(
        "".join(ch if ch.isalnum() else " " for ch in title.lower()).split()
    )
    config.paths.figures_dir.mkdir(parents=True, exist_ok=True)
    cm_path = config.paths.figures_dir / f"confusion_matrix_{safe_title}.png"
    plt.savefig(cm_path, dpi=120, bbox_inches="tight")
    plt.close()

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    with mlflow.start_run(run_name=f"{title}-eval"):
        mlflow.log_metrics({"test_loss": test_loss, "test_macro_f1": test_macro_f1})
        mlflow.log_artifact(str(cm_path))

    return cm_path
