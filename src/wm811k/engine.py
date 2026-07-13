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
from wm811k.registry import register_checkpoint


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_experiment(config: Config) -> None:
    """Set tracking URI + experiment, pinning artifact_location to <root>/mlruns.

    MLflow freezes an experiment's artifact_location at CREATION time, inferring
    it from the current working directory unless told otherwise. Running the first
    command from notebooks/ once permanently pins artifacts to notebooks/mlruns/.
    We pin it explicitly to config.paths.mlruns_dir so artifact location is a
    declared constant, independent of cwd -- required for reproducibility and for
    remote roots.
    """
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    name = config.mlflow.experiment_name
    if mlflow.get_experiment_by_name(name) is None:
        config.paths.mlruns_dir.mkdir(parents=True, exist_ok=True)
        artifact_location = (config.paths.mlruns_dir / name).as_uri()
        mlflow.create_experiment(name, artifact_location=artifact_location)
    mlflow.set_experiment(name)


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
) -> tuple[Path, str]:
    """Train with MLflow logging, LR scheduling, and early stopping.

    Returns (best_checkpoint_path, mlflow_run_id). The run_id is returned so the
    caller can resume the SAME run for test-metric logging, model logging, and
    registration -- keeping a model and all its metrics co-located in one run
    (required for `registry compare`, and the basis for preemption-resume in T11).
    Checkpoint selection uses best val_loss, never val_macro_f1.

    Ported from the notebook's run_experiment; hyperparameters now come from
    config instead of module globals.
    """
    device = device or _default_device()
    epochs = config.training.epochs
    lr = config.training.lr
    es_patience = config.training.early_stopping_patience

    _ensure_experiment(config)

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
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
            val_loss, val_preds, val_labels = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            val_macro_f1 = f1_score(
                y_true=val_labels, y_pred=val_preds, average="macro"
            )

            scheduler.step(metrics=val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_macro_f1": val_macro_f1,
                    "lr": current_lr,
                },
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
    return best_path, run.info.run_id


def evaluate_and_report(
    best_path: str | Path,
    model: nn.Module,
    loader: DataLoader,
    title: str,
    config: Config,
    device: torch.device | None = None,
    run_id: str | None = None,
) -> Path:
    """Load best checkpoint, run per-class report + confusion matrix on loader.

    Never reports aggregate accuracy alone: per-class precision/recall/F1 and
    the confusion matrix are the point (class imbalance hides regressions).

    Logs test_loss, test_macro_f1, and per-class test_f1_<label> to MLflow, plus
    the confusion matrix under config.paths.figures_dir. If run_id is given, logs
    into THAT run (co-locating test metrics with the model that fit() trained);
    otherwise opens a standalone '<title>-eval' run (backward-compatible path for
    the evaluate.py CLI). The per-class test_f1_<label> metrics are what
    `registry compare` reads to surface per-class regressions before promotion.
    """
    device = device or _default_device()
    criterion = nn.CrossEntropyLoss()

    model = model.to(device=device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_preds, test_labels = evaluate(
        model=model, loader=loader, criterion=criterion, device=device
    )
    test_macro_f1 = f1_score(y_true=test_labels, y_pred=test_preds, average="macro")
    # Per-class F1 for MLflow (registry compare reads test_f1_<label> back).
    # Keep label names verbatim so compare can round-trip the prefix strip.
    report_dict = classification_report(
        y_true=test_labels,
        y_pred=test_preds,
        target_names=config.labels,
        digits=3,
        output_dict=True,
    )
    per_class_f1 = {
        f"test_f1_{label}": report_dict[label]["f1-score"] for label in config.labels
    }

    print(f"Test loss for {title}: {test_loss:.4f}")
    print(
        classification_report(
            y_true=test_labels,
            y_pred=test_preds,
            target_names=config.labels,
            digits=3,
        )
    )

    cm = confusion_matrix(y_true=test_labels, y_pred=test_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.labels,
        yticklabels=config.labels,
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

    _ensure_experiment(config)

    metrics = {"test_loss": test_loss, "test_macro_f1": test_macro_f1, **per_class_f1}
    if run_id is not None:
        # Resume the fit() run: test metrics live with the model they describe.
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(cm_path))
    else:
        with mlflow.start_run(run_name=f"{title}-eval"):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(cm_path))

    return cm_path


def log_and_register(
    run_id: str,
    model: nn.Module,
    best_path: str | Path,
    config: Config,
    register: bool = False,
    device: torch.device | None = None,
) -> str | None:
    """Log the BEST model into the fit() run, optionally register a new version.

    Resumes run_id (the run fit() owns) so the model artifact lives with the
    metrics that describe it. Reloads the best checkpoint into `model` BEFORE
    logging -- after training, `model` holds last-epoch weights, which may be
    worse than the val_loss-selected best; logging the raw object would ship the
    wrong weights.

    log_model uses a batch=2 float32 input_example (2,1,64,64): MLflow 3.x runs
    torch.export, which raises a dynamic-shape ConstraintViolationError on a
    batch=1 example. Inference at any batch size still works.

    If register=True, creates a NEW registry version from this run's model. It is
    NEVER aliased to @production here -- promotion is a separate manual step
    (`registry promote`) taken after a human reviews `registry compare`.

    Returns the new version string if registered, else None.
    """
    device = device or _default_device()
    model = model.to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    input_example = torch.zeros(2, 1, 64, 64, dtype=torch.float32).numpy()

    _ensure_experiment(config)

    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.pytorch.log_model(
            model, name="model", input_example=input_example
        )
        model_uri = model_info.model_uri

    if not register:
        print(f"Logged model to run {run_id} (not registered).")
        return None

    version = register_checkpoint(model_uri, config.mlflow.tracking_uri)
    return version
