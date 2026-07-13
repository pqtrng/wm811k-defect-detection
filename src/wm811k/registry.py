"""MLflow Model Registry: register, promote, compare, load.

Registration is automatic and cheap;
PROMOTION IS A DELIBERATE MANUAL STEP. `register_checkpoint` creates a new
version but never aliases it to production. Aliasing to @production happens only
via `promote`, run by a human after reading `compare` -- because an aggregate
macro-F1 win can hide a per-class regression, and only a per-class read catches it.

MLflow 3.x has removed model stages entirely; we use the ALIAS API
(set_registered_model_alias / get_model_version_by_alias) exclusively. The
production alias is the single source of truth for "what serving loads".
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
import mlflow
from mlflow.tracking import MlflowClient
from wm811k.config import load_config

REGISTERED_MODEL_NAME = "wm811k-defect-classifier"
PRODUCTION_ALIAS = "production"
# Per-class test F1 is logged by evaluate_and_report under this prefix
# (metric name = f"{PER_CLASS_F1_PREFIX}{label}"). compare reads these back.
PER_CLASS_F1_PREFIX = "test_f1_"
MACRO_F1_METRIC = "test_macro_f1"


def _client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def register_checkpoint(
    model_uri: str,
    tracking_uri: str,
    model_name: str = REGISTERED_MODEL_NAME,
) -> str:
    """Register a logged model as a NEW version. Does NOT alias to production.

    Args:
        model_uri: source of the model artifact, e.g. "runs:/<run_id>/model"
            (the URI mlflow.pytorch.log_model produced) or any valid model URI.
        tracking_uri: MLflow tracking URI (from config.mlflow.tracking_uri).
        model_name: the registered model (the "drawer") to add a version to.

    Returns:
        The new version number as a string (MLflow's ModelVersion.version).
    """
    client = _client(tracking_uri)
    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        pass  # already exists -- adding a version to it, not recreating

    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(
        f"Registered {model_name} version {mv.version} from {model_uri}. "
        f"NOT promoted -- run `registry promote --version {mv.version}` after "
        f"reviewing `registry compare`."
    )
    return mv.version


def promote(
    version: str,
    tracking_uri: str,
    model_name: str = REGISTERED_MODEL_NAME,
) -> None:
    """Alias a version to @production. THE manual gate: run only after a human
    has reviewed `compare` and accepted any per-class trade-offs.
    """
    client = _client(tracking_uri)
    client.set_registered_model_alias(
        name=model_name, alias=PRODUCTION_ALIAS, version=version
    )
    print(f"Promoted {model_name} version {version} to @{PRODUCTION_ALIAS}.")


def load_production(
    tracking_uri: str,
    model_name: str = REGISTERED_MODEL_NAME,
):
    """Load the @production model, ready for inference. Used by T10 serving.

    Returns the reconstructed torch model (mlflow.pytorch flavor). Raises
    MlflowException if no version carries the production alias yet.
    """
    mlflow.set_tracking_uri(tracking_uri)
    uri = f"models:/{model_name}@{PRODUCTION_ALIAS}"
    return mlflow.pytorch.load_model(uri)


@dataclass
class _VersionMetrics:
    version: str
    run_id: str
    macro_f1: float | None
    per_class_f1: dict[str, float]


def _metrics_for_version(
    client: MlflowClient, model_name: str, version: str
) -> _VersionMetrics:
    """Pull the test metrics logged by the run that produced a model version."""
    mv = client.get_model_version(name=model_name, version=version)
    run = client.get_run(mv.run_id)
    metrics = run.data.metrics
    per_class = {
        k[len(PER_CLASS_F1_PREFIX) :]: v
        for k, v in metrics.items()
        if k.startswith(PER_CLASS_F1_PREFIX)
    }
    return _VersionMetrics(
        version=version,
        run_id=mv.run_id,
        macro_f1=metrics.get(MACRO_F1_METRIC),
        per_class_f1=per_class,
    )


def _resolve_production_version(client: MlflowClient, model_name: str) -> str | None:
    try:
        mv = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
        return mv.version
    except mlflow.exceptions.MlflowException:
        return None


def compare(
    candidate_version: str,
    tracking_uri: str,
    against_version: str | None = None,
    model_name: str = REGISTERED_MODEL_NAME,
) -> bool:
    """Print a per-class F1 delta table: candidate vs a baseline version.

    Baseline defaults to the current @production version. If nothing is in
    production yet, the candidate is shown alone (nothing to compare against).

    This is the artifact a human reads before `promote`: it surfaces per-class
    regressions that an aggregate macro-F1 delta would hide. Returns True if a
    comparison table was produced, False if there was no baseline.
    """
    client = _client(tracking_uri)
    cand = _metrics_for_version(client, model_name, candidate_version)

    baseline_version = against_version or _resolve_production_version(
        client, model_name
    )
    if baseline_version is None:
        print(
            f"No @{PRODUCTION_ALIAS} version and no --against given. "
            f"Candidate v{cand.version} metrics only:"
        )
        print(f"  {MACRO_F1_METRIC}: {cand.macro_f1}")
        for label in sorted(cand.per_class_f1):
            print(f"  {PER_CLASS_F1_PREFIX}{label}: {cand.per_class_f1[label]:.3f}")
        return False

    base = _metrics_for_version(client, model_name, baseline_version)

    print(f"\n=== compare: candidate v{cand.version} vs baseline v{base.version} ===")
    if cand.macro_f1 is not None and base.macro_f1 is not None:
        d = cand.macro_f1 - base.macro_f1
        arrow = "▲" if d > 0 else ("▼" if d < 0 else "=")
        print(
            f"macro-F1: {base.macro_f1:.3f} -> {cand.macro_f1:.3f}  ({d:+.3f} {arrow})"
        )

    labels = sorted(set(cand.per_class_f1) | set(base.per_class_f1))
    print(f"\n{'class':<12}{'baseline':>10}{'candidate':>11}{'delta':>9}")
    regressions = []
    for label in labels:
        b = base.per_class_f1.get(label)
        c = cand.per_class_f1.get(label)
        if b is None or c is None:
            print(f"{label:<12}{str(b):>10}{str(c):>11}{'n/a':>9}")
            continue
        d = c - b
        flag = "  <-- REGRESSION" if d < 0 else ""
        if d < 0:
            regressions.append((label, d))
        print(f"{label:<12}{b:>10.3f}{c:>11.3f}{d:>+9.3f}{flag}")

    if regressions:
        worst = min(regressions, key=lambda x: x[1])
        print(
            f"\n{len(regressions)} class(es) regressed; worst: "
            f"{worst[0]} ({worst[1]:+.3f}). Review before promoting."
        )
    else:
        print("\nNo per-class regressions vs baseline.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="MLflow model registry operations.")
    parser.add_argument("--config", default="configs/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_reg = sub.add_parser("register", help="Register a logged model as a new version")
    p_reg.add_argument(
        "--model-uri", required=True, help="Source model URI, e.g. runs:/<run_id>/model"
    )

    p_prom = sub.add_parser(
        "promote", help="Alias a version to @production (manual gate)"
    )
    p_prom.add_argument("--version", required=True)

    p_cmp = sub.add_parser(
        "compare", help="Per-class F1 delta vs @production (or --against)"
    )
    p_cmp.add_argument("--candidate", required=True, help="Candidate version to review")
    p_cmp.add_argument(
        "--against", default=None, help="Baseline version (default: @production)"
    )

    args = parser.parse_args()
    config = load_config(args.config)
    tracking_uri = config.mlflow.tracking_uri

    if args.command == "register":
        register_checkpoint(args.model_uri, tracking_uri)
    elif args.command == "promote":
        promote(args.version, tracking_uri)
    elif args.command == "compare":
        compare(args.candidate, tracking_uri, against_version=args.against)


if __name__ == "__main__":
    main()
