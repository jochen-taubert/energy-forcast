"""Training orchestration and artifact promotion."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import joblib
import pandas as pd

from energy_forecast.config import Settings, settings
from energy_forecast.features.build import build_features, split_xy
from energy_forecast.models.candidates import candidates
from energy_forecast.models.evaluation import Metrics, cross_validate, should_promote
from energy_forecast.models.registry import archive_current

log = logging.getLogger(__name__)


def load_champion_metrics(cfg: Settings) -> Metrics | None:
    """Load the active model's validation metrics, if a champion exists."""
    if not cfg.metadata_path.exists():
        return None
    metadata = json.loads(cfg.metadata_path.read_text(encoding="utf-8"))
    return Metrics(**metadata["metrics"])


def train(df: pd.DataFrame, cfg: Settings = settings, force: bool = False) -> dict[str, Any]:
    """Select, fit, evaluate, and optionally promote a forecasting model."""
    X, y = split_xy(build_features(df, cfg.horizon))
    model_factories = candidates()
    results = {
        name: cross_validate(make_model, X, y) for name, make_model in model_factories.items()
    }
    for name, metrics in results.items():
        log.info("%-15s MAE=%8.1f MWh  MAPE=%.2f%%", name, metrics.mae, metrics.mape * 100)

    best_name = min(results, key=lambda name: results[name].mae)
    model = model_factories[best_name]().fit(X, y)
    champion = load_champion_metrics(cfg)
    promoted = force or should_promote(results[best_name], champion, cfg.min_improvement)

    trained_at = datetime.now(UTC)
    metadata: dict[str, Any] = {
        "version": trained_at.strftime("%Y%m%dT%H%M%S%f"),
        "model": best_name,
        "metrics": asdict(results[best_name]),
        "all_metrics": {name: asdict(metrics) for name, metrics in results.items()},
        "features": list(X.columns),
        "trained_at": trained_at.isoformat(),
        "train_rows": len(X),
        "promoted": promoted,
    }

    if promoted:
        cfg.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, cfg.model_path)
        cfg.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        # Save the training reference distribution for drift monitoring.
        X.tail(24 * 30).to_parquet(cfg.model_dir / "reference.parquet")
        archive_current(cfg, metadata["version"])
        log.info("Promoted %s (version %s)", best_name, metadata["version"])
    else:
        assert champion is not None  # should_promote returns True if no champion exists.
        log.info(
            "Challenger (MAE %.1f) did not beat champion (MAE %.1f); not deploying",
            results[best_name].mae,
            champion.mae,
        )

    _log_mlflow(cfg, metadata, model)
    return metadata


def _log_mlflow(cfg: Settings, metadata: dict[str, Any], model: Any) -> None:
    """Log a training run only when MLflow tracking is configured."""
    if not cfg.mlflow_uri:
        return

    import mlflow

    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=metadata["model"]):
        mlflow.log_params({"model": metadata["model"], "horizon": cfg.horizon})
        for name, metrics in metadata["all_metrics"].items():
            mlflow.log_metrics(
                {f"{name}_mae": metrics["mae"], f"{name}_mape": metrics["mape"]}
            )
        mlflow.set_tag("promoted", metadata["promoted"])
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="energy-forecast" if metadata["promoted"] else None,
        )
