"""Training mit Zeitreihen-Cross-Validation und Champion/Challenger-Promotion.

Ablauf:
  1. Baselines (saisonal naiv, lineare Regression) und LightGBM per TimeSeriesSplit vergleichen
  2. Bestes Modell auf allen Daten trainieren
  3. Nur speichern ("promoten"), wenn es den aktuellen Champion schlägt
  4. Optional alles in MLflow loggen
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from energy_forecast.config import Settings, settings
from energy_forecast.features.build import build_features, split_xy
from energy_forecast.models.registry import archive_current

log = logging.getLogger(__name__)


@dataclass
class Metrics:
    mae: float
    mape: float


class SeasonalNaive:
    """Baseline: Last von vor einer Woche. Jedes echte Modell muss das schlagen."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaive:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_168h"].to_numpy()


def candidates(seed: int = 42) -> dict[str, Callable[[], object]]:
    return {
        "seasonal_naive": SeasonalNaive,
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "lightgbm": lambda: lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        ),
    }


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        mape=float(mean_absolute_percentage_error(y_true, y_pred)),
    )


def cross_validate(make_model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Metrics:
    """Expanding-Window-CV: Trainiert nur auf Vergangenheit, testet auf Zukunft."""
    folds = [
        evaluate(y.iloc[test], make_model().fit(X.iloc[train], y.iloc[train]).predict(X.iloc[test]))
        for train, test in TimeSeriesSplit(n_splits=n_splits, test_size=24 * 14).split(X)
    ]
    return Metrics(
        mae=float(np.mean([f.mae for f in folds])), mape=float(np.mean([f.mape for f in folds]))
    )


def load_champion_metrics(cfg: Settings) -> Metrics | None:
    if not cfg.metadata_path.exists():
        return None
    return Metrics(**json.loads(cfg.metadata_path.read_text())["metrics"])


def should_promote(challenger: Metrics, champion: Metrics | None, min_improvement: float) -> bool:
    return champion is None or challenger.mae < champion.mae * (1 - min_improvement)


def train(df: pd.DataFrame, cfg: Settings = settings, force: bool = False) -> dict:
    X, y = split_xy(build_features(df, cfg.horizon))
    results = {name: cross_validate(make, X, y) for name, make in candidates().items()}
    for name, m in results.items():
        log.info("%-15s MAE=%8.1f MWh  MAPE=%.2f%%", name, m.mae, m.mape * 100)

    best = min(results, key=lambda n: results[n].mae)
    model = candidates()[best]().fit(X, y)
    champion = load_champion_metrics(cfg)
    promoted = force or should_promote(results[best], champion, cfg.min_improvement)

    now = datetime.now(UTC)
    metadata = {
        "version": now.strftime("%Y%m%dT%H%M%S%f"),
        "model": best,
        "metrics": asdict(results[best]),
        "all_metrics": {n: asdict(m) for n, m in results.items()},
        "features": list(X.columns),
        "trained_at": now.isoformat(),
        "train_rows": len(X),
        "promoted": promoted,
    }

    if promoted:
        cfg.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, cfg.model_path)
        cfg.metadata_path.write_text(json.dumps(metadata, indent=2))
        # Referenzverteilung für Drift-Monitoring
        X.tail(24 * 30).to_parquet(cfg.model_dir / "reference.parquet")
        archive_current(cfg, metadata["version"])
        log.info("Neues Modell promotet: %s (Version %s)", best, metadata["version"])
    else:
        log.info(
            "Challenger (MAE %.1f) schlägt Champion (MAE %.1f) nicht – kein Deployment",
            results[best].mae,
            champion.mae,
        )

    _log_mlflow(cfg, metadata, model)
    return metadata


def _log_mlflow(cfg: Settings, metadata: dict, model) -> None:
    if not cfg.mlflow_uri:
        return
    import mlflow

    mlflow.set_tracking_uri(cfg.mlflow_uri)
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name=metadata["model"]):
        mlflow.log_params({"model": metadata["model"], "horizon": cfg.horizon})
        for name, m in metadata["all_metrics"].items():
            mlflow.log_metrics({f"{name}_mae": m["mae"], f"{name}_mape": m["mape"]})
        mlflow.set_tag("promoted", metadata["promoted"])
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name="energy-forecast" if metadata["promoted"] else None,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Trainiert das Lastprognose-Modell")
    p.add_argument("--synthetic", action="store_true", help="synthetische Daten statt SMARD")
    p.add_argument("--force", action="store_true", help="Promotion erzwingen")
    args = p.parse_args()

    if args.synthetic:
        from energy_forecast.data.synthetic import make_load

        df = make_load()
    else:
        df = pd.read_parquet(settings.raw_path)
    print(json.dumps(train(df, force=args.force)["all_metrics"], indent=2))


if __name__ == "__main__":
    main()
