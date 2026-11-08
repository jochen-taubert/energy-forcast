"""Model evaluation and promotion rules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class Metrics:
    """Regression metrics measured on held-out observations."""

    mae: float
    mape: float


class Regressor(Protocol):
    """Minimal estimator interface required by the evaluation helpers."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Regressor: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


ModelFactory = Callable[[], Regressor]


class SeasonalNaive:
    """Last-week baseline. Every learned model should outperform this baseline."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaive:
        del y  # This baseline has no learned parameters.
        if "lag_168h" not in X:
            raise ValueError("SeasonalNaive requires the 'lag_168h' feature")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "lag_168h" not in X:
            raise ValueError("SeasonalNaive requires the 'lag_168h' feature")
        return X["lag_168h"].to_numpy()


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Metrics:
    """Calculate MAE and MAPE for a set of predictions."""
    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        mape=float(mean_absolute_percentage_error(y_true, y_pred)),
    )


def cross_validate(
    make_model: ModelFactory,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: int = 24 * 14,
) -> Metrics:
    """Evaluate with expanding-window folds, always training on the past."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if test_size < 1:
        raise ValueError("test_size must be positive")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")

    minimum_rows = n_splits * test_size + 1
    if len(X) < minimum_rows:
        raise ValueError(
            f"Not enough samples for {n_splits} folds of {test_size} rows; "
            f"need at least {minimum_rows}, got {len(X)}"
        )

    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    fold_metrics: list[Metrics] = []
    for train_indices, test_indices in splitter.split(X):
        model = make_model()
        model.fit(X.iloc[train_indices], y.iloc[train_indices])
        predictions = model.predict(X.iloc[test_indices])
        fold_metrics.append(evaluate(y.iloc[test_indices], predictions))

    return Metrics(
        mae=float(np.mean([metrics.mae for metrics in fold_metrics])),
        mape=float(np.mean([metrics.mape for metrics in fold_metrics])),
    )


def should_promote(
    challenger: Metrics,
    champion: Metrics | None,
    min_improvement: float,
) -> bool:
    """Return whether the challenger clears the configured MAE improvement gate."""
    if not 0 <= min_improvement < 1:
        raise ValueError("min_improvement must be in the range [0, 1)")
    return champion is None or challenger.mae < champion.mae * (1 - min_improvement)
