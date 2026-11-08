"""Factories for the forecasting models evaluated during training."""

from __future__ import annotations

from collections.abc import Callable

import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from energy_forecast.models.evaluation import Regressor, SeasonalNaive


def candidates(seed: int = 42) -> dict[str, Callable[[], Regressor]]:
    """Return a fresh estimator factory for each candidate model."""
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
