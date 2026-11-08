import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from energy_forecast.models.evaluation import cross_validate


def test_cross_validate_rejects_insufficient_rows():
    X = pd.DataFrame({"feature": range(10)})
    y = pd.Series(range(10))

    with pytest.raises(ValueError, match="Not enough samples"):
        cross_validate(DummyRegressor(), X, y, n_splits=2, test_size=5)


def test_cross_validate_averages_temporal_folds():
    X = pd.DataFrame({"feature": range(25)})
    y = pd.Series(range(25))

    metrics = cross_validate(
        lambda: DummyRegressor(strategy="mean"), X, y, n_splits=2, test_size=5
    )

    assert metrics.mae > 0
    assert metrics.mape > 0
