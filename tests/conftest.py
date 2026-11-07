import pytest

from energy_forecast.config import Settings
from energy_forecast.data.synthetic import make_load


@pytest.fixture(scope="session")
def df():
    return make_load(days=120)


@pytest.fixture
def cfg(tmp_path):
    return Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models", mlflow_uri=None)
