import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(df, cfg, monkeypatch):
    from energy_forecast.models.train import train
    from energy_forecast.serving import app as app_module

    train(df, cfg)
    monkeypatch.setattr(app_module, "settings", cfg)
    with TestClient(app_module.app) as c:
        yield c


def history(df, hours=240):
    tail = df.tail(hours).reset_index()
    return [{"ts": r.ts.isoformat(), "load": r.load, "temp": r.temp} for r in tail.itertuples()]


def test_forecast_returns_24_hours(client, df):
    r = client.post("/forecast", json={"history": history(df)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["forecast"]) == 24
    assert all(30_000 < p["load"] < 80_000 for p in body["forecast"])


def test_short_history_rejected(client, df):
    assert client.post("/forecast", json={"history": history(df, 50)}).status_code == 422


def test_health_and_metrics(client):
    assert client.get("/health").json()["model_loaded"]
    assert "ef_predictions_total" in client.get("/metrics").text
