"""Tägliche Pipeline mit Prefect: Daten holen -> prüfen -> Drift messen -> trainieren.

Lokal starten:   python -m energy_forecast.pipelines.flow
Als Deployment:  prefect deploy (Schedule z. B. täglich 06:00 Europe/Berlin)
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from prefect import flow, get_run_logger, task

from energy_forecast.config import settings
from energy_forecast.data.smard import fetch_load, fetch_temperature
from energy_forecast.features.build import build_features, split_xy
from energy_forecast.models.train import train
from energy_forecast.monitoring.drift import drift_report


@task(retries=3, retry_delay_seconds=60)
def ingest(weeks: int) -> pd.DataFrame:
    load = fetch_load(weeks=weeks)
    temp = fetch_temperature(load.index.min().date(), load.index.max().date())
    df = load.join(temp, how="left")
    df["temp"] = df["temp"].interpolate(limit=3)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.raw_path)
    return df


@task
def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Datenqualitäts-Gates: Pipeline bricht ab, statt mit schlechten Daten zu trainieren."""
    df = df.asfreq("h")
    gaps = df["load"].isna().mean()
    assert gaps < 0.01, f"Zu viele Lücken in der Last: {gaps:.1%}"
    assert df["load"].between(20_000, 100_000).all(), "Lastwerte außerhalb plausibler Grenzen"
    assert df.index.max() > pd.Timestamp.now(tz="UTC") - timedelta(days=3), "Daten sind veraltet"
    return df.interpolate(limit=2).dropna(subset=["load"])


@task
def check_drift(df: pd.DataFrame) -> bool:
    ref_path = settings.model_dir / "reference.parquet"
    if not ref_path.exists():
        return False
    current, _ = split_xy(build_features(df.tail(24 * 37), settings.horizon))
    report = drift_report(pd.read_parquet(ref_path), current.tail(24 * 7))
    get_run_logger().info("Drift-Report:\n%s", report.head(10).to_string())
    return bool(report["drift"].any())


@task
def retrain(df: pd.DataFrame) -> dict:
    return train(df)


@flow(name="energy-forecast-daily")
def daily(weeks: int = 156) -> dict:
    df = validate(ingest(weeks))
    drifted = check_drift(df)
    get_run_logger().info("Drift erkannt: %s", drifted)
    # Täglich neu trainieren; die Champion/Challenger-Logik entscheidet über das Deployment
    return retrain(df)


if __name__ == "__main__":
    daily()
