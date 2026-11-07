"""FastAPI-Service: Day-Ahead-Prognose aus der jüngsten Lasthistorie.

Der Client schickt die letzten >= 8 Tage stündliche Last, der Service liefert
die nächsten 24 Stunden. So bleibt der Service zustandslos und skaliert horizontal.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from energy_forecast.config import settings
from energy_forecast.features.build import build_features, split_xy

PREDICTIONS = Counter("ef_predictions_total", "Anzahl Prognose-Requests")
LATENCY = Histogram("ef_predict_seconds", "Latenz der Prognose")
MIN_HISTORY = 168 + 24 + 24

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(settings.model_path)
    state["meta"] = json.loads(settings.metadata_path.read_text())
    yield
    state.clear()


app = FastAPI(title="Energy Forecast", version="0.1.0", lifespan=lifespan)


class Observation(BaseModel):
    ts: datetime
    load: float = Field(gt=0, description="Netzlast in MWh")
    temp: float | None = None


class ForecastRequest(BaseModel):
    history: list[Observation] = Field(min_length=MIN_HISTORY)


class ForecastPoint(BaseModel):
    ts: datetime
    load: float


class ForecastResponse(BaseModel):
    model: str
    trained_at: str
    forecast: list[ForecastPoint]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model")
def model_info() -> dict:
    return {k: v for k, v in state["meta"].items() if k != "all_metrics"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    start = time.perf_counter()
    meta, horizon = state["meta"], settings.horizon

    hist = pd.DataFrame([o.model_dump() for o in req.history])
    hist["ts"] = pd.to_datetime(hist["ts"], utc=True)
    hist = hist.set_index("ts").sort_index().asfreq("h")
    if hist["load"].isna().any():
        raise HTTPException(422, "Historie muss lückenlos stündlich sein")
    if "temp" in meta["features"]:
        if hist["temp"].isna().any():
            raise HTTPException(422, "Modell benötigt Temperaturwerte")
    else:
        hist = hist.drop(columns="temp")

    # Zukünftige Zeitstempel anhängen; Lags (>= horizon) sind aus der Historie bekannt
    future_idx = pd.date_range(hist.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    future = pd.DataFrame(index=future_idx, columns=hist.columns, dtype=float)
    future["load"] = 1.0  # Platzhalter, wird nicht als Feature genutzt
    if "temp" in hist:
        future["temp"] = hist["temp"].iloc[-horizon:].to_numpy()  # naive Wetterannahme
    X, _ = split_xy(build_features(pd.concat([hist, future]), horizon))
    X = X.loc[future_idx, meta["features"]]

    preds = state["model"].predict(X)
    PREDICTIONS.inc()
    LATENCY.observe(time.perf_counter() - start)
    return ForecastResponse(
        model=meta["model"],
        trained_at=meta["trained_at"],
        forecast=[
            ForecastPoint(ts=t, load=float(p)) for t, p in zip(future_idx, preds, strict=True)
        ],
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
