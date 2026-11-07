"""Freigabe-API für Menschen: offene Vorschläge ansehen, freigeben, ablehnen.

Läuft getrennt vom Prognose-Service, damit dieser schlank bleibt.
Start: uvicorn energy_forecast.agent.api:app --port 8001

In Produktion gehört davor eine Authentifizierung (z. B. Keycloak/OIDC); `by` käme
dann aus dem Token statt aus dem Request.
"""

from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from energy_forecast.agent.cli import stores
from energy_forecast.agent.tools import Workspace
from energy_forecast.config import settings

app = FastAPI(title="Energy Forecast – Agent Approvals")


class DecisionIn(BaseModel):
    by: str
    note: str = ""


@app.get("/proposals")
def proposals(status: str | None = "pending") -> dict:
    return stores(settings)[1].list(status)


@app.post("/proposals/{run_id}/approve")
def approve(run_id: str, d: DecisionIn) -> dict:
    audit, queue = stores(settings)
    ws = Workspace(settings, pd.read_parquet(settings.raw_path), audit=audit)
    try:
        return queue.approve(run_id, d.by, ws, d.note)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.post("/proposals/{run_id}/reject")
def reject(run_id: str, d: DecisionIn) -> dict:
    try:
        stores(settings)[1].reject(run_id, d.by, d.note)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e
    return {"status": "rejected"}


@app.get("/audit/{run_id}")
def audit_trail(run_id: str) -> list[dict]:
    return [e for e in stores(settings)[0].read() if e["run_id"] == run_id]
