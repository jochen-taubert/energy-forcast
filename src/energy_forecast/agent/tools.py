"""Lesende Werkzeuge des Agenten. Keines davon verändert etwas.

Jedes Werkzeug ist eine reine Funktion (Workspace -> dict), dadurch einzeln testbar.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from energy_forecast.agent.audit import AuditLog
from energy_forecast.config import Settings
from energy_forecast.features.build import build_features, split_xy
from energy_forecast.models.registry import list_versions
from energy_forecast.monitoring.drift import drift_report


@dataclass
class Workspace:
    """Alles, worauf der Agent lesend zugreifen darf."""

    cfg: Settings
    data: pd.DataFrame
    now: pd.Timestamp = field(default=None)  # type: ignore[assignment]
    audit: AuditLog | None = None

    def __post_init__(self) -> None:
        if self.now is None:
            self.now = self.data.index.max() + pd.Timedelta(hours=1)

    def metadata(self) -> dict:
        return json.loads(self.cfg.metadata_path.read_text())


def get_model_info(ws: Workspace) -> dict:
    meta = ws.metadata()
    return {
        "version": meta.get("version"),
        "model": meta["model"],
        "cv_mae": round(meta["metrics"]["mae"], 1),
        "cv_mape_pct": round(meta["metrics"]["mape"] * 100, 2),
        "trained_at": meta["trained_at"],
    }


def get_recent_performance(ws: Workspace, days: int = 7) -> dict:
    """Vergleicht den Fehler des aktiven Modells auf den letzten Tagen mit seinem CV-Fehler."""
    meta = ws.metadata()
    X, y = split_xy(build_features(ws.data, ws.cfg.horizon))
    X, y = X.tail(days * 24), y.tail(days * 24)
    if len(X) == 0:
        return {"error": "Keine auswertbaren Daten im Zeitraum"}
    pred = joblib.load(ws.cfg.model_path).predict(X[meta["features"]])
    mae = float(np.mean(np.abs(y.to_numpy() - pred)))
    ratio = mae / meta["metrics"]["mae"]
    return {
        "days": days,
        "recent_mae": round(mae, 1),
        "cv_mae": round(meta["metrics"]["mae"], 1),
        "ratio": round(ratio, 2),
        "bias": round(float(np.mean(y.to_numpy() - pred)), 1),
        "assessment": "degraded" if ratio > 1.5 else "ok",
    }


# Überwacht werden die Eingangsgrößen selbst. Gleitende Wochenmittel schwanken innerhalb
# einer Woche kaum, dort schlägt PSI schon bei normalem Saisonverlauf an.
DRIFT_FEATURES = ["lag_24h", "lag_168h", "temp", "roll_std_24h"]


def get_drift_report(ws: Workspace, days: int = 7) -> dict:
    ref = pd.read_parquet(ws.cfg.model_dir / "reference.parquet")
    X, _ = split_xy(build_features(ws.data, ws.cfg.horizon))
    cols = [c for c in DRIFT_FEATURES if c in X]
    report = drift_report(ref[cols], X[cols].tail(days * 24))
    top = report.head(5)
    return {
        "drifted_features": report.index[report["drift"]].tolist(),
        "top_psi": {k: round(v, 3) for k, v in top["psi"].items()},
        "threshold": 0.25,
    }


def get_data_quality(ws: Workspace) -> dict:
    df = ws.data
    full = df.asfreq("h")
    age_h = (ws.now - df.index.max()) / pd.Timedelta(hours=1)
    gap_ratio = float(full["load"].isna().tail(24 * 14).mean())
    out_of_range = int((~df["load"].between(20_000, 100_000)).sum())
    issues = []
    if age_h > 48:
        issues.append(f"Daten veraltet: letzter Wert vor {age_h:.0f} h")
    if gap_ratio > 0.01:
        issues.append(f"Lücken in den letzten 14 Tagen: {gap_ratio:.1%}")
    if out_of_range:
        issues.append(f"{out_of_range} unplausible Lastwerte")
    return {
        "ok": not issues,
        "issues": issues,
        "last_timestamp": df.index.max().isoformat(),
        "age_hours": round(age_h, 1),
    }


def get_model_versions(ws: Workspace) -> dict:
    return {"versions": list_versions(ws.cfg)}


def get_action_history(ws: Workspace, hours: int = 48) -> dict:
    """Ausgeführte Aktionen der letzten Stunden (aus dem Audit-Log)."""
    if ws.audit is None:
        return {"executed": []}
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    executed = [
        {"ts": e["ts"], "action": e["action"]}
        for e in ws.audit.read()
        if e["event"] == "executed" and pd.Timestamp(e["ts"]) >= cutoff
    ]
    return {"hours": hours, "executed": executed, "retrain_limit_per_24h": 1}


READ_TOOLS: dict[str, Callable[..., dict]] = {
    "get_model_info": get_model_info,
    "get_recent_performance": get_recent_performance,
    "get_drift_report": get_drift_report,
    "get_data_quality": get_data_quality,
    "get_model_versions": get_model_versions,
    "get_action_history": get_action_history,
}

_NO_ARGS = {"type": "object", "properties": {}}
_DAYS = {
    "type": "object",
    "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 30}},
}

TOOL_SPECS = [
    {
        "name": "get_model_info",
        "description": "Aktives Modell: Version, Typ, CV-Fehler.",
        "input_schema": _NO_ARGS,
    },
    {
        "name": "get_recent_performance",
        "description": "Fehler des aktiven Modells auf den letzten Tagen "
        "im Vergleich zum CV-Fehler. "
        "ratio > 1.5 bedeutet deutliche Verschlechterung.",
        "input_schema": _DAYS,
    },
    {
        "name": "get_drift_report",
        "description": "Daten-Drift (PSI) der Features gegenüber den Trainingsdaten.",
        "input_schema": _DAYS,
    },
    {
        "name": "get_data_quality",
        "description": "Aktualität, Lücken und Plausibilität der Eingangsdaten.",
        "input_schema": _NO_ARGS,
    },
    {
        "name": "get_model_versions",
        "description": "Alle gespeicherten Modellversionen mit CV-Fehler, aktive markiert.",
        "input_schema": _NO_ARGS,
    },
    {
        "name": "get_action_history",
        "description": "Bereits ausgeführte Aktionen (z. B. Retrainings) der letzten Stunden.",
        "input_schema": {
            "type": "object",
            "properties": {"hours": {"type": "integer", "minimum": 1, "maximum": 168}},
        },
    },
]
