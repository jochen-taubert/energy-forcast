"""Einfache dateibasierte Modell-Registry.

Jede Promotion landet unveränderlich unter versions/<version>/. Das aktive Modell
(der Champion) liegt zusätzlich im Wurzelverzeichnis, dort liest es der API-Service.
Ein Rollback kopiert eine ältere Version zurück ins Wurzelverzeichnis.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from energy_forecast.config import Settings

ARTIFACTS = ("model.joblib", "metadata.json", "reference.parquet")


def versions_dir(cfg: Settings) -> Path:
    return cfg.model_dir / "versions"


def archive_current(cfg: Settings, version: str) -> Path:
    target = versions_dir(cfg) / version
    target.mkdir(parents=True, exist_ok=False)
    for name in ARTIFACTS:
        if (src := cfg.model_dir / name).exists():
            shutil.copy2(src, target / name)
    return target


def list_versions(cfg: Settings) -> list[dict]:
    if not versions_dir(cfg).exists():
        return []
    current = current_version(cfg)
    out = []
    for d in sorted(versions_dir(cfg).iterdir()):
        meta = json.loads((d / "metadata.json").read_text())
        out.append(
            {
                "version": d.name,
                "model": meta["model"],
                "cv_mae": round(meta["metrics"]["mae"], 1),
                "trained_at": meta["trained_at"],
                "is_current": d.name == current,
            }
        )
    return out


def current_version(cfg: Settings) -> str | None:
    if not cfg.metadata_path.exists():
        return None
    return json.loads(cfg.metadata_path.read_text()).get("version")


def restore(cfg: Settings, version: str) -> None:
    source = versions_dir(cfg) / version
    if not source.is_dir():
        raise ValueError(f"Unbekannte Version: {version}")
    for name in ARTIFACTS:
        if (src := source / name).exists():
            shutil.copy2(src, cfg.model_dir / name)
