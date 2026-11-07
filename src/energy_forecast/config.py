"""Zentrale Konfiguration, überschreibbar per Umgebungsvariablen."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("EF_DATA_DIR", "data")))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("EF_MODEL_DIR", "models_store")))
    mlflow_uri: str | None = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI"))
    experiment: str = "energy-forecast"
    # Prognosehorizont in Stunden (Day-Ahead)
    horizon: int = 24
    # Challenger muss den Champion um mindestens diesen Anteil schlagen
    min_improvement: float = 0.01

    @property
    def raw_path(self) -> Path:
        return self.data_dir / "load.parquet"

    @property
    def model_path(self) -> Path:
        return self.model_dir / "model.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / "metadata.json"


settings = Settings()
