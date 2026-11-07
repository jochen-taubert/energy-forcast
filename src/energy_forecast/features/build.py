"""Feature Engineering für Day-Ahead-Prognosen.

Wichtig gegen Data Leakage: Alle Lag-Features liegen mindestens `horizon` Stunden
in der Vergangenheit. Zum Prognosezeitpunkt sind sie also tatsächlich bekannt.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

TARGET = "load"


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert("Europe/Berlin")
    de_holidays = holidays.country_holidays(
        "DE", years=range(local.year.min(), local.year.max() + 1)
    )
    return pd.DataFrame(
        {
            "hour": local.hour,
            "dow": local.dayofweek,
            "month": local.month,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "is_holiday": np.isin(local.date, list(de_holidays)).astype(int),
            # zyklische Kodierung, damit 23 Uhr und 0 Uhr "nah" sind
            "hour_sin": np.sin(2 * np.pi * local.hour / 24),
            "hour_cos": np.cos(2 * np.pi * local.hour / 24),
        },
        index=index,
    )


def lag_features(load: pd.Series, horizon: int = 24) -> pd.DataFrame:
    shifted = load.shift(horizon)
    return pd.DataFrame(
        {
            f"lag_{horizon}h": shifted,
            "lag_168h": load.shift(168),  # gleiche Stunde letzte Woche
            "roll_mean_24h": shifted.rolling(24).mean(),
            "roll_std_24h": shifted.rolling(24).std(),
            "roll_mean_168h": shifted.rolling(168).mean(),
        }
    )


def build_features(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Erzeugt die Feature-Matrix. Zeilen ohne vollständige Historie werden verworfen."""
    feats = pd.concat([calendar_features(df.index), lag_features(df[TARGET], horizon)], axis=1)
    if "temp" in df:
        feats["temp"] = df["temp"]
        feats["heating_degree"] = (15 - df["temp"]).clip(lower=0)
    feats[TARGET] = df[TARGET]
    return feats.dropna()


def split_xy(feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return feats.drop(columns=TARGET), feats[TARGET]
