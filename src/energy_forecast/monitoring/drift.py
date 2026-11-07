"""Daten-Drift per Population Stability Index (PSI).

Bewusst selbst implementiert, damit das Prinzip sichtbar ist.
Faustregel: PSI < 0.1 stabil, 0.1–0.25 beobachten, > 0.25 Retraining.
Für Reports im Projekt-Ausbau: Evidently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_ALERT = 0.25


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference.dropna(), np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (fast) konstantes Feature
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-6, None), np.clip(cur, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    cols = reference.select_dtypes("number").columns.intersection(current.columns)
    report = pd.DataFrame({"psi": {c: psi(reference[c], current[c]) for c in cols}})
    report["drift"] = report["psi"] > PSI_ALERT
    return report.sort_values("psi", ascending=False)
