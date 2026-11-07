import numpy as np
import pandas as pd

from energy_forecast.monitoring.drift import drift_report, psi


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi(ref, pd.Series(rng.normal(0, 1, 5000))) < 0.1
    assert psi(ref, pd.Series(rng.normal(2, 1, 5000))) > 0.25


def test_report_flags_drifted_column():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(3, 1, 2000)})
    report = drift_report(ref, cur)
    assert report.loc["b", "drift"] and not report.loc["a", "drift"]
