import numpy as np

from energy_forecast.features.build import build_features


def test_no_nan_and_target_present(df):
    feats = build_features(df)
    assert not feats.isna().any().any()
    assert "load" in feats


def test_lags_do_not_leak_future(df):
    """Lag-Feature zum Zeitpunkt t muss exakt der Last von t-24h entsprechen."""
    feats = build_features(df, horizon=24)
    t = feats.index[100]
    assert np.isclose(feats.loc[t, "lag_24h"], df.loc[t - np.timedelta64(24, "h"), "load"])


def test_christmas_is_holiday(df):
    from energy_forecast.data.synthetic import make_load

    feats = build_features(make_load(days=400, start="2024-01-01"))
    assert feats.loc["2024-12-25 12:00", "is_holiday"] == 1
    assert feats.loc["2024-12-27 12:00", "is_holiday"] == 0
