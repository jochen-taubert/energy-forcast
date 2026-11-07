from energy_forecast.models.train import Metrics, should_promote, train


def test_model_beats_naive_baseline(df, cfg):
    meta = train(df, cfg)
    m = meta["all_metrics"]
    assert m["lightgbm"]["mae"] < m["seasonal_naive"]["mae"]
    assert meta["promoted"] and cfg.model_path.exists()


def test_champion_is_kept_when_challenger_not_better(df, cfg):
    train(df, cfg)
    assert train(df, cfg)["promoted"] is False


def test_promotion_rule():
    champ = Metrics(mae=1000, mape=0.02)
    assert should_promote(Metrics(mae=980, mape=0.02), champ, 0.01)
    assert not should_promote(Metrics(mae=995, mape=0.02), champ, 0.01)
    assert should_promote(Metrics(mae=5000, mape=0.1), None, 0.01)
