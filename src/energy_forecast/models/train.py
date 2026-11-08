"""Command-line entrypoint for model training.

The training implementation lives in :mod:`energy_forecast.models.training`.
This module re-exports the former public helpers to keep existing imports working.
"""

from __future__ import annotations

import argparse
import json
import logging

import pandas as pd

from energy_forecast.config import settings
from energy_forecast.models.candidates import candidates
from energy_forecast.models.evaluation import (
    Metrics,
    SeasonalNaive,
    cross_validate,
    evaluate,
    should_promote,
)
from energy_forecast.models.training import load_champion_metrics, train

__all__ = [
    "Metrics",
    "SeasonalNaive",
    "candidates",
    "cross_validate",
    "evaluate",
    "load_champion_metrics",
    "main",
    "should_promote",
    "train",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train the energy demand forecasting model")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--force", action="store_true", help="Force model promotion")
    args = parser.parse_args()

    if args.synthetic:
        from energy_forecast.data.synthetic import make_load

        df = make_load()
    else:
        df = pd.read_parquet(settings.raw_path)
    print(json.dumps(train(df, force=args.force)["all_metrics"], indent=2))


if __name__ == "__main__":
    main()
