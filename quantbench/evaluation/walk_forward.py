"""Walk-forward evaluation.

Splits the full price history into successive (train, test) windows that
roll forward in time, e.g.:

  Train 2016-2018 -> Test 2019
  Train 2017-2019 -> Test 2020
  Train 2018-2020 -> Test 2021

This is the core anti-overfitting device in the project: a strategy (rule-
based or AI-assisted) is only ever judged on data that comes strictly
AFTER the window it was configured on. No fold's test period ever
overlaps, or precedes, its own train period.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Fold:
    train: pd.DataFrame
    test: pd.DataFrame
    train_range: tuple[str, str]
    test_range: tuple[str, str]


def make_walk_forward_folds(
    price_data: pd.DataFrame,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
) -> list[Fold]:
    """Generate rolling (train, test) folds over `price_data`'s date index.

    Uses calendar years for simplicity/readability in reports; a fold is
    only included if both its train and test windows are fully covered by
    the available data (no partial, silently-shortened windows).
    """
    if not isinstance(price_data.index, pd.DatetimeIndex):
        raise TypeError("price_data must be indexed by date")

    start_year = price_data.index.min().year
    end_year = price_data.index.max().year

    folds = []
    train_start_year = start_year
    while True:
        train_end_year = train_start_year + train_years
        test_end_year = train_end_year + test_years
        if test_end_year > end_year + 1:
            break

        train_start = f"{train_start_year}-01-01"
        train_end = f"{train_end_year - 1}-12-31"
        test_start = f"{train_end_year}-01-01"
        test_end = f"{test_end_year - 1}-12-31"

        train = price_data.loc[train_start:train_end]
        test = price_data.loc[test_start:test_end]

        if len(train) > 0 and len(test) > 0:
            folds.append(Fold(
                train=train, test=test,
                train_range=(train_start, train_end),
                test_range=(test_start, test_end),
            ))

        train_start_year += step_years

    return folds
