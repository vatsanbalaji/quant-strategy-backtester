import pandas as pd
import pytest

from quantbench.evaluation.walk_forward import make_walk_forward_folds


def _synthetic_prices(start="2010-01-01", end="2024-12-31"):
    dates = pd.bdate_range(start, end)
    df = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                        "Close": 100.0, "Volume": 1000}, index=dates)
    return df


def test_folds_are_non_overlapping_and_forward_only():
    data = _synthetic_prices()
    folds = make_walk_forward_folds(data, train_years=3, test_years=1, step_years=1)
    assert len(folds) > 0
    for fold in folds:
        assert fold.train.index.max() < fold.test.index.min(), \
            "train window must end strictly before test window begins"


def test_folds_roll_forward_in_time():
    data = _synthetic_prices()
    folds = make_walk_forward_folds(data, train_years=3, test_years=1, step_years=1)
    for a, b in zip(folds, folds[1:]):
        assert b.train.index.min() > a.train.index.min()


def test_no_folds_when_data_too_short():
    data = _synthetic_prices(start="2023-01-01", end="2023-06-30")
    folds = make_walk_forward_folds(data, train_years=3, test_years=1)
    assert folds == []
