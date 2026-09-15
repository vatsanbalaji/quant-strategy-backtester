import numpy as np
import pandas as pd
import pytest

from quantbench.evaluation.metrics import (
    annualized_return, annualized_volatility, max_drawdown,
    profit_factor, sharpe_ratio, sortino_ratio, total_return, win_rate,
)


def test_total_return_simple():
    returns = pd.Series([0.10, 0.10])  # 10% then 10% -> 21% total
    assert total_return(returns) == pytest.approx(0.21, abs=1e-9)


def test_total_return_flat():
    returns = pd.Series([0.0, 0.0, 0.0])
    assert total_return(returns) == pytest.approx(0.0)


def test_max_drawdown_detects_known_drop():
    # equity goes 100 -> 150 -> 75 -> 90 : drawdown from 150 to 75 = -50%
    returns = pd.Series([0.5, -0.5, 0.2])
    dd = max_drawdown(returns)
    assert dd == pytest.approx(-0.5, abs=1e-6)


def test_sharpe_ratio_zero_vol_is_zero_not_inf():
    returns = pd.Series([0.001] * 100)  # constant returns, zero std
    assert sharpe_ratio(returns) == 0.0


def test_sharpe_ratio_positive_for_upward_trend():
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.001, 0.005, 500))
    assert sharpe_ratio(returns) > 0


def test_win_rate_bounds():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
    wr = win_rate(returns)
    assert 0.0 <= wr <= 1.0
    assert wr == pytest.approx(0.5)  # 2 wins, 2 losses, one zero excluded


def test_profit_factor_no_losses_is_inf():
    returns = pd.Series([0.01, 0.02, 0.0])
    assert profit_factor(returns) == float("inf")


def test_annualized_return_matches_total_for_one_year():
    returns = pd.Series([0.0004] * 252)
    ann = annualized_return(returns)
    tot = total_return(returns)
    assert ann == pytest.approx(tot, abs=1e-6)
