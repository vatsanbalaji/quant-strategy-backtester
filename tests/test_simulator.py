import numpy as np
import pandas as pd
import pytest

from quantbench.portfolio.simulator import run_backtest
from quantbench.strategies.rule_based import BuyAndHold


def _flat_price_data(n=50, price=100.0):
    dates = pd.bdate_range("2021-01-01", periods=n)
    return pd.DataFrame({"Open": price, "High": price, "Low": price,
                          "Close": price, "Volume": 1000}, index=dates)


def test_flat_prices_buy_and_hold_only_pays_entry_cost():
    data = _flat_price_data()
    result = run_backtest(BuyAndHold(), data, initial_capital=100_000, transaction_cost_bps=10)
    # Only one trade should occur (the initial entry); flat prices afterward
    # produce zero further rebalancing for buy-and-hold.
    assert len(result.trades) == 1
    assert result.total_transaction_costs > 0
    # Equity should be initial capital minus the one-time transaction cost,
    # unchanged thereafter since price never moves.
    assert result.equity_curve.iloc[-1] == pytest.approx(result.equity_curve.iloc[1], rel=1e-6)


def test_transaction_costs_reduce_equity_vs_zero_cost():
    data = _flat_price_data()
    cheap = run_backtest(BuyAndHold(), data, transaction_cost_bps=0)
    costly = run_backtest(BuyAndHold(), data, transaction_cost_bps=50)
    assert costly.equity_curve.iloc[-1] < cheap.equity_curve.iloc[-1]


def test_execution_happens_at_next_open_not_current_close():
    """A strategy that flips from flat to long based on today's close should
    not be able to capture today's close-to-close return -- it should only
    capture returns starting from tomorrow's open."""
    n = 10
    dates = pd.bdate_range("2021-01-01", periods=n)
    # Big jump on day 3 that a same-day-execution bug would incorrectly capture.
    closes = [100, 100, 100, 200, 200, 200, 200, 200, 200, 200]
    opens = [100, 100, 100, 100, 200, 200, 200, 200, 200, 200]
    data = pd.DataFrame({"Open": opens, "High": closes, "Low": opens,
                          "Close": closes, "Volume": 1000}, index=dates)

    class FlipOnDay3:
        name = "flip_on_day3"
        def reset(self): pass
        def generate_signal(self, market_state):
            return 1.0 if len(market_state) >= 4 else 0.0  # decides to go long after seeing day-3 close

    result = run_backtest(FlipOnDay3(), data, initial_capital=100_000, transaction_cost_bps=0)
    # Position should still be 0 on day index 3 (decision made using data
    # through day 3, executed at day 4's open) -- it must not retroactively
    # capture day 3's jump from 100 to 200.
    assert result.positions.iloc[3] == 0.0
    assert result.equity_curve.iloc[3] == pytest.approx(100_000, rel=1e-6)
