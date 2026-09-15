"""Portfolio and execution simulator.

Anti-leakage rule enforced structurally, not just by convention: on day t
the strategy is shown data up to and including day t's close, and its
resulting target position is executed at day t+1's OPEN, not day t's
close. A strategy literally cannot trade on information from a bar that
hasn't happened yet, because the loop never gives it that data.

Transaction costs are modeled as a flat bps charge on the traded notional
(the change in position size), applied at execution time -- this is what
turns "strategy has positive raw returns" into "strategy has positive
returns after realistic frictions," which is the whole point of including
it rather than reporting frictionless backtests like most toy projects do.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantbench.strategies.base import Strategy


@dataclass
class BacktestResult:
    strategy_name: str
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    total_transaction_costs: float
    initial_capital: float


def run_backtest(
    strategy: Strategy,
    price_data: pd.DataFrame,
    initial_capital: float = 100_000.0,
    transaction_cost_bps: float = 5.0,
    max_position_change_per_day: float = 1.0,
    max_lookback: int = 260,
) -> BacktestResult:
    """Simulate `strategy` against `price_data` (must contain Open, Close columns,
    sorted ascending by date, no gaps introduced by the caller).

    Position sizing: target position (from generate_signal) is a fraction
    of current equity, in [-1, 1]. Execution happens at next bar's open.

    `max_lookback` bounds how much history is handed to the strategy each
    day (a rolling window rather than the full history-to-date). This is a
    performance optimization, not a leakage change: strategies in this repo
    use bounded rolling windows (moving averages, RSI, z-scores) so a
    window this wide (~1 trading year by default) never truncates data a
    strategy would actually use. Without this bound, growing the slice to
    the full history every day makes a backtest O(n^2) in the number of
    days, which is the difference between seconds and minutes on 15 years
    of daily data.
    """
    strategy.reset()
    dates = price_data.index
    opens = price_data["Open"].values
    closes = price_data["Close"].values
    n = len(price_data)

    equity = np.zeros(n)
    equity[0] = initial_capital
    positions = np.zeros(n)  # position fraction actually held during bar i
    cash = initial_capital
    shares_held = 0.0
    cost_bps = transaction_cost_bps / 10_000.0
    total_costs = 0.0
    trade_log = []

    target_position = 0.0  # decided using data through bar i-1, executed at open of bar i
    last_executed_target = None  # the position fraction we last actually rebalanced to
    REBALANCE_TOLERANCE = 1e-6  # ignore sub-basis-point target changes -- avoids phantom
    # trades caused by floating-point drift in the equity used to size positions

    for i in range(n):
        # Execute a trade at TODAY's open ONLY if the strategy's target position
        # actually changed from what we last executed. Without this guard, tiny
        # equity fluctuations (e.g. from a transaction cost paid yesterday) would
        # cause the position-fraction-of-equity math to imply a "rebalance" every
        # single day even when the strategy's signal hasn't changed -- a
        # simulator artifact, not real trading behavior.
        if last_executed_target is None or abs(target_position - last_executed_target) > REBALANCE_TOLERANCE:
            current_equity_at_open = cash + shares_held * opens[i]
            target_shares = (target_position * current_equity_at_open) / opens[i] if opens[i] else shares_held
            traded_shares = target_shares - shares_held

            if abs(traded_shares) > 1e-9:
                notional = abs(traded_shares) * opens[i]
                cost = notional * cost_bps
                cash -= traded_shares * opens[i]  # buy costs cash, sell adds cash
                cash -= cost
                total_costs += cost
                trade_log.append({
                    "date": dates[i], "traded_shares": traded_shares,
                    "price": opens[i], "cost": cost,
                })
                shares_held = target_shares
            last_executed_target = target_position

        positions[i] = target_position
        equity[i] = cash + shares_held * closes[i]

        # Now decide tomorrow's target position using data through TODAY's close.
        window_start = max(0, i + 1 - max_lookback)
        market_state = price_data.iloc[window_start: i + 1]
        raw_signal = strategy.generate_signal(market_state)
        target_position = max(-max_position_change_per_day, min(max_position_change_per_day, raw_signal))

    equity_curve = pd.Series(equity, index=dates, name="equity")
    returns = equity_curve.pct_change().fillna(0.0)
    trades_df = pd.DataFrame(trade_log)

    return BacktestResult(
        strategy_name=strategy.name,
        equity_curve=equity_curve,
        returns=returns,
        positions=pd.Series(positions, index=dates, name="position"),
        trades=trades_df,
        total_transaction_costs=total_costs,
        initial_capital=initial_capital,
    )
