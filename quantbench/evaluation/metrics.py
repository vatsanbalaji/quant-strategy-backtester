"""Performance and risk metrics. All functions take a daily returns Series
(simple returns, not log returns) and are self-contained / testable in
isolation with synthetic data -- see tests/test_metrics.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def total_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series) -> float:
    n = len(returns)
    if n == 0:
        return 0.0
    cumulative = (1 + returns).prod()
    years = n / TRADING_DAYS_PER_YEAR
    if years <= 0 or cumulative <= 0:
        return -1.0
    return float(cumulative ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


_NEAR_ZERO_STD = 1e-10  # below this, treat volatility as zero rather than dividing by float noise


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    std = excess.std()
    if std < _NEAR_ZERO_STD or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std < _NEAR_ZERO_STD or np.isnan(downside_std):
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def win_rate(returns: pd.Series) -> float:
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).mean())


def profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def turnover(positions: pd.Series) -> float:
    """Average absolute daily change in position -- a proxy for how much
    trading a strategy does (and therefore how cost-sensitive it is)."""
    return float(positions.diff().abs().mean())


def summarize(returns: pd.Series, positions: pd.Series | None = None,
              total_transaction_costs: float = 0.0, initial_capital: float = 100_000.0) -> dict:
    """One-stop metrics dict used by both the report writer and the tests."""
    out = {
        "total_return": total_return(returns),
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "total_transaction_costs": total_transaction_costs,
        "transaction_cost_pct_of_capital": total_transaction_costs / initial_capital if initial_capital else 0.0,
    }
    if positions is not None:
        out["turnover"] = turnover(positions)
    return out
