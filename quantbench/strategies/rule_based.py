"""Transparent, fully-inspectable rule-based strategies.

These exist as baselines, not as claims that they are profitable. The
whole point of the project is to see whether anything fancier, including
the AI-assisted strategy in ai_assisted.py, actually beats these after
costs, on data the strategy has never seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantbench.strategies.base import Strategy


class BuyAndHold(Strategy):
    name = "buy_and_hold"

    def generate_signal(self, market_state: pd.DataFrame) -> float:
        return 1.0


class MovingAverageCrossover(Strategy):
    """Long when fast SMA > slow SMA, flat otherwise."""

    def __init__(self, fast: int = 20, slow: int = 100):
        if fast >= slow:
            raise ValueError("fast window must be < slow window")
        self.fast = fast
        self.slow = slow
        self.name = f"ma_crossover_{fast}_{slow}"

    def generate_signal(self, market_state: pd.DataFrame) -> float:
        closes = market_state["Close"]
        if len(closes) < self.slow:
            return 0.0
        fast_ma = closes.tail(self.fast).mean()
        slow_ma = closes.tail(self.slow).mean()
        return 1.0 if fast_ma > slow_ma else 0.0


class Momentum(Strategy):
    """Long if trailing `lookback`-day return exceeds `entry_threshold`,
    flat if it falls below `exit_threshold`. Otherwise hold prior state."""

    def __init__(self, lookback: int = 60, entry_threshold: float = 0.05,
                 exit_threshold: float = -0.02):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.name = f"momentum_{lookback}d"
        self._position = 0.0

    def generate_signal(self, market_state: pd.DataFrame) -> float:
        closes = market_state["Close"]
        if len(closes) < self.lookback + 1:
            return 0.0
        ret = closes.iloc[-1] / closes.iloc[-self.lookback - 1] - 1.0
        if ret > self.entry_threshold:
            self._position = 1.0
        elif ret < self.exit_threshold:
            self._position = 0.0
        return self._position

    def reset(self) -> None:
        self._position = 0.0


class MeanReversionZScore(Strategy):
    """Long when price z-score vs its own rolling mean is below -entry_z
    (oversold), exit back to flat when z-score reverts above -exit_z."""

    def __init__(self, window: int = 20, entry_z: float = 1.5, exit_z: float = 0.25):
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.name = f"mean_reversion_z{entry_z}"
        self._position = 0.0

    def generate_signal(self, market_state: pd.DataFrame) -> float:
        closes = market_state["Close"]
        if len(closes) < self.window:
            return 0.0
        window = closes.tail(self.window)
        mean, std = window.mean(), window.std()
        if std == 0 or np.isnan(std):
            return self._position
        z = (closes.iloc[-1] - mean) / std
        if z < -self.entry_z:
            self._position = 1.0
        elif z > -self.exit_z:
            self._position = 0.0
        return self._position

    def reset(self) -> None:
        self._position = 0.0


class RSIStrategy(Strategy):
    """Classic RSI(14): long below oversold threshold, flat above overbought."""

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 60.0):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.name = f"rsi_{period}"
        self._position = 0.0

    def _rsi(self, closes: pd.Series) -> float:
        delta = closes.diff().dropna()
        gains = delta.clip(lower=0).tail(self.period)
        losses = -delta.clip(upper=0).tail(self.period)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def generate_signal(self, market_state: pd.DataFrame) -> float:
        closes = market_state["Close"]
        if len(closes) < self.period + 1:
            return 0.0
        rsi = self._rsi(closes)
        if rsi < self.oversold:
            self._position = 1.0
        elif rsi > self.overbought:
            self._position = 0.0
        return self._position

    def reset(self) -> None:
        self._position = 0.0


class VolatilityTargeting(Strategy):
    """Scales exposure inversely to trailing realized volatility, aiming to
    hold a roughly constant level of portfolio risk rather than a constant
    dollar/share amount. Classic risk-management overlay: cuts exposure
    automatically when a market gets choppy, adds it back when things calm
    down, independent of any return forecast."""

    def __init__(self, target_annual_vol: float = 0.10, window: int = 20, max_leverage: float = 1.0):
        self.target_annual_vol = target_annual_vol
        self.window = window
        self.max_leverage = max_leverage
        self.name = f"vol_target_{int(target_annual_vol*100)}pct"

    def generate_signal(self, market_state: pd.DataFrame) -> float:
        closes = market_state["Close"]
        if len(closes) < self.window + 1:
            return 0.0
        daily_rets = closes.pct_change().dropna().tail(self.window)
        realized_daily_vol = daily_rets.std()
        if realized_daily_vol == 0 or np.isnan(realized_daily_vol):
            return self.max_leverage
        realized_annual_vol = realized_daily_vol * np.sqrt(252)
        exposure = self.target_annual_vol / realized_annual_vol
        return float(max(0.0, min(self.max_leverage, exposure)))
