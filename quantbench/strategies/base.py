"""Strategy interface.

Every strategy, rule-based or AI-assisted, implements the same contract:
given market state UP TO AND INCLUDING today, return a target position
(as a fraction of portfolio equity, -1.0 to 1.0) for tomorrow.

The "up to and including today, decide for tomorrow" framing is the main
structural defense against look-ahead bias: the simulator (see
portfolio/simulator.py) only ever passes a strategy the data slice that
would genuinely have been available at decision time, and executes the
resulting target position at the NEXT bar's open, not today's close.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Base class all strategies must inherit from."""

    name: str = "unnamed_strategy"

    @abstractmethod
    def generate_signal(self, market_state: pd.DataFrame) -> float:
        """Return a target position in [-1.0, 1.0] as a fraction of equity.

        `market_state` is a DataFrame of OHLCV history up to and including
        the current bar. The strategy must not be given, and must not look
        at, anything beyond the last row.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any internal state between backtest runs. No-op by default."""
        return None
