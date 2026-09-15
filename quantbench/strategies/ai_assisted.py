"""Level 1 AI-assisted strategy: constrained parameter selection.

Design constraints, deliberately strict:

1.  The AI never executes arbitrary code. It only ever returns a small
    JSON-like config, which is validated against a pydantic schema before
    it's allowed to touch a strategy object. If the config doesn't pass
    validation, the run fails loudly instead of silently doing something
    unintended.
2.  The AI only ever sees the TRAINING window. Parameter selection happens
    once, on in-sample data, and the chosen parameters are then frozen and
    evaluated on a completely separate out-of-sample window. This is the
    single most important anti-overfitting rule in the whole project --
    an AI (or a human) tuning parameters against the same data it's
    graded on will always look artificially good in-sample.
3.  The "AI" here is pluggable. `select_fn` defaults to a deterministic
    grid-search proxy so the project runs end-to-end with zero API keys
    and zero network dependency on an LLM provider. Swapping in a real
    LLM call (e.g. asking Claude to propose parameters given a text
    summary of in-sample statistics) means writing a new `select_fn` that
    returns a dict matching `AIStrategyConfig` -- nothing else changes.
"""
from __future__ import annotations

from typing import Callable, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from quantbench.strategies.base import Strategy
from quantbench.strategies.rule_based import Momentum, MovingAverageCrossover


class AIStrategyConfig(BaseModel):
    """Schema the AI's output must conform to. Anything outside these
    bounds is rejected before it ever reaches a live simulation."""

    signal: Literal["momentum", "ma_crossover"]
    lookback_days: int = Field(ge=5, le=252)
    entry_threshold: float = Field(ge=-0.5, le=0.5)
    exit_threshold: float = Field(ge=-0.5, le=0.5)
    maximum_position: float = Field(ge=0.0, le=1.0)

    @field_validator("exit_threshold")
    @classmethod
    def exit_below_entry(cls, v, info):
        entry = info.data.get("entry_threshold")
        if entry is not None and v >= entry:
            raise ValueError("exit_threshold must be below entry_threshold")
        return v


def default_grid_search_select_fn(train_data: pd.DataFrame) -> dict:
    """Deterministic stand-in for an LLM call: grid-searches a small space
    of momentum parameters on the TRAINING window only and returns the
    config with the best in-sample Sharpe ratio. This is intentionally
    "dumb" (no news, no reasoning) -- it exists so the pipeline is testable
    without hitting a real model API. The point of the project isn't
    whether this particular selector is smart; it's whether ANY
    AI/optimization-assisted parameter choice beats fixed rules
    out-of-sample.
    """
    from quantbench.evaluation.metrics import sharpe_ratio
    from quantbench.portfolio.simulator import run_backtest

    best_cfg, best_sharpe = None, -float("inf")
    for lookback in (20, 40, 60, 90):
        for entry in (0.02, 0.05, 0.08):
            for exit_ in (-0.01, -0.02, -0.05):
                cfg = dict(
                    signal="momentum",
                    lookback_days=lookback,
                    entry_threshold=entry,
                    exit_threshold=exit_,
                    maximum_position=1.0,
                )
                strat = Momentum(lookback=lookback, entry_threshold=entry, exit_threshold=exit_)
                result = run_backtest(strat, train_data, transaction_cost_bps=5)
                s = sharpe_ratio(result.returns)
                if s > best_sharpe:
                    best_sharpe, best_cfg = s, cfg
    return best_cfg


class AIParameterSelector(Strategy):
    """Wraps a rule-based strategy whose parameters were chosen by
    `select_fn` on a training window, then frozen for out-of-sample use."""

    name = "ai_parameter_selector"

    def __init__(self, select_fn: Callable[[pd.DataFrame], dict] = default_grid_search_select_fn):
        self.select_fn = select_fn
        self._inner: Strategy | None = None
        self.chosen_config: AIStrategyConfig | None = None

    def fit(self, train_data: pd.DataFrame) -> "AIParameterSelector":
        """Select and freeze parameters using ONLY the training window."""
        raw_cfg = self.select_fn(train_data)
        cfg = AIStrategyConfig(**raw_cfg)  # raises if the AI proposed something invalid
        self.chosen_config = cfg

        if cfg.signal == "momentum":
            self._inner = Momentum(
                lookback=cfg.lookback_days,
                entry_threshold=cfg.entry_threshold,
                exit_threshold=cfg.exit_threshold,
            )
        elif cfg.signal == "ma_crossover":
            fast = max(5, cfg.lookback_days // 3)
            self._inner = MovingAverageCrossover(fast=fast, slow=cfg.lookback_days)
        else:
            raise ValueError(f"Unsupported signal type: {cfg.signal}")

        self.name = f"ai_selector({self._inner.name})"
        return self

    def generate_signal(self, market_state: pd.DataFrame) -> float:
        if self._inner is None:
            raise RuntimeError("Call .fit(train_data) before generate_signal(). "
                                "AI parameter selection must happen on training data only.")
        raw = self._inner.generate_signal(market_state)
        cap = self.chosen_config.maximum_position
        return max(-cap, min(cap, raw))

    def reset(self) -> None:
        if self._inner is not None:
            self._inner.reset()
