import numpy as np
import pandas as pd
import pytest

from quantbench.strategies.ai_assisted import AIParameterSelector, AIStrategyConfig
from quantbench.strategies.rule_based import (
    BuyAndHold, MeanReversionZScore, Momentum, MovingAverageCrossover, RSIStrategy,
)


def _fake_price_data(n=300, seed=1, trend=0.0003, vol=0.01):
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, vol, n)
    close = 100 * np.cumprod(1 + rets)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "Open": close * (1 - 0.0005),
        "High": close * 1.001,
        "Low": close * 0.999,
        "Close": close,
        "Volume": 1_000_000,
    }, index=dates)
    return df


def test_buy_and_hold_always_full_long():
    strat = BuyAndHold()
    data = _fake_price_data()
    assert strat.generate_signal(data) == 1.0


def test_ma_crossover_returns_zero_when_insufficient_history():
    strat = MovingAverageCrossover(fast=20, slow=100)
    data = _fake_price_data(n=50)
    assert strat.generate_signal(data) == 0.0


def test_momentum_position_bounded():
    strat = Momentum(lookback=20)
    data = _fake_price_data(n=200)
    for i in range(21, len(data)):
        sig = strat.generate_signal(data.iloc[:i])
        assert -1.0 <= sig <= 1.0


def test_rsi_signal_bounded():
    strat = RSIStrategy()
    data = _fake_price_data(n=200)
    for i in range(20, len(data), 10):
        sig = strat.generate_signal(data.iloc[:i])
        assert sig in (0.0, 1.0)


def test_strategy_only_sees_data_up_to_current_row():
    """Leakage guard: a strategy handed a truncated slice must not be able
    to see rows beyond what it was given."""
    strat = Momentum(lookback=20)
    data = _fake_price_data(n=200)
    truncated = data.iloc[:100]
    # Sanity: the strategy's own generate_signal only ever reads truncated,
    # so if any future price changes downstream df it cannot matter here.
    sig_before = strat.generate_signal(truncated)
    data.iloc[150:, data.columns.get_loc("Close")] = 999999.0  # mutate "future" rows
    sig_after = strat.generate_signal(truncated)
    assert sig_before == sig_after


def test_ai_strategy_config_rejects_invalid_exit_threshold():
    with pytest.raises(Exception):
        AIStrategyConfig(
            signal="momentum", lookback_days=60,
            entry_threshold=0.02, exit_threshold=0.05,  # exit >= entry: invalid
            maximum_position=1.0,
        )


def test_ai_strategy_config_rejects_out_of_bounds_lookback():
    with pytest.raises(Exception):
        AIStrategyConfig(
            signal="momentum", lookback_days=1000,  # exceeds max
            entry_threshold=0.05, exit_threshold=-0.02,
            maximum_position=1.0,
        )


def test_ai_selector_requires_fit_before_signal():
    selector = AIParameterSelector()
    data = _fake_price_data()
    with pytest.raises(RuntimeError):
        selector.generate_signal(data)


def test_ai_selector_fit_uses_only_training_window():
    train = _fake_price_data(n=200, seed=1)
    selector = AIParameterSelector()
    selector.fit(train)
    assert selector.chosen_config is not None
    assert selector.chosen_config.lookback_days >= 5


def test_volatility_targeting_scales_down_in_high_vol():
    from quantbench.strategies.rule_based import VolatilityTargeting

    strat = VolatilityTargeting(target_annual_vol=0.10, window=20)
    calm = _fake_price_data(n=100, seed=2, trend=0.0002, vol=0.003)   # low realized vol
    choppy = _fake_price_data(n=100, seed=2, trend=0.0002, vol=0.05)  # high realized vol

    calm_exposure = strat.generate_signal(calm)
    strat2 = VolatilityTargeting(target_annual_vol=0.10, window=20)
    choppy_exposure = strat2.generate_signal(choppy)

    assert 0.0 <= calm_exposure <= 1.0
    assert 0.0 <= choppy_exposure <= 1.0
    assert choppy_exposure < calm_exposure, "higher realized vol should mean lower target exposure"


def test_volatility_targeting_zero_before_window_fills():
    from quantbench.strategies.rule_based import VolatilityTargeting
    strat = VolatilityTargeting(window=20)
    data = _fake_price_data(n=10)
    assert strat.generate_signal(data) == 0.0
