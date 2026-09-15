"""Defines the scope of the ongoing multi-asset study.

This is intentionally the single place that controls how big the study is.
Widening ASSET_UNIVERSE, STRATEGY_KEYS, or COST_SCENARIOS_BPS gives the
automated queue (see queue.py) more genuinely new work to do -- that's the
lever for "more breadth" rather than re-running the same cells more often.
"""
from __future__ import annotations

# Broad multi-asset universe spanning equities (US + international),
# fixed income, commodities, and equity sectors -- chosen so the study
# says something about strategy robustness across very different return
# and volatility regimes, not just "works on SPY."
ASSET_UNIVERSE = [
    "SPY",  # US large-cap equities
    "QQQ",  # US tech-heavy equities
    "IWM",  # US small-cap equities
    "DIA",  # US large-cap (price-weighted)
    "EFA",  # developed international equities
    "EEM",  # emerging market equities
    "TLT",  # long-duration US Treasuries
    "IEF",  # intermediate US Treasuries
    "LQD",  # investment-grade corporate bonds
    "GLD",  # gold
    "SLV",  # silver
    "XLF",  # financial sector
    "XLK",  # technology sector
    "XLE",  # energy sector
    "XLV",  # healthcare sector
]

STRATEGY_KEYS = [
    "buy_and_hold",
    "momentum_60_day",
    "ma_crossover_20_100",
    "mean_reversion_z",
    "rsi_14",
    "vol_target_10pct",
    "ai_parameter_selector",
]

# Sensitivity axis: does a strategy's edge survive realistic and pessimistic
# transaction cost assumptions, or does it only "win" in a frictionless
# fantasy? 5bps ~ a retail ETF trade; 20bps is a deliberately harsh stress case.
COST_SCENARIOS_BPS = [5, 10, 20]

WALK_FORWARD_CONFIG = dict(train_years=3, test_years=1, step_years=1)

# Data start date is intentionally early; load_prices/yfinance will simply
# return whatever history actually exists for a given ticker (e.g. GLD only
# goes back to 2004), so this does not need to be tuned per asset.
STUDY_START_DATE = "2000-01-01"
