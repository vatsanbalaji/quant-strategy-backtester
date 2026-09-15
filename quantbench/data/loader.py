"""Historical market data loading.

Design goal: make it structurally hard to leak future information into a
backtest. `load_prices` returns a plain daily OHLCV DataFrame indexed by
date with no forward-fill beyond what the exchange itself would have known
on that date.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data_cache"


def load_prices(ticker: str, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """Load daily OHLCV data for `ticker` between `start` and `end` (inclusive).

    Data is cached to disk as CSV so experiments are reproducible even if
    the upstream data source changes its history later (a real risk with
    free data vendors that revise / adjust historical prices).
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker}_{start}_{end}.csv"

    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df

    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}")

    # yfinance can return a MultiIndex when columns include the ticker level
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "date"
    df.to_csv(cache_file)
    return df


def load_universe(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Load multiple tickers, aligned to a common trading calendar via inner join
    on the intersection of dates actually available for all of them."""
    frames = {t: load_prices(t, start, end) for t in tickers}
    common_index = None
    for df in frames.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    return {t: df.loc[common_index] for t, df in frames.items()}
