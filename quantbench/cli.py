"""Command-line entry point.

Usage:
    python -m quantbench.cli run experiments/ai_vs_momentum_spy.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from quantbench.data.loader import load_prices
from quantbench.evaluation.metrics import summarize
from quantbench.evaluation.walk_forward import make_walk_forward_folds
from quantbench.portfolio.simulator import run_backtest
from quantbench.reporting.report import build_report, write_report
from quantbench.strategies.ai_assisted import AIParameterSelector
from quantbench.strategies.rule_based import (
    BuyAndHold, MeanReversionZScore, Momentum, MovingAverageCrossover, RSIStrategy,
    VolatilityTargeting,
)

STRATEGY_REGISTRY = {
    "buy_and_hold": lambda: BuyAndHold(),
    "momentum_60_day": lambda: Momentum(lookback=60),
    "ma_crossover_20_100": lambda: MovingAverageCrossover(fast=20, slow=100),
    "mean_reversion_z": lambda: MeanReversionZScore(),
    "rsi_14": lambda: RSIStrategy(),
    "vol_target_10pct": lambda: VolatilityTargeting(target_annual_vol=0.10),
    "ai_parameter_selector": lambda: AIParameterSelector(),
}


def run_cell(
    asset: str,
    strategy_key: str,
    start_date: str,
    end_date: str,
    transaction_cost_bps: float,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
) -> dict:
    """Run one (asset, strategy, transaction-cost) walk-forward evaluation and
    return the fold-averaged metrics dict. This is the atomic unit of work
    shared by both the YAML-driven `run_experiment` (one-off, human-picked
    experiments) and the automation batch runner (`automation/run_batch.py`,
    which works through a large queue of these cells over many scheduled runs).
    """
    if strategy_key not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy '{strategy_key}'. Known: {list(STRATEGY_REGISTRY)}")

    price_data = load_prices(asset, start_date, end_date)
    folds = make_walk_forward_folds(price_data, train_years=train_years, test_years=test_years, step_years=step_years)
    if not folds:
        raise ValueError(f"No walk-forward folds fit for {asset} in {start_date}..{end_date} "
                          f"with train_years={train_years}, test_years={test_years}.")

    fold_metrics = []
    for fold in folds:
        strat = STRATEGY_REGISTRY[strategy_key]()
        if hasattr(strat, "fit"):
            strat.fit(fold.train)  # AI-assisted: parameters chosen on TRAIN only
        result = run_backtest(strat, fold.test, transaction_cost_bps=transaction_cost_bps)
        m = summarize(result.returns, result.positions, result.total_transaction_costs)
        m["test_period"] = f"{fold.test_range[0]} to {fold.test_range[1]}"
        fold_metrics.append(m)

    numeric_keys = [k for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))]
    agg = {k: sum(fm[k] for fm in fold_metrics) / len(fold_metrics) for k in numeric_keys}
    agg["n_folds"] = len(fold_metrics)
    return agg


def run_experiment(config_path: str) -> Path:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    asset = cfg["assets"][0]  # MVP: single-asset experiments
    cost_bps = cfg.get("transaction_cost_bps", 5)

    all_results = {}
    for strat_key in cfg["strategies"]:
        all_results[strat_key] = run_cell(
            asset=asset,
            strategy_key=strat_key,
            start_date=cfg["start_date"],
            end_date=cfg["end_date"],
            transaction_cost_bps=cost_bps,
            train_years=cfg.get("train_years", 3),
            test_years=cfg.get("test_years", 1),
            step_years=cfg.get("step_years", 1),
        )

    report = build_report(
        experiment_name=cfg["experiment_name"],
        asset=asset,
        date_range=(cfg["start_date"], cfg["end_date"]),
        transaction_cost_bps=cost_bps,
        results=all_results,
        notes=cfg.get("notes", ""),
    )
    return write_report(report)


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "run":
        print("Usage: python -m quantbench.cli run <experiment.yaml>")
        sys.exit(1)
    path = run_experiment(sys.argv[2])
    print(f"Report written to: {path}")


if __name__ == "__main__":
    main()
