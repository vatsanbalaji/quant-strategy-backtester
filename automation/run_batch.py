"""Run the next batch of pending (asset, strategy, cost) cells from the study
queue, append results to the aggregate CSV, and append a dated entry to
RESEARCH_LOG.md. Designed to be invoked repeatedly (by a human or by the
scheduled GitHub Actions workflow) -- each invocation advances the queue
and commits only genuinely new results.

Usage:
    python -m automation.run_batch [--batch-size N]

Exit code is always 0 on a normal run (including "queue is empty") so a
scheduled workflow doesn't get marked failed just because the study is
between phases; individual cell failures are recorded in the queue state
and logged, not raised.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from automation.queue import load_state, mark_done, mark_error, pop_next_batch, progress_summary, save_state
from automation.universe import STUDY_START_DATE, WALK_FORWARD_CONFIG
from quantbench.cli import run_cell

REPO_ROOT = Path(__file__).resolve().parent.parent
AGGREGATE_CSV = REPO_ROOT / "reports" / "aggregate_results.csv"
RESEARCH_LOG = REPO_ROOT / "RESEARCH_LOG.md"

CSV_FIELDS = [
    "run_date", "asset", "strategy", "cost_bps", "n_folds",
    "annualized_return", "annualized_volatility", "sharpe_ratio", "sortino_ratio",
    "max_drawdown", "win_rate", "profit_factor", "turnover",
    "total_transaction_costs", "transaction_cost_pct_of_capital",
]


def _today_end_date() -> str:
    return date.today().isoformat()


def _append_to_csv(rows: list[dict]) -> None:
    AGGREGATE_CSV.parent.mkdir(exist_ok=True)
    file_exists = AGGREGATE_CSV.exists()
    with open(AGGREGATE_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _git_commit_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else "no-git-repo"
    except Exception:
        return "no-git-repo"


def _append_to_research_log(batch_results: list[dict], summary: dict) -> None:
    RESEARCH_LOG.touch(exist_ok=True)
    existing = RESEARCH_LOG.read_text()

    lines = []
    lines.append(f"## {datetime.now(timezone.utc).isoformat()} (commit base: {_git_commit_hash()})\n")
    lines.append(f"Ran {len(batch_results)} cells this batch. "
                 f"Study progress: {summary['done']}/{summary['total']} complete ({summary['pct_complete']}%).\n")

    successes = [r for r in batch_results if r["status"] == "done"]
    failures = [r for r in batch_results if r["status"] == "error"]

    if successes:
        best = max(successes, key=lambda r: r["result"]["sharpe_ratio"])
        lines.append(f"- Best Sharpe this batch: `{best['id']}` "
                     f"(Sharpe={best['result']['sharpe_ratio']:.2f}, "
                     f"ann. return={best['result']['annualized_return']:.2%})\n")
        ai_cells = [r for r in successes if r["strategy"] == "ai_parameter_selector"]
        rule_cells = [r for r in successes if r["strategy"] != "ai_parameter_selector"]
        if ai_cells and rule_cells:
            avg_ai_sharpe = sum(r["result"]["sharpe_ratio"] for r in ai_cells) / len(ai_cells)
            avg_rule_sharpe = sum(r["result"]["sharpe_ratio"] for r in rule_cells) / len(rule_cells)
            lines.append(f"- This batch: AI-selector avg Sharpe={avg_ai_sharpe:.2f} "
                         f"vs. rule-based avg Sharpe={avg_rule_sharpe:.2f}.\n")

    if failures:
        lines.append(f"- {len(failures)} cell(s) failed this batch (see queue_state.json `error` field): "
                     + ", ".join(r["id"] for r in failures) + "\n")

    lines.append("\n")

    header = "# Research Log\n\nAppend-only log of what each automated batch run covered and found. " \
             "Full numeric results live in `reports/aggregate_results.csv`; this file is the human-readable trail.\n\n"
    if not existing.strip():
        RESEARCH_LOG.write_text(header + "".join(lines))
    else:
        RESEARCH_LOG.write_text(existing + "".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=15)
    args = parser.parse_args()

    state = load_state()
    batch = pop_next_batch(state, args.batch_size)

    if not batch:
        print("Queue is empty -- current study scope is fully covered. "
              "Widen automation/universe.py to add more work.")
        sys.exit(0)

    end_date = _today_end_date()
    batch_results = []

    for cell in batch:
        try:
            metrics = run_cell(
                asset=cell["asset"],
                strategy_key=cell["strategy"],
                start_date=STUDY_START_DATE,
                end_date=end_date,
                transaction_cost_bps=cell["cost_bps"],
                **WALK_FORWARD_CONFIG,
            )
            mark_done(state, cell["id"], metrics)
            batch_results.append({**cell, "status": "done", "result": metrics})
            print(f"OK   {cell['id']}: sharpe={metrics['sharpe_ratio']:.2f} "
                  f"ann_return={metrics['annualized_return']:.2%}")
        except Exception as e:
            mark_error(state, cell["id"], str(e))
            batch_results.append({**cell, "status": "error", "error": str(e)})
            print(f"FAIL {cell['id']}: {e}")

    save_state(state)

    csv_rows = []
    for r in batch_results:
        if r["status"] == "done":
            csv_rows.append({"run_date": end_date, "asset": r["asset"], "strategy": r["strategy"],
                              "cost_bps": r["cost_bps"], **r["result"]})
    if csv_rows:
        _append_to_csv(csv_rows)

    summary = progress_summary(state)
    _append_to_research_log(batch_results, summary)

    print(f"\nBatch complete. Study progress: {summary['done']}/{summary['total']} "
          f"({summary['pct_complete']}%), {summary['errored']} errored, {summary['pending']} pending.")


if __name__ == "__main__":
    main()
