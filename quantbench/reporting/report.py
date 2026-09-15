"""Experiment report generation.

Every report captures enough metadata that a stranger (or the student, six
months later) could reproduce the exact result: dataset, date range,
strategy config, transaction cost assumption, random seed if any, git
commit hash, and a timestamp. This is what makes an experiment a claim you
can defend rather than a number you happened to get once.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _git_commit_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "no-git-repo"


def build_report(
    experiment_name: str,
    asset: str,
    date_range: tuple[str, str],
    transaction_cost_bps: float,
    results: dict,  # strategy_name -> metrics dict
    random_seed: int | None = None,
    notes: str = "",
) -> dict:
    return {
        "experiment_name": experiment_name,
        "asset": asset,
        "date_range": {"start": date_range[0], "end": date_range[1]},
        "transaction_cost_bps": transaction_cost_bps,
        "random_seed": random_seed,
        "git_commit_hash": _git_commit_hash(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "quantbench_version": "0.1.0-mvp",
        "results": results,
        "notes": notes,
    }


def write_report(report: dict, out_dir: str = "reports") -> Path:
    Path(out_dir).mkdir(exist_ok=True)
    stamp = report["generated_at_utc"].replace(":", "-")
    json_path = Path(out_dir) / f"{report['experiment_name']}_{stamp}.json"
    md_path = Path(out_dir) / f"{report['experiment_name']}_{stamp}.md"

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    with open(md_path, "w") as f:
        f.write(f"# Experiment: {report['experiment_name']}\n\n")
        f.write(f"- Asset: {report['asset']}\n")
        f.write(f"- Date range: {report['date_range']['start']} to {report['date_range']['end']}\n")
        f.write(f"- Transaction cost: {report['transaction_cost_bps']} bps\n")
        f.write(f"- Git commit: `{report['git_commit_hash']}`\n")
        f.write(f"- Generated (UTC): {report['generated_at_utc']}\n\n")
        f.write("## Results\n\n")
        rows = list(report["results"].items())
        if rows:
            metric_names = list(next(iter(rows))[1].keys())
            f.write("| Strategy | " + " | ".join(metric_names) + " |\n")
            f.write("|---" * (len(metric_names) + 1) + "|\n")
            for strat_name, metrics in rows:
                vals = [f"{metrics[m]:.4f}" if isinstance(metrics[m], float) else str(metrics[m])
                        for m in metric_names]
                f.write(f"| {strat_name} | " + " | ".join(vals) + " |\n")
        if report.get("notes"):
            f.write(f"\n## Notes\n\n{report['notes']}\n")

    return md_path
