"""Minimal Streamlit viewer for a saved experiment report.

Run with: streamlit run dashboard/app.py -- --report reports/<name>.json

Deliberately simple: loads one report JSON and renders the results table
plus a couple of comparison charts. Not a live research tool -- that was
cut from scope on purpose (see README "Honest limitations").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import streamlit as st


def load_latest_report(reports_dir: str = "reports") -> dict | None:
    files = sorted(Path(reports_dir).glob("*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def main():
    st.set_page_config(page_title="quantbench", layout="wide")
    st.title("quantbench: AI-Assisted vs. Rule-Based Strategy Comparison")

    report = load_latest_report()
    if report is None:
        st.warning("No reports found. Run `python -m quantbench.cli run experiments/<name>.yaml` first.")
        return

    st.caption(
        f"Experiment: **{report['experiment_name']}** | Asset: {report['asset']} | "
        f"Range: {report['date_range']['start']} to {report['date_range']['end']} | "
        f"Cost: {report['transaction_cost_bps']} bps | "
        f"Commit: `{report['git_commit_hash']}` | Generated: {report['generated_at_utc']}"
    )

    df = pd.DataFrame(report["results"]).T
    st.subheader("Metrics by strategy (averaged across walk-forward folds)")
    st.dataframe(df.style.format("{:.4f}"))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Annualized return")
        st.bar_chart(df["annualized_return"])
    with col2:
        st.subheader("Sharpe ratio")
        st.bar_chart(df["sharpe_ratio"])

    st.subheader("Risk: max drawdown")
    st.bar_chart(df["max_drawdown"])

    if report.get("notes"):
        st.subheader("Notes")
        st.write(report["notes"])


if __name__ == "__main__":
    main()
