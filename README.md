# quantbench: AI-assisted vs. rule-based quant strategy backtester

**Research question:** does AI-assisted parameter selection produce better
out-of-sample, after-cost, risk-adjusted performance than fixed rule-based
strategies, or does it just look better in-sample?

This is a deliberately scoped-down MVP. It does one thing end-to-end and
honestly: load real historical data, run several transparent rule-based
strategies and one constrained AI-assisted strategy through the same
walk-forward, cost-aware simulator, and report whether the AI actually
wins out-of-sample. It is not trying to be a full trading platform.

## Why the scope is small on purpose

An earlier draft of this project included four tiers of AI sophistication
(parameter selection → strategy generation → regime detection → news
sentiment), a live dashboard, an experiment registry, and full open-source
contributor infrastructure (CI, CONTRIBUTING guidelines, beginner-tagged
issues). That version does not fit in a few weeks of part-time work by one
person and risks shipping a scaffold with no real result inside it, which
is worse than a smaller, finished project. This version keeps only what's
needed to answer the research question honestly:

- 5 rule-based strategies as transparent baselines
- 1 AI-assisted strategy: constrained parameter selection (Level 1 only),
  validated against a strict schema, fit on training data only
- A simulator that executes at next-bar open (not same-bar close) and
  charges transaction costs on every trade
- Walk-forward evaluation so no fold's test period overlaps or precedes
  its own training period
- A reproducible experiment report (git commit hash, timestamp, full
  config) for every run

Not included, deliberately: news/sentiment analysis, a dashboard beyond a
simple optional Streamlit viewer, multi-asset *portfolios* (a single trade
that spans several assets at once), and any open-source contributor
infrastructure aimed at outside contributors who won't materialize in the
short term. These are documented as **future work** below, not built.

## The ongoing study

Beyond the single SPY experiment above, the repo runs a much larger
walk-forward study: every combination of 15 liquid assets across equities,
bonds, commodities, and sectors (`automation/universe.py`), 7 strategies
(6 rule-based + the AI parameter selector), and 3 transaction-cost
assumptions (5/10/20 bps): 315 (asset, strategy, cost) cells in total,
each evaluated with the same walk-forward, leakage-guarded simulator as
the single-asset example.

This runs incrementally rather than all at once. `automation/queue.py`
tracks exactly which cells have been evaluated in a checked-in state file
(`automation/state/queue_state.json`), and `automation/run_batch.py` picks
up the next batch of *not-yet-run* cells each time it's invoked, appends
their results to `reports/aggregate_results.csv`, and appends a dated
summary to `RESEARCH_LOG.md`. A scheduled GitHub Actions workflow
(`.github/workflows/run_experiments.yml`) calls this a few times a week,
so the repo's commit history is a real, growing record of the study's
progress, not a backfilled or padded log. Widening the study later (more
assets, more strategies, a sensitivity sweep on walk-forward window
lengths) just means adding to `automation/universe.py`; already-completed
cells are never re-run or lost.

Run a batch yourself at any time with:

```bash
python -m automation.run_batch --batch-size 20
```

See "Running this yourself on GitHub" below for how the scheduled version
is wired up.

## How it works

```
Historical daily OHLCV (yfinance, cached to disk)
        |
Walk-forward split (train N years -> test 1 year, rolled forward)
        |
   +---------------------------+
   | Rule-based strategies     |    AI-assisted strategy
   | (buy&hold, momentum,      |    (grid-searches parameters on
   |  MA crossover, mean       |     TRAIN fold only, freezes them,
   |  reversion, RSI)          |     validated via pydantic schema)
   +---------------------------+
        |
Portfolio simulator (next-open execution, transaction costs)
        |
Metrics (Sharpe, Sortino, max drawdown, win rate, profit factor, turnover)
        |
Reproducible report (json + markdown, git hash, timestamp, config)
```

The anti-leakage rule that matters most: a strategy is only ever handed
data up to and including today's close, and any resulting position change
executes at **tomorrow's open**, never today's close. This is enforced in
the simulator loop itself (`quantbench/portfolio/simulator.py`), not just
documented as a convention.

The AI-assisted strategy (`quantbench/strategies/ai_assisted.py`) is
intentionally constrained: it never generates or executes arbitrary code.
It returns a small config (signal type, lookback window, entry/exit
thresholds, max position size) that is validated against a pydantic
schema before it's allowed to touch a real backtest. Parameters are chosen
using only the training fold and then frozen for the test fold. This is
the project's central defense against an "AI" (or a human) that just
overfits to the period it's graded on. The default parameter-selection
function is a deterministic grid search so the whole pipeline runs with
zero API keys. Swapping in a real LLM call means writing a new function
with the same signature (train data in, config dict out); nothing else
changes.

## Quickstart

```bash
pip install -r requirements.txt
pytest                                              # 31 tests, all passing
python -m quantbench.cli run experiments/ai_vs_momentum_spy.yaml
```

The bundled experiment (`experiments/ai_vs_momentum_spy.yaml`) runs SPY,
2010–2024, 12 rolling walk-forward folds (3-year train / 1-year test,
stepped forward 1 year at a time), all 6 strategies, 5 bps transaction
costs. It takes roughly one minute on a normal laptop; most of that time
is the AI strategy's in-sample grid search, run fresh on every fold's
training data.

### A real result (sandbox smoke test)

To confirm the whole pipeline actually works end-to-end before handing
this off, it was run in a sandboxed environment against real SPY data
with a sparser 3-fold schedule (for speed; the full 12-fold config is what
ships in `experiments/`). Real, unedited output:

| Strategy | Annualized Return | Sharpe | Max Drawdown | Turnover |
|---|---|---|---|---|
| buy_and_hold | 26.8% | 2.46 | -4.4% | 0.004 |
| ai_parameter_selector | 20.2% | 2.15 | -4.6% | 0.007 |
| momentum_60_day | 18.3% | 2.03 | -4.3% | 0.004 |
| ma_crossover_20_100 | 12.8% | 1.72 | -4.2% | 0.004 |
| mean_reversion_z | 7.8% | 1.88 | -2.2% | 0.053 |
| rsi_14 | 3.9% | 1.08 | -2.4% | 0.013 |

In this quick run, the AI-assisted strategy beat every rule-based
*timing* strategy on both return and Sharpe, but still lost to plain
buy-and-hold. That's a useful, non-obvious finding: exactly the kind of
result an admissions reader (or an interviewer) can be shown and asked to
defend. It is a 3-fold smoke test, not the final answer; the full 12-fold
run is what should be reported as the real experiment.

## Running this yourself on GitHub

The scheduled study needs the code pushed to a real GitHub repo you
control. This never requires giving anyone (including an AI assistant) a
personal access token or password. GitHub Actions provides its own
short-lived token to every workflow run automatically.

1. Create an empty repository on github.com (no README/license, so it
   stays truly empty).
2. From this project's folder:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: quantbench MVP + multi-asset study automation"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
3. On GitHub, go to **Settings → Actions → General → Workflow permissions**
   and select **"Read and write permissions"**. This lets the scheduled
   workflow commit its own results back to the repo using GitHub's
   built-in token (`secrets.GITHUB_TOKEN`); there's nothing to create or paste in.
4. That's it. The workflow in `.github/workflows/run_experiments.yml` runs
   Monday/Wednesday/Friday automatically, or click **Actions → Run
   walk-forward study batch → Run workflow** to trigger one manually right
   away instead of waiting for the schedule.

## Reproducibility

Every report (`reports/*.json` and `.md`) records: experiment name, asset,
date range, transaction cost assumption, git commit hash, generation
timestamp, and the full per-strategy metrics for every walk-forward fold,
averaged. Anyone can re-run `quantbench run experiments/<name>.yaml` and
get the same numbers, because market data is cached to disk on first
fetch (`data_cache/`) rather than silently re-pulled and potentially
revised by the data vendor later.

## Project structure

```
quantbench/
├── data/loader.py           # cached historical OHLCV loading
├── strategies/
│   ├── base.py               # Strategy interface
│   ├── rule_based.py         # buy&hold, momentum, MA crossover, mean reversion, RSI, vol targeting
│   └── ai_assisted.py        # constrained, schema-validated AI parameter selector
├── portfolio/simulator.py   # next-open execution, transaction costs, leakage guards
├── evaluation/
│   ├── metrics.py            # Sharpe, Sortino, drawdown, win rate, profit factor, turnover
│   └── walk_forward.py       # rolling train/test fold generation
├── reporting/report.py      # reproducible experiment reports
└── cli.py                    # `quantbench run experiments/x.yaml`, shared run_cell()
automation/
├── universe.py               # scope of the ongoing study (assets, strategies, cost scenarios)
├── queue.py                  # tracks which (asset, strategy, cost) cells are done vs. pending
├── run_batch.py               # runs the next batch, updates CSV + research log
└── state/queue_state.json    # checked-in progress record (not hand-edited)
.github/workflows/
└── run_experiments.yml       # scheduled batch runs (Mon/Wed/Fri), pytest gate before committing
experiments/                  # YAML experiment configs (one-off, human-picked)
reports/aggregate_results.csv # every completed study cell, one row each
RESEARCH_LOG.md                # human-readable, append-only log of what each batch found
tests/                        # 31 tests covering metrics, strategies, leakage, walk-forward, queue
dashboard/app.py              # optional Streamlit viewer for a saved report
```

## Testing

```bash
pytest -q
```

31 tests, including specific regression tests for the two subtlest bugs
in a project like this: a strategy retroactively capturing a price move
it shouldn't have been able to see (`test_execution_happens_at_next_open_not_current_close`),
and walk-forward folds silently overlapping train/test periods
(`test_folds_are_non_overlapping_and_forward_only`), plus tests confirming
the study queue never loses or re-runs a completed cell even as the
universe grows (`test_load_state_persists_completed_cells_across_universe_growth`).
The scheduled workflow runs this full suite before every commit, so a
broken change can't silently poison the study's results.

## Honest limitations / future work

- Each study cell is single-asset (no cross-asset portfolio construction:
  the study measures how strategies perform *per asset*, not a blended
  portfolio across all 15 at once). Portfolio-level backtesting is a
  natural next step, not built here.
- The AI strategy is Level 1 (parameter selection) only. Higher levels
  (strategy generation, regime detection, news sentiment) are real ideas
  but each is its own project-sized scope and was deliberately cut.
- No slippage model beyond the flat bps transaction cost. Real slippage
  varies with order size and liquidity.
- The dashboard (`dashboard/app.py`) reads a single saved report; it does
  not yet visualize the growing `aggregate_results.csv` study (a natural
  next addition once there's enough data in it to be worth charting).
- No contributor-facing infrastructure (CONTRIBUTING beyond the basics,
  beginner-tagged issues). If this project gets real outside interest
  later, that's the right time to add it, not before there's anyone to
  use it.

## A note on the writing

Most of the code and prose here was built with an AI coding assistant.
`RESEARCH_LOG.md` entries are generated automatically by
`automation/run_batch.py`; everything else (this README, `CONTRIBUTING.md`,
code comments) went through an editing pass to strip out the usual signs
of unedited AI writing (em dashes standing in for punctuation, inflated
"marks a significant milestone" language, filler that pads a sentence
without adding information). See `CONTRIBUTING.md` for the actual rule
applied to new writing going forward.
