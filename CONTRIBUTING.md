# Contributing

This is currently a solo project built as a focused, honest MVP rather
than a community-ready platform. See the README's "Honest limitations"
section for why the scope stops where it does.

If you're looking at this and want to extend it, the natural next steps
(in rough order of how self-contained they are) are:

1. A cross-asset portfolio mode that blends multiple assets into one
   backtest, instead of the current per-asset study.
2. A Level 2 AI strategy (structured strategy generation instead of just
   parameter selection). See `quantbench/strategies/ai_assisted.py` for
   the validation pattern to follow (pydantic schema, no arbitrary code
   execution, train/test separation enforced).
3. A slippage model that scales with order size, not just a flat bps cost.
4. More rule-based baselines beyond the current six (risk parity is a
   natural next one).

Before opening a PR: run `pytest -q` and make sure it's green, and add a
test for whatever you're changing. The existing tests
(`tests/test_simulator.py`, `tests/test_walk_forward.py`) are written as
regression guards against the two easiest mistakes to make in a
backtester (look-ahead bias and overlapping train/test folds), and new
code should hold itself to the same bar.

## Writing style for docs and comments

Most of this project's code and prose was written with an AI coding
assistant. Before any new README section, docstring, or `RESEARCH_LOG.md`
entry gets committed, read it back for the usual tells of unedited AI
writing: em dashes used as a substitute for periods or commas, inflated
"this represents a significant milestone" language, and filler phrases
that pad a sentence without adding information. Cut those and say the
plain thing instead. The goal is prose that reads like a person wrote it
and meant it, not a rewrite of every finding into a press release.
