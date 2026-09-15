import automation.queue as queue_mod
from automation.queue import mark_done, mark_error, pop_next_batch, progress_summary


def test_build_full_queue_size_matches_universe(monkeypatch):
    # queue.py did `from automation.universe import ASSET_UNIVERSE, ...`, so the
    # names live on queue_mod itself -- patch them there directly rather than on
    # automation.universe (which queue.py no longer looks at after import time).
    monkeypatch.setattr(queue_mod, "ASSET_UNIVERSE", ["SPY", "QQQ"])
    monkeypatch.setattr(queue_mod, "STRATEGY_KEYS", ["buy_and_hold", "momentum_60_day"])
    monkeypatch.setattr(queue_mod, "COST_SCENARIOS_BPS", [5, 10])

    q = queue_mod.build_full_queue()
    assert len(q) == 2 * 2 * 2  # assets x strategies x costs
    assert all(c["status"] == "pending" for c in q)
    ids = {c["id"] for c in q}
    assert "SPY|buy_and_hold|5bps" in ids
    assert "QQQ|momentum_60_day|10bps" in ids


def test_pop_next_batch_respects_size_and_status():
    state = {"cells": [
        {"id": "a", "status": "pending"},
        {"id": "b", "status": "done"},
        {"id": "c", "status": "pending"},
        {"id": "d", "status": "pending"},
    ]}
    batch = pop_next_batch(state, batch_size=2)
    assert [c["id"] for c in batch] == ["a", "c"]


def test_mark_done_updates_status_and_result():
    state = {"cells": [{"id": "x", "status": "pending", "result": None, "completed_at": None}]}
    mark_done(state, "x", {"sharpe_ratio": 1.5})
    cell = state["cells"][0]
    assert cell["status"] == "done"
    assert cell["result"] == {"sharpe_ratio": 1.5}
    assert cell["completed_at"] is not None


def test_mark_error_records_message():
    state = {"cells": [{"id": "x", "status": "pending", "error": None, "completed_at": None}]}
    mark_error(state, "x", "no data returned")
    cell = state["cells"][0]
    assert cell["status"] == "error"
    assert cell["error"] == "no data returned"


def test_progress_summary_counts_correctly():
    state = {"cells": [
        {"status": "done"}, {"status": "done"}, {"status": "error"}, {"status": "pending"},
    ]}
    summary = progress_summary(state)
    assert summary == {"total": 4, "done": 2, "errored": 1, "pending": 1, "pct_complete": 50.0}


def test_load_state_persists_completed_cells_across_universe_growth(tmp_path, monkeypatch):
    monkeypatch.setattr(queue_mod, "STATE_PATH", tmp_path / "queue_state.json")
    monkeypatch.setattr(queue_mod, "ASSET_UNIVERSE", ["SPY"])
    monkeypatch.setattr(queue_mod, "STRATEGY_KEYS", ["buy_and_hold"])
    monkeypatch.setattr(queue_mod, "COST_SCENARIOS_BPS", [5])

    state = queue_mod.load_state()
    assert len(state["cells"]) == 1
    queue_mod.mark_done(state, state["cells"][0]["id"], {"sharpe_ratio": 2.0})
    queue_mod.save_state(state)

    # Widen the universe -- the previously-completed cell must survive as "done",
    # and the new cell must show up as "pending". This is the property that makes
    # it safe to grow automation/universe.py mid-study without losing history.
    monkeypatch.setattr(queue_mod, "ASSET_UNIVERSE", ["SPY", "QQQ"])
    reloaded = queue_mod.load_state()
    by_id = {c["id"]: c for c in reloaded["cells"]}
    assert len(reloaded["cells"]) == 2
    assert by_id["SPY|buy_and_hold|5bps"]["status"] == "done"
    assert by_id["QQQ|buy_and_hold|5bps"]["status"] == "pending"
