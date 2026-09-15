"""Work queue for the ongoing automated study.

The whole point of this file: every automated run should do genuinely NEW
analysis, not re-run something already committed to the repo. The queue is
itself a checked-in JSON file (automation/state/queue_state.json) so its
progress is part of the repo's real history -- anyone can see exactly how
much of the study has been completed as of any commit.
"""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

from automation.universe import ASSET_UNIVERSE, COST_SCENARIOS_BPS, STRATEGY_KEYS

STATE_PATH = Path(__file__).resolve().parent / "state" / "queue_state.json"


def _cell_id(asset: str, strategy: str, cost_bps: int) -> str:
    return f"{asset}|{strategy}|{cost_bps}bps"


def build_full_queue() -> list[dict]:
    """Every (asset, strategy, cost) combination in the current study scope."""
    cells = []
    for asset, strategy, cost_bps in itertools.product(ASSET_UNIVERSE, STRATEGY_KEYS, COST_SCENARIOS_BPS):
        cells.append({
            "id": _cell_id(asset, strategy, cost_bps),
            "asset": asset,
            "strategy": strategy,
            "cost_bps": cost_bps,
            "status": "pending",
            "result": None,
            "completed_at": None,
            "error": None,
        })
    return cells


def load_state() -> dict:
    """Load existing queue state, or initialize it from the full queue if
    this is the first run. Growing ASSET_UNIVERSE/STRATEGY_KEYS/COST_SCENARIOS_BPS
    later will merge in newly-added cells as 'pending' without touching
    already-completed ones.
    """
    full_queue = build_full_queue()
    full_by_id = {c["id"]: c for c in full_queue}

    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            existing = json.load(f)
        existing_by_id = {c["id"]: c for c in existing.get("cells", [])}
        # Merge: keep completed/errored history for cells that still exist,
        # add any brand-new cells (from a widened universe) as pending.
        merged = []
        for cell_id, cell in full_by_id.items():
            merged.append(existing_by_id.get(cell_id, cell))
        return {"cells": merged, "schema_version": 1}

    return {"cells": full_queue, "schema_version": 1}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def pop_next_batch(state: dict, batch_size: int) -> list[dict]:
    """Return the next `batch_size` pending cells (does not mutate state --
    caller marks them done/errored after actually running them)."""
    pending = [c for c in state["cells"] if c["status"] == "pending"]
    return pending[:batch_size]


def mark_done(state: dict, cell_id: str, result: dict) -> None:
    for c in state["cells"]:
        if c["id"] == cell_id:
            c["status"] = "done"
            c["result"] = result
            c["completed_at"] = datetime.now(timezone.utc).isoformat()
            return
    raise KeyError(f"Unknown cell id: {cell_id}")


def mark_error(state: dict, cell_id: str, error_message: str) -> None:
    for c in state["cells"]:
        if c["id"] == cell_id:
            c["status"] = "error"
            c["error"] = error_message
            c["completed_at"] = datetime.now(timezone.utc).isoformat()
            return
    raise KeyError(f"Unknown cell id: {cell_id}")


def progress_summary(state: dict) -> dict:
    total = len(state["cells"])
    done = sum(1 for c in state["cells"] if c["status"] == "done")
    errored = sum(1 for c in state["cells"] if c["status"] == "error")
    pending = total - done - errored
    return {"total": total, "done": done, "errored": errored, "pending": pending,
            "pct_complete": round(100 * done / total, 1) if total else 0.0}
