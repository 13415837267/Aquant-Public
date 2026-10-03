"""Deterministic multi-day paper replay runner.

A replay is a sequence of already validated (plan, next-open snapshot) pairs.
Each cycle carries forward the same paper state and emits a compact equity
curve. No live market or broker API is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.paper_execution_cycle import run_cycle


class ReplayError(ValueError):
    """Replay sequence is invalid or not strictly chronological."""


def run_replay(
    cycles: list[dict[str, Any]],
    initial_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not cycles:
        raise ReplayError("replay requires at least one cycle")

    state = initial_state
    equity_curve: list[dict[str, Any]] = []
    previous_execution_date: str | None = None

    for index, item in enumerate(cycles, start=1):
        if not isinstance(item, dict):
            raise ReplayError(f"cycle {index}: expected object")
        plan = item.get("plan")
        snapshot = item.get("snapshot")
        if not isinstance(plan, dict) or not isinstance(snapshot, dict):
            raise ReplayError(f"cycle {index}: plan and snapshot objects are required")

        execution_date = str(snapshot.get("execution_date", ""))
        if not execution_date:
            raise ReplayError(f"cycle {index}: missing execution_date")
        if previous_execution_date is not None and execution_date <= previous_execution_date:
            raise ReplayError("replay execution dates must be strictly increasing")

        # Strategy provenance must remain continuous throughout a replay.
        if state is not None:
            if state.get("strategy_version") != plan.get("strategy_version"):
                raise ReplayError(f"cycle {index}: strategy version changed during replay")
            if state.get("strategy_commit") != plan.get("strategy_commit"):
                raise ReplayError(f"cycle {index}: strategy commit changed during replay")

        next_state, entries, gate = run_cycle(plan, snapshot, state)
        equity_curve.append(
            {
                "cycle": index,
                "execution_date": execution_date,
                "plan_id": gate["plan_id"],
                "fills": len(entries),
                "cash": round(float(next_state["cash"]), 6),
                "market_value": round(float(next_state.get("market_value", 0.0)), 6),
                "equity": round(float(next_state.get("equity", next_state["cash"])), 6),
                "position_count": len(next_state["positions"]),
            }
        )
        state = next_state
        previous_execution_date = execution_date

    assert state is not None
    state = dict(state)
    state["replay"] = {
        "status": "complete",
        "cycle_count": len(equity_curve),
        "first_execution_date": equity_curve[0]["execution_date"],
        "last_execution_date": equity_curve[-1]["execution_date"],
        "broker_api_used": False,
        "future_function": False,
    }
    return state, equity_curve


def _load_sequence(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ReplayError("replay input must be a JSON array")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic multi-day paper replay")
    parser.add_argument("--input", required=True, help="JSON array of plan/snapshot pairs")
    parser.add_argument("--output", required=True, help="output JSON report")
    parser.add_argument("--state", default=None, help="optional starting paper state")
    args = parser.parse_args()

    cycles = _load_sequence(Path(args.input))
    initial = (
        json.loads(Path(args.state).read_text(encoding="utf-8"))
        if args.state
        else None
    )
    final_state, curve = run_replay(cycles, initial)
    payload = {
        "schema_version": 1,
        "mode": "paper_replay",
        "status": "complete",
        "equity_curve": curve,
        "final_state": final_state,
        "audit": {
            "broker_api_used": False,
            "live_order_submission": False,
            "future_function": False,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "cycle_count": len(curve),
        "final_equity": curve[-1]["equity"],
        "broker_api_used": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
