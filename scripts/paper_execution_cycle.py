"""Run one broker-free paper execution cycle.

Input:
  execution_plan.json
  next-open snapshot JSON: {"execution_date": "...", "settlement_date": "...",
  "symbols": {"600000": {...}}}

Output:
  data/paper/portfolio.json
  data/paper/fills.jsonl

The command fails closed if the next-open gate or portfolio accounting rejects
any order. It never calls a broker API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from scripts.paper_execution_gate import GateError, build_paper_decisions
from scripts.paper_portfolio import PortfolioError, apply_paper_fills, mark_to_market, new_paper_state


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "data" / "execution_plan.json"
DEFAULT_SNAPSHOT = ROOT / "data" / "paper" / "next_open_snapshot.json"
DEFAULT_STATE = ROOT / "data" / "paper" / "portfolio.json"
DEFAULT_LEDGER = ROOT / "data" / "paper" / "fills.jsonl"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing input: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return raw


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        temp = Path(fh.name)
    os.replace(temp, path)


def _ledger_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        fill_id = str(row["fill_id"])
        if fill_id in records and records[fill_id] != row:
            raise RuntimeError(f"duplicate fill_id with conflicting ledger rows: {fill_id}")
        records[fill_id] = row
    return records


def _ledger_ids(path: Path) -> set[str]:
    return set(_ledger_records(path))


def _append_ledger(path: Path, entries: list[dict[str, Any]], existing_ids: set[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as fh:
        for entry in entries:
            fill_id = str(entry["fill_id"])
            if fill_id in existing_ids:
                continue
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            existing_ids.add(fill_id)
            written += 1
    return written


def _snapshot_fingerprint(
    market: dict[str, dict[str, Any]],
    required_symbols: set[str],
) -> str:
    payload = {
        symbol: market[symbol]
        for symbol in sorted(required_symbols)
        if symbol in market
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]


def reconcile_ledger(state: dict[str, Any], ledger_path: Path) -> int:
    """Repair missing audit rows and fail on orphan/conflicting ledger data."""
    records = _ledger_records(ledger_path)
    state_fills = state.get("applied_fills", {})
    for fill_id, row in records.items():
        record = state_fills.get(fill_id)
        if record is None:
            raise RuntimeError(f"ledger contains fill absent from state: {fill_id}")
        ledger = record.get("ledger") if isinstance(record, dict) else None
        if ledger != row:
            raise RuntimeError(f"ledger/state mismatch for fill_id: {fill_id}")

    missing_entries = []
    for fill_id, record in state_fills.items():
        ledger = record.get("ledger") if isinstance(record, dict) else None
        if ledger and fill_id not in records:
            missing_entries.append(ledger)
    return _append_ledger(ledger_path, missing_entries, set(records))


def run_cycle(
    plan: dict[str, Any],
    snapshot_payload: dict[str, Any],
    state: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    execution_date = str(snapshot_payload.get("execution_date", ""))
    settlement_date = str(snapshot_payload.get("settlement_date", ""))
    market = snapshot_payload.get("symbols")
    if not execution_date or not settlement_date:
        raise RuntimeError("execution_date and settlement_date are required")
    if settlement_date <= execution_date:
        raise RuntimeError("settlement_date must be after execution_date")
    if not isinstance(market, dict) or not market:
        raise RuntimeError("symbols must be a non-empty object")

    gate = build_paper_decisions(plan, market)
    decision_dates = {
        str(order["execution_date"])
        for order in gate["orders"]
        if order.get("execution_date") is not None
    }
    if decision_dates and decision_dates != {execution_date}:
        raise RuntimeError("next-open snapshots use inconsistent execution dates")
    if state is None:
        state = new_paper_state(
            initial_cash=float(plan["starting_cash_reference"]),
            as_of=str(plan["reference_date"]),
            strategy_version=str(plan["strategy_version"]),
            strategy_commit=str(plan["strategy_commit"]),
            cash_floor=float(plan["equity_reference"]) * float(plan["cash_buffer"]),
        )
    else:
        if state.get("strategy_version") != plan.get("strategy_version"):
            raise RuntimeError("paper state strategy version does not match plan")
        if state.get("strategy_commit") != plan.get("strategy_commit"):
            raise RuntimeError("paper state strategy commit does not match plan")

    if not isinstance(state.get("applied_plans", {}), dict):
        raise RuntimeError("paper state applied_plans must be an object")

    plan_id = gate["plan_id"]
    required_symbols = {
        str(order["symbol"]).zfill(6)
        for order in gate["orders"]
    } | {
        str(symbol).zfill(6)
        for symbol in state.get("positions", {})
        if int(state["positions"][symbol].get("shares", 0)) > 0
    }
    missing_valuation_inputs = sorted(required_symbols - set(market))
    if missing_valuation_inputs:
        raise RuntimeError(
            "missing next-open snapshots: " + ", ".join(missing_valuation_inputs)
        )

    snapshot_fingerprint = _snapshot_fingerprint(market, required_symbols)
    applied_plan = state["applied_plans"].get(plan_id)
    if applied_plan is not None:
        if str(applied_plan.get("execution_date")) != execution_date:
            raise RuntimeError("execution plan already applied on a different date")
        if str(applied_plan.get("snapshot_fingerprint")) != snapshot_fingerprint:
            raise RuntimeError("execution plan already applied with a different snapshot")
        return deepcopy(state), [], gate

    fills = []
    for order in gate["orders"]:
        fill_id = f"paper:{execution_date}:{order['order_id']}"
        fills.append(
            {
                "fill_id": fill_id,
                "order_id": order["order_id"],
                "symbol": order["symbol"],
                "name": next(
                    (o.get("name", order["symbol"]) for o in plan["orders"] if str(o["symbol"]).zfill(6) == order["symbol"]),
                    order["symbol"],
                ),
                "side": order["side"],
                "shares": order["shares"],
                "price": order["reference_price"],
                "execution_date": execution_date,
                "settlement_date": settlement_date,
            }
        )

    # Keep accounting constraints aligned with the current plan on a working
    # copy so a rejected cycle never mutates the prior persisted state.
    working_state = deepcopy(state)
    working_state["cash_floor"] = float(plan["equity_reference"]) * float(plan["cash_buffer"])

    next_state, ledger_entries = apply_paper_fills(
        working_state,
        fills,
        execution_date=execution_date,
        lot_size=int(plan["lot_size"]),
        commission_bps=float(plan["cost_assumptions_bps"]["commission"]),
        stamp_duty_sell_bps=float(plan["cost_assumptions_bps"]["stamp_duty_sell"]),
        slippage_bps=float(plan["cost_assumptions_bps"]["slippage"]),
    )
    valuation_missing = sorted(set(next_state["positions"]) - set(market))
    if valuation_missing:
        raise RuntimeError(
            "missing next-open valuation snapshots: " + ", ".join(valuation_missing)
        )
    next_state = mark_to_market(next_state, market)
    next_state["plan_reference_date"] = plan["reference_date"]
    next_state["last_gate_status"] = gate["status"]
    next_state["broker_submission"] = False
    next_state["applied_plans"][plan_id] = {
        "execution_date": execution_date,
        "snapshot_fingerprint": snapshot_fingerprint,
    }
    return next_state, ledger_entries, gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run broker-free paper execution cycle")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    args = parser.parse_args()

    try:
        plan = _load(Path(args.plan))
        snapshot = _load(Path(args.snapshot))
        state_path = Path(args.state)
        state = _load(state_path) if state_path.exists() else None
        ledger_path = Path(args.ledger)
        if state is not None:
            reconcile_ledger(state, ledger_path)
        next_state, entries, gate = run_cycle(plan, snapshot, state)
        _atomic_write(state_path, next_state)
        written = _append_ledger(ledger_path, entries, _ledger_ids(ledger_path))
    except (GateError, PortfolioError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"PAPER_EXECUTION_BLOCKED: {exc}") from exc

    print(json.dumps({
        "status": "paper_cycle_complete",
        "execution_date": next_state["as_of"],
        "orders_released": len(gate["orders"]),
        "fills_written": written,
        "cash": round(float(next_state["cash"]), 2),
        "equity": round(float(next_state.get("equity", next_state["cash"])), 2),
        "position_count": len(next_state["positions"]),
        "broker_submission": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
