"""Fail-closed next-open execution gate for paper trading.

This module never submits orders to a broker. It converts an execution plan plus
an actual next-open market snapshot into a paper-execution decision.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from decimal import Decimal, InvalidOperation
from typing import Any


REQUIRED_MARKET_FIELDS = {
    "symbol",
    "open",
    "high_limit",
    "low_limit",
    "is_paused",
    "is_st",
}


class GateError(ValueError):
    """A safety/validation condition prevents paper order release."""


@dataclass(frozen=True)
class PaperDecision:
    order_id: str
    symbol: str
    side: str
    shares: int
    reference_price: float
    notional: float
    execution_date: str | None
    status: str
    reason: str


def _decimal(value: Any, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise GateError(f"invalid {field}: {value!r}") from exc
    if not result.is_finite():
        raise GateError(f"non-finite {field}: {value!r}")
    return result


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise GateError(f"{field} must be an integer")
    try:
        dec = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise GateError(f"invalid {field}: {value!r}") from exc
    if not dec.is_finite() or dec != dec.to_integral_value():
        raise GateError(f"{field} must be an integer: {value!r}")
    return int(dec)


def validate_market_snapshot(snapshot: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_MARKET_FIELDS - snapshot.keys())
    if missing:
        raise GateError(f"missing market fields: {', '.join(missing)}")

    if bool(snapshot["is_paused"]):
        raise GateError("paused security")
    if bool(snapshot["is_st"]):
        raise GateError("ST security")

    opening = _decimal(snapshot["open"], "open")
    high_limit = _decimal(snapshot["high_limit"], "high_limit")
    low_limit = _decimal(snapshot["low_limit"], "low_limit")

    if opening <= 0:
        raise GateError("open must be positive")
    if high_limit <= 0 or low_limit <= 0:
        raise GateError("price limits must be positive")
    if low_limit > high_limit:
        raise GateError("low_limit exceeds high_limit")


def build_paper_decisions(
    execution_plan: dict[str, Any],
    market_by_symbol: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate and release only paper orders; fail closed on missing data."""
    if execution_plan.get("status") != "ready_for_next_open_recheck":
        raise GateError("execution plan is not awaiting next-open recheck")

    orders = execution_plan.get("orders", [])
    if not isinstance(orders, list):
        raise GateError("execution_plan.orders must be a list")

    turnover = _decimal(execution_plan["summary"]["turnover"], "turnover")
    turnover_cap = _decimal(execution_plan["turnover_cap"], "turnover_cap")
    if turnover > turnover_cap:
        raise GateError("planned turnover exceeds cap")

    lot_size = _strict_int(execution_plan["lot_size"], "lot_size")
    if lot_size <= 0:
        raise GateError("lot_size must be positive")

    decisions: list[PaperDecision] = []
    seen_order_ids: set[str] = set()
    for index, order in enumerate(orders, start=1):
        symbol = str(order["symbol"]).zfill(6)
        snapshot = market_by_symbol.get(symbol)
        if snapshot is None:
            raise GateError(f"missing next-open snapshot: {symbol}")

        validate_market_snapshot(snapshot)
        snapshot_symbol = str(snapshot["symbol"]).zfill(6)
        if snapshot_symbol != symbol:
            raise GateError(f"{symbol}: snapshot symbol mismatch")

        side = str(order["side"]).lower()
        if side not in {"buy", "sell"}:
            raise GateError(f"unsupported side: {side}")

        opening = _decimal(snapshot["open"], "open")
        high_limit = _decimal(snapshot["high_limit"], "high_limit")
        low_limit = _decimal(snapshot["low_limit"], "low_limit")

        if side == "buy" and opening >= high_limit:
            raise GateError(f"{symbol}: buy blocked at upper limit")
        if side == "sell" and opening <= low_limit:
            raise GateError(f"{symbol}: sell blocked at lower limit")

        shares = _strict_int(order["shares"], f"{symbol}: shares")
        if shares <= 0 or shares % lot_size != 0:
            raise GateError(f"{symbol}: invalid lot size")

        order_id = str(order.get("order_id") or f"{execution_plan['reference_date']}:{symbol}:{side}:{index}")
        if order_id in seen_order_ids:
            raise GateError(f"duplicate order_id: {order_id}")
        seen_order_ids.add(order_id)

        execution_date = snapshot.get("date")
        if execution_date is not None:
            execution_date = str(execution_date)
            if execution_date <= str(execution_plan["reference_date"]):
                raise GateError(
                    f"{symbol}: next-open snapshot date must be after plan reference date"
                )
        if bool(order.get("blocked", False)):
            raise GateError(f"{symbol}: blocked order cannot be released")
        if "requested_shares" in order:
            requested = _strict_int(order["requested_shares"], f"{symbol}: requested_shares")
            if shares > requested:
                raise GateError(f"{symbol}: executable shares exceed requested shares")

        notional = opening * shares
        decisions.append(
            PaperDecision(
                order_id=order_id,
                symbol=symbol,
                side=side,
                shares=shares,
                reference_price=float(opening),
                notional=float(notional),
                execution_date=execution_date,
                status="paper_released",
                reason="next-open safety gate passed",
            )
        )

    return {
        "schema_version": 1,
        "mode": "paper",
        "status": "paper_released",
        "plan_reference_date": execution_plan["reference_date"],
        "strategy_version": execution_plan["strategy_version"],
        "strategy_commit": execution_plan["strategy_commit"],
        "orders": [asdict(item) for item in decisions],
        "broker_submission": False,
        "audit": {
            "fail_closed": True,
            "next_open_snapshot_required": True,
            "broker_api_used": False,
            "future_function": False,
        },
    }
