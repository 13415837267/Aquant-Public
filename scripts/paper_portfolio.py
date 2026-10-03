"""Paper portfolio state and fill-ledger accounting.

The accounting layer is deliberately broker-free. It applies only already
validated paper fills, enforces cash/T+1 constraints, and keeps an idempotent
fill record so repeated workflow runs cannot double-apply a fill.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any


class PortfolioError(ValueError):
    """Paper accounting rejected an unsafe or inconsistent state transition."""


def _decimal(value: Any, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise PortfolioError(f"invalid {field}: {value!r}") from exc
    if not result.is_finite():
        raise PortfolioError(f"non-finite {field}: {value!r}")
    return result


def _shares(value: Any, field: str) -> int:
    dec = _decimal(value, field)
    if dec != dec.to_integral_value() or dec < 0:
        raise PortfolioError(f"{field} must be a non-negative integer")
    return int(dec)


def new_paper_state(
    *,
    initial_cash: float,
    as_of: str,
    strategy_version: str,
    strategy_commit: str,
    cash_floor: float,
) -> dict[str, Any]:
    cash = _decimal(initial_cash, "initial_cash")
    floor = _decimal(cash_floor, "cash_floor")
    if cash < 0 or floor < 0 or floor > cash:
        raise PortfolioError("cash floor must be between zero and initial cash")
    return {
        "schema_version": 1,
        "mode": "paper",
        "status": "ready",
        "as_of": as_of,
        "initial_cash": float(cash),
        "cash": float(cash),
        "cash_floor": float(floor),
        "positions": {},
        "realized_pnl": 0.0,
        "pending_settlements": [],
        "applied_fills": {},
        "strategy_version": strategy_version,
        "strategy_commit": strategy_commit,
        "audit": {
            "broker_api_used": False,
            "live_orders_submitted": False,
            "t_plus_1_sell_inventory_enforced": True,
            "idempotent_fill_application": True,
            "future_function": False,
        },
    }


def _fill_signature(fill: dict[str, Any]) -> dict[str, Any]:
    return {
        "order_id": str(fill["order_id"]),
        "symbol": str(fill["symbol"]).zfill(6),
        "side": str(fill["side"]).lower(),
        "shares": _shares(fill["shares"], "fill.shares"),
        "price": float(_decimal(fill["price"], "fill.price")),
        "execution_date": str(fill["execution_date"]),
        "settlement_date": str(fill["settlement_date"]),
    }


def _settle_due(state: dict[str, Any], execution_date: str) -> None:
    remaining = []
    for item in state.get("pending_settlements", []):
        symbol = str(item["symbol"]).zfill(6)
        if str(item["settlement_date"]) <= execution_date:
            position = state["positions"].setdefault(
                symbol,
                {
                    "name": item.get("name", symbol),
                    "shares": 0,
                    "available_shares": 0,
                    "avg_cost": 0.0,
                    "market_price": None,
                    "market_value": 0.0,
                    "unrealized_pnl": 0.0,
                    "realized_pnl": 0.0,
                },
            )
            position["available_shares"] += int(item["shares"])
        else:
            remaining.append(item)
    state["pending_settlements"] = remaining


def _recalculate_position(position: dict[str, Any], market_price: float | None = None) -> None:
    shares = int(position["shares"])
    if market_price is not None:
        price = float(_decimal(market_price, "market_price"))
        if price <= 0:
            raise PortfolioError("market price must be positive")
        position["market_price"] = price
    price = position.get("market_price")
    if shares <= 0:
        position["market_value"] = 0.0
        position["unrealized_pnl"] = 0.0
        return
    if price is not None:
        position["market_value"] = round(shares * float(price), 6)
        position["unrealized_pnl"] = round(
            shares * (float(price) - float(position["avg_cost"])), 6
        )
    else:
        position["market_value"] = 0.0
        position["unrealized_pnl"] = 0.0


def apply_paper_fills(
    state: dict[str, Any],
    fills: list[dict[str, Any]],
    *,
    execution_date: str,
    lot_size: int = 100,
    commission_bps: float = 3.0,
    stamp_duty_sell_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply paper fills atomically and return (new_state, new_ledger_entries)."""
    if state.get("mode") != "paper":
        raise PortfolioError("state mode must be paper")
    if not isinstance(state.get("applied_fills", {}), dict):
        raise PortfolioError("applied_fills must be an object")
    if lot_size <= 0:
        raise PortfolioError("lot_size must be positive")
    if any(x < 0 for x in (commission_bps, stamp_duty_sell_bps, slippage_bps)):
        raise PortfolioError("fee assumptions must be non-negative")

    next_state = deepcopy(state)
    _settle_due(next_state, str(execution_date))
    ledger_entries: list[dict[str, Any]] = []

    for raw in fills:
        signature = _fill_signature(raw)
        fill_id = str(raw["fill_id"])
        previous = next_state["applied_fills"].get(fill_id)
        if previous is not None:
            previous_signature = (
                previous.get("signature", previous)
                if isinstance(previous, dict)
                else previous
            )
            if previous_signature != signature:
                raise PortfolioError(f"fill_id reused with different payload: {fill_id}")
            continue

        symbol = signature["symbol"]
        side = signature["side"]
        shares = signature["shares"]
        price = signature["price"]
        if side not in {"buy", "sell"}:
            raise PortfolioError(f"unsupported fill side: {side}")
        if shares <= 0 or shares % lot_size != 0:
            raise PortfolioError(f"{symbol}: invalid lot size in fill")
        if price <= 0:
            raise PortfolioError(f"{symbol}: fill price must be positive")
        if signature["execution_date"] != str(execution_date):
            raise PortfolioError(f"{symbol}: execution date mismatch")
        if signature["settlement_date"] <= signature["execution_date"]:
            raise PortfolioError(f"{symbol}: settlement date must be after execution date")

        position = next_state["positions"].setdefault(
            symbol,
            {
                "name": str(raw.get("name", symbol)),
                "shares": 0,
                "available_shares": 0,
                "avg_cost": 0.0,
                "market_price": price,
                "market_value": 0.0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
            },
        )
        if raw.get("name"):
            position["name"] = str(raw["name"])

        gross = Decimal(shares) * Decimal(str(price))
        commission = gross * Decimal(str(commission_bps)) / Decimal("10000")
        slippage = gross * Decimal(str(slippage_bps)) / Decimal("10000")
        stamp = (
            gross * Decimal(str(stamp_duty_sell_bps)) / Decimal("10000")
            if side == "sell"
            else Decimal("0")
        )

        if side == "buy":
            total_debit = gross + commission + slippage
            projected_cash = Decimal(str(next_state["cash"])) - total_debit
            if projected_cash < Decimal(str(next_state["cash_floor"])) - Decimal("1e-8"):
                raise PortfolioError(f"{symbol}: cash floor would be violated")

            old_shares = int(position["shares"])
            old_cost = Decimal(str(position["avg_cost"]))
            total_cost_basis = old_cost * old_shares + total_debit
            new_shares = old_shares + shares
            position["shares"] = new_shares
            position["available_shares"] = int(position["available_shares"])
            position["avg_cost"] = float(total_cost_basis / Decimal(new_shares))
            next_state["cash"] = float(projected_cash)
            next_state["pending_settlements"].append(
                {
                    "fill_id": fill_id,
                    "symbol": symbol,
                    "name": position["name"],
                    "shares": shares,
                    "execution_date": signature["execution_date"],
                    "settlement_date": signature["settlement_date"],
                }
            )
            realized = Decimal("0")
            cash_delta = -total_debit
        else:
            available = int(position["available_shares"])
            if available < shares:
                raise PortfolioError(
                    f"{symbol}: insufficient T+1 sellable shares ({available} < {shares})"
                )
            old_shares = int(position["shares"])
            old_cost = Decimal(str(position["avg_cost"]))
            proceeds_net = gross - commission - stamp - slippage
            realized = proceeds_net - old_cost * shares
            position["shares"] = old_shares - shares
            position["available_shares"] = available - shares
            next_state["cash"] = float(Decimal(str(next_state["cash"])) + proceeds_net)
            position["realized_pnl"] = float(
                Decimal(str(position.get("realized_pnl", 0.0))) + realized
            )
            cash_delta = proceeds_net
            if position["shares"] == 0:
                position["avg_cost"] = 0.0
                position["market_price"] = None
                position["market_value"] = 0.0
                position["unrealized_pnl"] = 0.0

        _recalculate_position(position, price)
        if position["shares"] == 0:
            next_state["positions"].pop(symbol, None)

        next_state["realized_pnl"] = float(
            Decimal(str(next_state["realized_pnl"])) + realized
        )
        ledger_entry = {
            **signature,
            "fill_id": fill_id,
            "name": position.get("name", raw.get("name", symbol)),
            "gross_notional": round(float(gross), 6),
            "commission": round(float(commission), 6),
            "stamp_duty": round(float(stamp), 6),
            "slippage": round(float(slippage), 6),
            "cash_delta": round(float(cash_delta), 6),
            "realized_pnl": round(float(realized), 6),
        }
        next_state["applied_fills"][fill_id] = {
            "signature": signature,
            "ledger": ledger_entry,
        }
        ledger_entries.append(ledger_entry)

    next_state["as_of"] = str(execution_date)
    next_state["status"] = "ready"
    return next_state, ledger_entries


def mark_to_market(
    state: dict[str, Any],
    market_by_symbol: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Update paper market values from a supplied snapshot without changing cash."""
    next_state = deepcopy(state)
    total_market_value = 0.0
    total_unrealized = 0.0
    for symbol, position in next_state["positions"].items():
        snapshot = market_by_symbol.get(symbol)
        if snapshot is None:
            raise PortfolioError(f"missing valuation snapshot: {symbol}")
        price = _decimal(snapshot["open"], f"{symbol}.open")
        if price <= 0:
            raise PortfolioError(f"{symbol}: valuation price must be positive")
        _recalculate_position(position, float(price))
        total_market_value += float(position["market_value"])
        total_unrealized += float(position["unrealized_pnl"])
    next_state["market_value"] = round(total_market_value, 6)
    next_state["equity"] = round(float(next_state["cash"]) + total_market_value, 6)
    next_state["unrealized_pnl"] = round(total_unrealized, 6)
    return next_state
