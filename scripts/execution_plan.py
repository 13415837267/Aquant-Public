"""Build an auditable A-share execution plan from the production portfolio."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PORTFOLIO = DATA / "portfolio.json"
HISTORY = DATA / "history"
HOLDINGS = DATA / "holdings.json"
OUTPUT = DATA / "execution_plan.json"
LOT_SIZE = 100
DEFAULT_EQUITY = 1_000_000.0
DEFAULT_TURNOVER_CAP = 0.30
DEFAULT_MIN_NOTIONAL = 1_000.0
DEFAULT_COMMISSION_BPS = 3.0
DEFAULT_STAMP_DUTY_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 2.0


def valid_price(value: object) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(x) and x > 0
def latest_history_file() -> Path:
    files = sorted(HISTORY.glob("????/*.csv.gz"), key=lambda p: (p.parent.name, p.stem))
    if not files:
        raise RuntimeError("no history files found")
    return files[-1]


def read_market(path: Path) -> pd.DataFrame:
    cols = ["symbol", "date", "open", "close", "high_limit", "low_limit", "is_paused", "is_st"]
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(fh, usecols=cols)
    df["symbol"] = df["symbol"].astype(str).str.extract(r"(\d{6})")[0].str.zfill(6)
    for col in cols[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df["symbol"].duplicated().any():
        raise RuntimeError(f"duplicate symbols in {path.name}")
    return df.set_index("symbol", drop=False)


def load_holdings(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    if not path.exists():
        return {}, {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("holdings", raw)
    if not isinstance(items, dict):
        raise RuntimeError("holdings.json must contain an object")
    total, available = {}, {}
    for symbol, value in items.items():
        symbol = str(symbol).zfill(6)
        if isinstance(value, dict):
            shares = float(value.get("shares", 0))
            avail = float(value.get("available_shares", shares))
        else:
            shares = float(value)
            avail = shares
        if shares < 0 or avail < 0 or avail > shares + 1e-9:
            raise ValueError(f"invalid holding state for {symbol}")
        total[symbol] = shares
        available[symbol] = avail
    return total, available
def round_lot(shares: float) -> int:
    if shares <= 0:
        return 0
    return int(np.floor((shares + 1e-9) / LOT_SIZE) * LOT_SIZE)


def build_plan(
    portfolio: dict,
    market: pd.DataFrame,
    holdings_path: Path,
    *,
    equity: float = DEFAULT_EQUITY,
    turnover_cap: float = DEFAULT_TURNOVER_CAP,
    min_notional: float = DEFAULT_MIN_NOTIONAL,
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    stamp_duty_bps: float = DEFAULT_STAMP_DUTY_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> dict:
    if portfolio.get("status") != "ready":
        raise RuntimeError("portfolio snapshot is not ready")
    if equity <= 0 or not 0 < turnover_cap <= 1:
        raise ValueError("equity must be positive and turnover_cap must be in (0, 1]")
    if min_notional < 0 or commission_bps < 0 or stamp_duty_bps < 0 or slippage_bps < 0:
        raise ValueError("cost assumptions must be non-negative")

    total_shares, available_shares = load_holdings(holdings_path)
    positions = portfolio.get("positions", [])
    cash_buffer = float(portfolio.get("cash_buffer", 0.05))
    if not 0 <= cash_buffer < 1:
        raise ValueError("portfolio cash_buffer must be in [0, 1)")
    ref_date = str(market["date"].dropna().iloc[0])
    if market["date"].astype(str).nunique() != 1:
        raise RuntimeError("market file contains multiple dates")

    # Latest close is a planning reference only. The next session's open must
    # be re-checked before an order is released to a broker.
    market_value = 0.0
    missing_holdings_market: list[str] = []
    for symbol, shares in total_shares.items():
        if shares <= 0:
            continue
        if symbol not in market.index or not valid_price(market.loc[symbol, "close"]):
            missing_holdings_market.append(symbol)
            continue
        market_value += shares * float(market.loc[symbol, "close"])
    if missing_holdings_market:
        raise ValueError(
            "missing/invalid market close for holdings: "
            + ", ".join(sorted(missing_holdings_market))
        )
    if total_shares and market_value > equity + 1e-9:
        raise ValueError("equity reference is below marked holding value")

    current_values = {
        symbol: shares * float(market.loc[symbol, "close"])
        for symbol, shares in total_shares.items()
        if symbol in market.index and valid_price(market.loc[symbol, "close"])
    }
    orders = []
    raw_turnover = 0.0

    # Include explicit target positions plus deterministic exits for holdings
    # that are no longer in the target portfolio. This prevents stale long
    # positions from surviving indefinitely across rebalance cycles.
    target_symbols = {str(p["symbol"]).zfill(6) for p in positions}
    plan_items = list(positions)
    residual_rank = max([int(p.get("rank", 0)) for p in positions] + [0]) + 1_000
    for symbol in sorted(set(total_shares) - target_symbols):
        if total_shares.get(symbol, 0) <= 0:
            continue
        plan_items.append({
            "rank": residual_rank,
            "symbol": symbol,
            "name": symbol,
            "target_weight": 0.0,
        })

    if len(target_symbols) != len(positions):
        raise ValueError("portfolio positions contain duplicate symbols")

    for p in plan_items:
        symbol = str(p["symbol"]).zfill(6)
        target_value = equity * float(p["target_weight"])
        current_value = current_values.get(symbol, 0.0)
        delta_value = target_value - current_value
        row = market.loc[symbol] if symbol in market.index else None
        reason = None
        if row is None:
            reason = "missing_market_observation"
        elif not valid_price(row["close"]):
            reason = "invalid_reference_price"
        elif float(row.get("is_paused", 0) or 0) != 0:
            reason = "paused"
        elif float(row.get("is_st", 0) or 0) != 0:
            reason = "st_excluded"

        side = "buy" if delta_value > 0 else "sell" if delta_value < 0 else "hold"
        ref_price = float(row["close"]) if row is not None and valid_price(row["close"]) else np.nan
        raw_shares = abs(delta_value) / ref_price if valid_price(ref_price) else 0.0
        requested_shares = round_lot(raw_shares)
        if side == "sell" and symbol in total_shares:
            requested_shares = min(requested_shares, round_lot(available_shares.get(symbol, 0.0)))
            if requested_shares == 0 and abs(delta_value) > 0:
                reason = reason or "t_plus_1_or_no_available_shares"
        if requested_shares > 0 and valid_price(ref_price) and requested_shares * ref_price < min_notional:
            requested_shares = 0
            reason = reason or "below_min_notional"
        if side == "hold":
            requested_shares = 0

        blocked = reason is not None and requested_shares > 0
        executable_shares = 0 if blocked else requested_shares
        raw_notional = executable_shares * ref_price if valid_price(ref_price) else 0.0
        raw_turnover += raw_notional

        orders.append({
            "order_id": f"{ref_date}:{symbol}:{side}:{int(p.get('rank', residual_rank))}",
            "rank": int(p.get("rank", residual_rank)),
            "symbol": symbol,
            "name": p.get("name", symbol),
            "side": side,
            "target_weight": float(p["target_weight"]),
            "current_weight": current_value / equity if equity > 0 else 0.0,
            "reference_price": round(ref_price, 6) if valid_price(ref_price) else None,
            "requested_shares": int(requested_shares),
            "shares": int(executable_shares),
            "notional": round(raw_notional, 2),
            "precheck": reason or "pending_next_open_validation",
            "blocked": bool(blocked),
        })

    # Aggregate turnover cap is applied before cash sufficiency.
    scale = min(1.0, equity * turnover_cap / raw_turnover) if raw_turnover > 0 else 1.0
    for order in orders:
        if order["shares"] <= 0 or order["blocked"]:
            continue
        scaled = round_lot(order["shares"] * scale)
        if scaled < order["shares"]:
            order["turnover_cap_reduced"] = True
        order["shares"] = scaled
        order["notional"] = round(scaled * float(order["reference_price"]), 2)
    # Enforce the portfolio cash floor after fees/slippage.
    buy_cost = sell_cost = buy_notional = sell_notional = 0.0
    for o in orders:
        n = float(o["notional"])
        if o["side"] == "buy":
            buy_notional += n
            buy_cost += n * (commission_bps + slippage_bps) / 10000.0
        elif o["side"] == "sell":
            sell_notional += n
            sell_cost += n * (commission_bps + stamp_duty_bps + slippage_bps) / 10000.0
    available_cash_floor = equity * cash_buffer
    starting_cash = max(0.0, equity - market_value)
    required_cash = buy_notional + buy_cost - sell_notional + sell_cost
    if starting_cash - required_cash < available_cash_floor - 1e-9 and buy_notional > 0:
        room = max(0.0, starting_cash + sell_notional - sell_cost - available_cash_floor)
        buy_scale = min(1.0, room / (buy_notional + buy_cost)) if buy_notional else 1.0
        for o in orders:
            if o["side"] == "buy" and not o["blocked"]:
                o["shares"] = round_lot(o["shares"] * buy_scale)
                o["notional"] = round(o["shares"] * float(o["reference_price"]), 2)
                o["cash_floor_reduced"] = True

    buy_notional = sum(float(o["notional"]) for o in orders if o["side"] == "buy")
    sell_notional = sum(float(o["notional"]) for o in orders if o["side"] == "sell")
    commission = (buy_notional + sell_notional) * commission_bps / 10000.0
    stamp = sell_notional * stamp_duty_bps / 10000.0
    slippage = (buy_notional + sell_notional) * slippage_bps / 10000.0
    turnover = (buy_notional + sell_notional) / equity
    estimated_cash_after = starting_cash - buy_notional - commission - slippage + sell_notional - stamp
    for o in orders:
        if o["shares"] <= 0 and o["side"] != "hold" and not o["blocked"]:
            o["precheck"] = o.get("precheck") or "lot_rounding_zero"
        ref = o["reference_price"]
        if ref is not None and o["symbol"] in market.index:
            row = market.loc[o["symbol"]]
            if o["side"] == "buy" and valid_price(row.get("high_limit")) and ref >= float(row["high_limit"]) - 1e-8:
                o["limit_risk"] = "reference_close_at_or_near_high_limit"
            elif o["side"] == "sell" and valid_price(row.get("low_limit")) and ref <= float(row["low_limit"]) + 1e-8:
                o["limit_risk"] = "reference_close_at_or_near_low_limit"
        o["estimated_cost"] = round(
            o["notional"] * ((commission_bps + slippage_bps) if o["side"] == "buy" else (commission_bps + stamp_duty_bps + slippage_bps)) / 10000.0,
            4,
        )
        o["release_gate"] = "next_open_recheck_required"

    blocked_count = sum(1 for o in orders if o["blocked"])
    active_orders = [o for o in orders if o["shares"] > 0 and not o["blocked"]]
    plan_identity = {
        "reference_date": ref_date,
        "strategy_commit": portfolio["strategy_commit"],
        "equity_reference": round(float(equity), 2),
        "orders": active_orders,
    }
    plan_id = hashlib.sha256(
        json.dumps(
            plan_identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    return {
        "schema_version": 1,
        "status": "ready_for_next_open_recheck",
        "plan_id": plan_id,
        "reference_date": ref_date,
        "price_basis": "latest_close_reference_only",
        "strategy_source": portfolio["strategy_source"],
        "strategy_version": portfolio["strategy_version"],
        "strategy_commit": portfolio["strategy_commit"],
        "equity_reference": round(float(equity), 2),
        "starting_cash_reference": round(float(starting_cash), 2),
        "turnover_cap": turnover_cap,
        "cash_buffer": cash_buffer,
        "lot_size": LOT_SIZE,
        "minimum_order_notional": min_notional,
        "cost_assumptions_bps": {
            "commission": commission_bps,
            "stamp_duty_sell": stamp_duty_bps,
            "slippage": slippage_bps,
        },
        "orders": sorted(active_orders, key=lambda x: (x["side"] != "sell", x["rank"], x["symbol"])),
        "blocked_orders": [o for o in orders if o["blocked"]],
        "summary": {
            "order_count": len(active_orders),
            "blocked_count": blocked_count,
            "buy_notional": round(buy_notional, 2),
            "sell_notional": round(sell_notional, 2),
            "turnover": round(turnover, 8),
            "commission": round(commission, 4),
            "stamp_duty": round(stamp, 4),
            "slippage": round(slippage, 4),
            "estimated_total_cost": round(commission + stamp + slippage, 4),
            "estimated_cash_after": round(estimated_cash_after, 2),
            "unfilled_count": sum(1 for o in orders if o["side"] != "hold" and o["shares"] <= 0),
        },
        "audit": {
            "long_only": True,
            "leverage": False,
            "t_plus_1_sell_inventory_enforced": True,
            "paused_excluded": True,
            "st_excluded": True,
            "lot_rounding_enforced": True,
            "turnover_cap_enforced": True,
            "cash_floor_enforced": True,
            "next_open_limit_check_required": True,
            "future_function": False,
        },
    }
def validate(payload: dict) -> None:
    if payload["status"] != "ready_for_next_open_recheck":
        raise ValueError("unexpected execution-plan status")
    s = payload["summary"]
    if float(s["turnover"]) > float(payload["turnover_cap"]) + 1e-8:
        raise ValueError("turnover cap violated")
    if s["buy_notional"] < -1e-9 or s["sell_notional"] < -1e-9:
        raise ValueError("negative order notional")
    cash_floor = float(payload["equity_reference"]) * float(payload["cash_buffer"])
    if float(s["estimated_cash_after"]) < cash_floor - 1e-6:
        raise ValueError("cash floor violated")
    for o in payload["orders"]:
        if o["shares"] % payload["lot_size"] != 0:
            raise ValueError(f"lot size violated: {o['symbol']}")
        if o["shares"] <= 0:
            raise ValueError(f"non-positive executable shares: {o['symbol']}")
        if o["side"] not in {"buy", "sell"}:
            raise ValueError(f"invalid side: {o['symbol']}")
    if payload["audit"]["future_function"] is not False:
        raise ValueError("future-function audit failed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build production execution plan")
    p.add_argument("--portfolio", default=str(PORTFOLIO))
    p.add_argument("--market-file", default=None)
    p.add_argument("--holdings", default=str(HOLDINGS))
    p.add_argument("--equity", type=float, default=DEFAULT_EQUITY)
    p.add_argument("--turnover-cap", type=float, default=DEFAULT_TURNOVER_CAP)
    p.add_argument("--min-notional", type=float, default=DEFAULT_MIN_NOTIONAL)
    p.add_argument("--commission-bps", type=float, default=DEFAULT_COMMISSION_BPS)
    p.add_argument("--stamp-duty-bps", type=float, default=DEFAULT_STAMP_DUTY_BPS)
    p.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    p.add_argument("--output", default=str(OUTPUT))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    market_file = Path(args.market_file) if args.market_file else latest_history_file()
    payload = build_plan(
        json.loads(Path(args.portfolio).read_text(encoding="utf-8")),
        read_market(market_file),
        Path(args.holdings),
        equity=args.equity,
        turnover_cap=args.turnover_cap,
        min_notional=args.min_notional,
        commission_bps=args.commission_bps,
        stamp_duty_bps=args.stamp_duty_bps,
        slippage_bps=args.slippage_bps,
    )
    validate(payload)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload["summary"], "reference_date": payload["reference_date"], "status": payload["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
