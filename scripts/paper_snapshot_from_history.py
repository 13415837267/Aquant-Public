"""Build a paper next-open snapshot from a committed historical market file.

This utility is for deterministic historical replay/testing only. It never
queries a live market feed and never submits orders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.execution_plan import latest_history_file, read_market


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "data" / "execution_plan.json"
DEFAULT_STATE = ROOT / "data" / "paper" / "portfolio.json"
DEFAULT_OUTPUT = ROOT / "data" / "paper" / "next_open_snapshot.json"


def build_snapshot(
    plan: dict[str, Any],
    market,
    *,
    settlement_date: str,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference_date = str(plan["reference_date"])
    execution_date = str(market["date"].dropna().iloc[0])
    if market["date"].astype(str).nunique() != 1:
        raise ValueError("historical market file contains multiple dates")
    if execution_date <= reference_date:
        raise ValueError("historical execution date must be after plan reference date")
    if not settlement_date or settlement_date <= execution_date:
        raise ValueError("settlement_date must be after execution_date")

    symbols = {
        str(order["symbol"]).zfill(6)
        for order in plan.get("orders", [])
        if int(order.get("shares", 0)) > 0
    }
    if state is not None:
        for symbol, position in state.get("positions", {}).items():
            if int(position.get("shares", 0)) > 0:
                symbols.add(str(symbol).zfill(6))

    if not symbols:
        raise ValueError("no symbols require a next-open snapshot")

    snapshots: dict[str, dict[str, Any]] = {}
    for symbol in sorted(symbols):
        if symbol not in market.index:
            raise ValueError(f"missing historical row for {symbol}")
        row = market.loc[symbol]
        snapshots[symbol] = {
            "symbol": symbol,
            "date": execution_date,
            "open": float(row["open"]),
            "high_limit": float(row["high_limit"]),
            "low_limit": float(row["low_limit"]),
            "is_paused": int(row["is_paused"]),
            "is_st": int(row["is_st"]),
        }

    return {
        "schema_version": 1,
        "mode": "paper_historical_replay",
        "plan_reference_date": reference_date,
        "execution_date": execution_date,
        "settlement_date": settlement_date,
        "source": "committed historical daily market file",
        "symbols": snapshots,
        "audit": {
            "live_market_api_used": False,
            "broker_api_used": False,
            "future_function": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical next-open paper snapshot")
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--state", default=None)
    parser.add_argument("--market-file", default=None)
    parser.add_argument("--settlement-date", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    market_file = Path(args.market_file) if args.market_file else latest_history_file()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    state = json.loads(Path(args.state).read_text(encoding="utf-8")) if args.state else None
    payload = build_snapshot(
        plan,
        read_market(market_file),
        settlement_date=args.settlement_date,
        state=state,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "snapshot_ready",
        "mode": payload["mode"],
        "execution_date": payload["execution_date"],
        "settlement_date": payload["settlement_date"],
        "symbol_count": len(payload["symbols"]),
        "broker_api_used": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
