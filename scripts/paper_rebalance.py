"""Build the next paper execution plan from the current paper portfolio.

This is the bridge from one paper cycle to the next:
paper portfolio state -> mark at latest close -> dynamic equity/cash/holdings
-> normal execution-plan engine. It never creates or submits live orders.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from scripts.execution_plan import build_plan, read_market
from scripts.paper_portfolio import PortfolioError, apply_paper_fills


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORTFOLIO = ROOT / "data" / "portfolio.json"
DEFAULT_STATE = ROOT / "data" / "paper" / "portfolio.json"
DEFAULT_MARKET = None
DEFAULT_OUTPUT = ROOT / "data" / "paper" / "execution_plan.json"


def derive_paper_inputs(
    state: dict[str, Any],
    market,
) -> tuple[dict[str, dict[str, int]], float, dict[str, Any]]:
    if state.get("mode") != "paper":
        raise PortfolioError("paper rebalance requires a paper portfolio state")
    if state.get("audit", {}).get("broker_api_used") is not False:
        raise PortfolioError("paper state broker boundary is not clean")

    ref_date = str(market["date"].dropna().iloc[0])
    state_as_of = str(state.get("as_of", ""))
    if not state_as_of or ref_date < state_as_of:
        raise PortfolioError("market reference date is older than paper state")

    # Release any T+1 inventory whose settlement date is now due before sizing
    # the next plan. No new fills are applied by this call.
    normalized_state, _ = apply_paper_fills(
        state,
        [],
        execution_date=ref_date,
    )

    holdings: dict[str, dict[str, int]] = {}
    market_value = 0.0
    for symbol, position in normalized_state.get("positions", {}).items():
        symbol = str(symbol).zfill(6)
        shares = int(position.get("shares", 0))
        available = int(position.get("available_shares", 0))
        if shares <= 0:
            continue
        if available < 0 or available > shares:
            raise PortfolioError(f"invalid paper inventory for {symbol}")
        if symbol not in market.index:
            raise PortfolioError(f"missing market close for paper holding: {symbol}")
        close = float(market.loc[symbol, "close"])
        if close <= 0:
            raise PortfolioError(f"invalid market close for paper holding: {symbol}")
        market_value += shares * close
        holdings[symbol] = {
            "shares": shares,
            "available_shares": available,
        }

    cash = float(normalized_state["cash"])
    if cash < 0:
        raise PortfolioError("paper cash cannot be negative")
    equity = cash + market_value
    if equity <= 0:
        raise PortfolioError("paper equity must be positive")

    provenance = {
        "mode": "paper",
        "reference_date": ref_date,
        "state_as_of_before_settlement": state_as_of,
        "state_as_of_after_settlement": normalized_state["as_of"],
        "cash": round(cash, 6),
        "market_value": round(market_value, 6),
        "equity": round(equity, 6),
        "position_count": len(holdings),
        "broker_api_used": False,
        "future_function": False,
    }
    return holdings, equity, provenance


def _write_temp_holdings(holdings: dict[str, dict[str, int]]) -> Path:
    fh = NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False)
    try:
        json.dump({"holdings": holdings}, fh, ensure_ascii=False)
        fh.write("\n")
        fh.close()
        return Path(fh.name)
    except Exception:
        fh.close()
        os.unlink(fh.name)
        raise


def build_paper_plan(
    portfolio: dict[str, Any],
    state: dict[str, Any],
    market,
    *,
    turnover_cap: float = 0.30,
    min_notional: float = 1000.0,
    commission_bps: float = 3.0,
    stamp_duty_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> dict[str, Any]:
    if state.get("strategy_version") != portfolio.get("strategy_version"):
        raise PortfolioError("paper state strategy version does not match target portfolio")
    if state.get("strategy_commit") != portfolio.get("strategy_commit"):
        raise PortfolioError("paper state strategy commit does not match target portfolio")

    holdings, equity, provenance = derive_paper_inputs(state, market)
    temp_path = _write_temp_holdings(holdings)
    try:
        payload = build_plan(
            portfolio,
            market,
            temp_path,
            equity=equity,
            turnover_cap=turnover_cap,
            min_notional=min_notional,
            commission_bps=commission_bps,
            stamp_duty_bps=stamp_duty_bps,
            slippage_bps=slippage_bps,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    payload["mode"] = "paper"
    payload["paper_provenance"] = provenance
    payload["audit"]["broker_api_used"] = False
    payload["audit"]["live_order_submission"] = False
    payload["audit"]["future_function"] = False
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build next paper execution plan")
    parser.add_argument("--portfolio", default=str(DEFAULT_PORTFOLIO))
    parser.add_argument("--state", default=str(DEFAULT_STATE))
    parser.add_argument("--market-file", default=DEFAULT_MARKET)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--turnover-cap", type=float, default=0.30)
    parser.add_argument("--min-notional", type=float, default=1000.0)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--stamp-duty-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    args = parser.parse_args()

    market_file = Path(args.market_file) if args.market_file else None
    if market_file is None:
        from scripts.execution_plan import latest_history_file
        market_file = latest_history_file()

    portfolio = json.loads(Path(args.portfolio).read_text(encoding="utf-8"))
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    market = read_market(market_file)
    payload = build_paper_plan(
        portfolio,
        state,
        market,
        turnover_cap=args.turnover_cap,
        min_notional=args.min_notional,
        commission_bps=args.commission_bps,
        stamp_duty_bps=args.stamp_duty_bps,
        slippage_bps=args.slippage_bps,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "mode": payload["mode"],
        "reference_date": payload["reference_date"],
        "equity": payload["paper_provenance"]["equity"],
        "cash": payload["paper_provenance"]["cash"],
        "position_count": payload["paper_provenance"]["position_count"],
        "order_count": payload["summary"]["order_count"],
        "broker_api_used": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
