"""Stateful historical backtest with limit-up/down execution constraints.

This module reuses the canonical point-in-time signal path from backtest.py.
Only the execution/accounting layer changes:
  - long-only cash + share state is carried across sessions;
  - T+1 open is the rebalance point;
  - buy orders are blocked at provider high_limit;
  - sell orders are blocked at provider low_limit;
  - paused/missing-price orders are blocked;
  - blocked holdings remain in the portfolio;
  - overnight gaps are included because positions persist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import backtest as base

ROOT = base.ROOT
OUT_FILE = ROOT / "data" / "backtest" / "execution_constrained.json"
DEFAULT_TOP_N = base.DEFAULT_TOP_N
DEFAULT_COST_BPS = base.DEFAULT_COST_BPS
DEFAULT_SLIPPAGE_BPS = base.DEFAULT_SLIPPAGE_BPS


def valid_price(value: object) -> bool:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    return np.isfinite(value) and value > 0


def row_value(df: pd.DataFrame, symbol: str, column: str, fallback: float = np.nan) -> float:
    if symbol not in df.index:
        return fallback
    value = df.loc[symbol, column]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if np.isfinite(value) else fallback


def blocked_for_side(row: pd.Series, side: str) -> bool:
    paused = row.get("is_paused", 0)
    if pd.notna(paused) and float(paused) != 0:
        return True

    open_price = row.get("open", np.nan)
    if not valid_price(open_price):
        return True

    if side == "buy":
        high_limit = row.get("high_limit", np.nan)
        return pd.notna(high_limit) and valid_price(high_limit) and float(open_price) >= float(high_limit) - 1e-8

    if side == "sell":
        low_limit = row.get("low_limit", np.nan)
        return pd.notna(low_limit) and valid_price(low_limit) and float(open_price) <= float(low_limit) + 1e-8

    raise ValueError(f"unsupported execution side: {side}")


class StatefulPortfolio:
    """Long-only portfolio carried from close to close through T+1 execution."""

    def __init__(self, initial_cash: float = 1.0) -> None:
        self.cash = float(initial_cash)
        self.initial_cash = float(initial_cash)
        self.shares: dict[str, float] = {}
        self.last_close: dict[str, float] = {}
        self.prev_close_equity = float(initial_cash)

    def open_equity(self, execution: pd.DataFrame) -> float:
        value = self.cash
        for symbol, shares in self.shares.items():
            price = row_value(execution, symbol, "open", self.last_close.get(symbol, np.nan))
            if not valid_price(price):
                price = self.last_close.get(symbol, 0.0)
            value += shares * max(float(price), 0.0)
        return float(value)

    def rebalance(
        self,
        targets: pd.DataFrame,
        execution_df: pd.DataFrame,
        cost_rate: float,
    ) -> dict:
        execution = execution_df.set_index("symbol", drop=False)
        target_symbols = [str(x).zfill(6) for x in targets.get("symbol", pd.Series(dtype=str)).tolist()]
        target_weights = base.normalize_weights(target_symbols)

        prior_equity = float(self.prev_close_equity)
        equity_open = self.open_equity(execution)
        if not valid_price(equity_open) or equity_open <= 0:
            equity_open = prior_equity

        open_values: dict[str, float] = {}
        for symbol, shares in self.shares.items():
            price = row_value(execution, symbol, "open", self.last_close.get(symbol, np.nan))
            if not valid_price(price):
                price = self.last_close.get(symbol, 0.0)
            open_values[symbol] = max(float(price), 0.0) * float(shares)

        sell_notional = 0.0
        blocked_sell = 0
        sell_symbols = []

        # First remove overweight/non-target holdings where selling is possible.
        for symbol in list(self.shares):
            current_value = open_values.get(symbol, 0.0)
            desired_value = equity_open * target_weights.get(symbol, 0.0)
            delta = current_value - desired_value
            if delta <= 1e-12:
                continue

            row = execution.loc[symbol] if symbol in execution.index else None
            if row is None or blocked_for_side(row, "sell"):
                blocked_sell += 1
                continue

            open_price = float(row_value(execution, symbol, "open"))
            shares_to_sell = min(self.shares[symbol], delta / open_price)
            notional = shares_to_sell * open_price
            self.shares[symbol] -= shares_to_sell
            if self.shares[symbol] <= 1e-12:
                del self.shares[symbol]
            self.cash += notional * (1.0 - cost_rate)
            sell_notional += notional
            sell_symbols.append(symbol)

        cash_after_sells = float(self.cash)

        # Then fill buy deficits. When cash is insufficient, scale all eligible
        # buys proportionally so the portfolio never creates leverage.
        buy_orders: dict[str, float] = {}
        blocked_buy = 0
        for symbol, weight in target_weights.items():
            current_value = open_values.get(symbol, 0.0)
            desired_value = equity_open * weight
            deficit = desired_value - current_value
            if deficit <= 1e-12:
                continue

            row = execution.loc[symbol] if symbol in execution.index else None
            if row is None or blocked_for_side(row, "buy"):
                blocked_buy += 1
                continue

            buy_orders[symbol] = deficit

        total_requested_buy = float(sum(buy_orders.values()))
        max_affordable_buy = cash_after_sells / (1.0 + cost_rate)
        fill_ratio = 1.0 if total_requested_buy <= max_affordable_buy + 1e-12 else (
            max_affordable_buy / total_requested_buy if total_requested_buy > 0 else 0.0
        )

        buy_notional = 0.0
        buy_symbols = []
        for symbol, requested in buy_orders.items():
            notional = requested * fill_ratio
            row = execution.loc[symbol]
            open_price = float(row_value(execution, symbol, "open"))
            shares = notional / open_price
            self.shares[symbol] = self.shares.get(symbol, 0.0) + shares
            buy_notional += notional
            buy_symbols.append(symbol)

        buy_cost = buy_notional * cost_rate
        self.cash -= buy_notional + buy_cost
        if self.cash < -1e-10:
            raise ValueError("stateful execution produced negative cash")
        self.cash = max(self.cash, 0.0)

        total_traded = sell_notional + buy_notional
        cost = total_traded * cost_rate

        close_equity = self.cash
        close_missing = 0
        for symbol, shares in list(self.shares.items()):
            close_price = row_value(execution, symbol, "close", self.last_close.get(symbol, np.nan))
            if not valid_price(close_price):
                close_missing += 1
                close_price = self.last_close.get(symbol, np.nan)
            if valid_price(close_price):
                self.last_close[symbol] = float(close_price)
                close_equity += shares * float(close_price)

        if not valid_price(close_equity) or close_equity <= 0:
            raise ValueError("stateful execution produced invalid close equity")

        net_return = close_equity / prior_equity - 1.0
        gross_return = net_return + cost

        turnover_value = total_traded / equity_open if equity_open > 0 else 0.0
        overnight_return = equity_open / prior_equity - 1.0

        self.prev_close_equity = float(close_equity)

        return {
            "net_return": float(net_return),
            "gross_return": float(gross_return),
            "turnover": float(turnover_value),
            "cost": float(cost),
            "equity_open": float(equity_open),
            "equity_close": float(close_equity),
            "overnight_return": float(overnight_return),
            "sell_count": len(sell_symbols),
            "buy_count": len(buy_symbols),
            "blocked_buy_count": blocked_buy,
            "blocked_sell_count": blocked_sell,
            "close_missing_count": close_missing,
            "cash": float(self.cash),
            "position_count": len(self.shares),
            "sold_symbols": sell_symbols,
            "bought_symbols": buy_symbols,
        }


def validate_constrained_payload(payload: dict) -> None:
    if payload.get("status") != "ready":
        raise ValueError("constrained payload is not ready")
    if payload.get("future_function") is not False:
        raise ValueError("future_function audit must be false")

    daily = payload.get("daily", [])
    overall = payload.get("overall", {})
    audit = payload.get("audit", {})
    if int(audit.get("performance_sessions", -1)) != len(daily):
        raise ValueError("performance_sessions does not match daily rows")
    if int(overall.get("trading_days", -1)) != len(daily):
        raise ValueError("overall trading_days does not match daily rows")

    dates = [row["date"] for row in daily]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ValueError("daily dates are not strictly increasing")

    for row in daily:
        for key in (
            "gross_return",
            "net_return",
            "turnover",
            "cost",
            "equity_open",
            "equity_close",
            "overnight_return",
            "cash",
        ):
            value = float(row[key])
            if not np.isfinite(value):
                raise ValueError(f"non-finite constrained field: {key}")
        if float(row["equity_close"]) <= 0 or float(row["equity_open"]) <= 0:
            raise ValueError("non-positive portfolio equity")
        if float(row["turnover"]) < -1e-12:
            raise ValueError("negative turnover")
        if float(row["cost"]) < -1e-12:
            raise ValueError("negative cost")

    for key in (
        "total_return_pct",
        "annualized_return_pct",
        "annualized_volatility_pct",
        "max_drawdown_pct",
        "average_turnover_pct",
        "total_turnover_pct",
        "win_rate_pct",
        "best_day_pct",
        "worst_day_pct",
    ):
        if overall.get(key) is not None and not np.isfinite(float(overall[key])):
            raise ValueError(f"non-finite overall metric: {key}")
    if not payload.get("strategy_commit"):
        raise ValueError("strategy commit audit is missing")


def run_constrained(
    *,
    start: str | None,
    end: str | None,
    top_n: int,
    cost_bps: float,
    slippage_bps: float,
    output: Path = OUT_FILE,
) -> dict:
    files = base.history_files()
    selected_files, _ = base.iter_selected_dates(files, start, end)
    if len(selected_files) < 2:
        raise ValueError("Backtest needs at least two trading days")

    strategy_model, strategy_version, strategy_commit = base.load_strategy()
    states = base.RollingFeatureState()
    portfolio = StatefulPortfolio(initial_cash=1.0)

    daily_rows = []
    selection_rows = []
    intraday_rows = []
    selected_total = 0
    missing_execution_total = 0
    limit_up_total = 0
    limit_down_total = 0
    blocked_buy_total = 0
    blocked_sell_total = 0

    current = base.read_daily(selected_files[0])
    cost_rate = (cost_bps + slippage_bps) / 10000.0

    for i in range(len(selected_files) - 1):
        next_path = selected_files[i + 1]
        next_date = next_path.name[:10]
        current_date = current["date"].iloc[0]
        next_day = base.read_daily(next_path)

        frame = base.build_strategy_frame(current, states)
        targets = base.select_targets(frame, strategy_model, top_n)
        selected_symbols = targets["symbol"].astype(str).tolist()
        selected_total += len(selected_symbols)

        limit_up, limit_down = base.execution_limit_diagnostics(targets, next_day)
        limit_up_total += limit_up
        limit_down_total += limit_down

        execution = next_day.set_index("symbol", drop=False)
        missing = 0
        for symbol in selected_symbols:
            if symbol not in execution.index:
                missing += 1
                continue
            row = execution.loc[symbol]
            if not valid_price(row.get("open", np.nan)) or not valid_price(row.get("close", np.nan)):
                missing += 1
        missing_execution_total += missing

        result = portfolio.rebalance(
            targets,
            next_day,
            cost_rate,
        )
        blocked_buy_total += result["blocked_buy_count"]
        blocked_sell_total += result["blocked_sell_count"]

        intraday_rows.append({"date": next_date, "net_return": float(result["equity_close"] / result["equity_open"] - 1.0), "turnover": result["turnover"]})

        daily_rows.append(
            {
                "date": next_date,
                "signal_date": current_date,
                **result,
                "target_count": len(selected_symbols),
                "missing_execution_count": missing,
                "limit_up_open_count": limit_up,
                "limit_down_open_count": limit_down,
            }
        )
        selection_rows.append(
            {
                "signal_date": current_date,
                "execution_date": next_date,
                "candidate_count": len(selected_symbols),
                "symbols": selected_symbols,
                "blocked_buy_count": result["blocked_buy_count"],
                "blocked_sell_count": result["blocked_sell_count"],
            }
        )

        if (i + 1) % 100 == 0:
            print(
                f"[constrained] {i + 1}/{len(selected_files) - 1} "
                f"signal={current_date} execution={next_date} "
                f"targets={len(selected_symbols)} positions={result['position_count']}"
            )

        current = next_day

    daily = pd.DataFrame(daily_rows)
    warmup_sessions = min(59, len(selected_files) - 1)
    performance = daily.reset_index(drop=True)

    overall = base.metrics(performance)
    annual = base.period_metrics(performance, "Y")
    monthly = base.period_metrics(performance, "M")
    rolling_252d = base.rolling_252d_metrics(performance)
    intraday_metrics = base.metrics(pd.DataFrame(intraday_rows))

    payload = {
        "schema_version": 1,
        "status": "ready",
        "method": "strict_point_in_time_stateful_execution",
        "future_function": False,
        "market_scope": (
            "沪深主板：000001-004999.SZ（排除001001-001199 CDR）"
            "+ 600/601/603/605.SH"
        ),
        "selection_rule": (
            "与基准回测完全相同：T日收盘后使用历史截面和Aquant-Private/main打分，取Top-N。"
        ),
        "execution_rule": (
            "T+1开盘进行持仓再平衡；涨停买入、跌停卖出、停牌和无有效价格均视为不可成交；"
            "无法成交的旧持仓继续持有；未成交买单保留现金。"
        ),
        "return_definition": "stateful close-to-close return, with T+1 open execution and overnight gaps",
        "lookback_trading_days": 60,
        "top_n": top_n,
        "transaction_cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "execution_cost_total_bps": cost_bps + slippage_bps,
        "warmup_sessions": warmup_sessions,
        "backtest_start": start or selected_files[0].name[:10],
        "backtest_end": end or selected_files[-1].name[:10],
        "trade_start": daily["date"].min() if not daily.empty else None,
        "trade_end": daily["date"].max() if not daily.empty else None,
        "strategy_source": "Aquant-Private/main",
        "strategy_version": strategy_version,
        "strategy_commit": strategy_commit,
        "overall": overall,
        "intraday_metrics": intraday_metrics,
        "annual": annual,
        "monthly": monthly,
        "rolling_252d": rolling_252d,
        "audit": {
            "historical_files_used": len(selected_files),
            "selected_sessions": len(selected_files) - 1,
            "performance_sessions": len(performance),
            "selected_stock_observations": selected_total,
            "missing_execution_observations": missing_execution_total,
            "limit_up_open_observations": limit_up_total,
            "limit_down_open_observations": limit_down_total,
            "blocked_buy_orders": blocked_buy_total,
            "blocked_sell_orders": blocked_sell_total,
            "stateful_positions": True,
            "overnight_gaps_included": True,
            "current_universe_not_used_for_history": True,
            "current_names_not_used_for_history": True,
            "future_adjusted_factor_not_used": True,
            "pit_fundamentals_required": False,
            "intraday_return_definition": "T+1 close equity / T+1 pre-trade open equity - 1",
        },
        "daily": daily.to_dict(orient="records"),
        "selection_audit": selection_rows,
    }

    validate_constrained_payload(payload)
    print("[constrained] quality audit passed")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stateful constrained Aquant historical backtest")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--output", default=str(OUT_FILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive")
    if args.cost_bps < 0 or args.slippage_bps < 0:
        raise SystemExit("--cost-bps and --slippage-bps must be >= 0")

    payload = run_constrained(
        start=args.start,
        end=args.end,
        top_n=args.top_n,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        output=Path(args.output),
    )
    print(json.dumps(payload["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
