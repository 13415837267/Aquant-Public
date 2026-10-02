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
from scripts.portfolio import allocate_weights

ROOT = base.ROOT
OUT_FILE = ROOT / "data" / "backtest" / "execution_constrained.json"
DEFAULT_TOP_N = base.DEFAULT_TOP_N
DEFAULT_COST_BPS = base.DEFAULT_COST_BPS
DEFAULT_SLIPPAGE_BPS = base.DEFAULT_SLIPPAGE_BPS
DEFAULT_TURNOVER_CAP = 0.30
DEFAULT_CASH_BUFFER = 0.05
LOT_SIZE = 100


def round_lot(shares: float) -> int:
    if shares <= 0:
        return 0
    return int(np.floor((shares + 1e-9) / LOT_SIZE) * LOT_SIZE)


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


def blocked_for_side(row: pd.Series, side: str, enforce_limits: bool = True) -> bool:
    paused = row.get("is_paused", 0)
    if pd.notna(paused) and float(paused) != 0:
        return True

    open_price = row.get("open", np.nan)
    if not valid_price(open_price):
        return True

    if not enforce_limits:
        return False

    if side == "buy":
        high_limit = row.get("high_limit", np.nan)
        return pd.notna(high_limit) and valid_price(high_limit) and float(open_price) >= float(high_limit) - 1e-8

    if side == "sell":
        low_limit = row.get("low_limit", np.nan)
        return pd.notna(low_limit) and valid_price(low_limit) and float(open_price) <= float(low_limit) + 1e-8

    raise ValueError(f"unsupported execution side: {side}")


class FlattenedIntradayPortfolio:
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
        *,
        enforce_limits: bool = True,
        turnover_cap: float = DEFAULT_TURNOVER_CAP,
    ) -> dict:
        execution = execution_df.set_index("symbol", drop=False)
        target_symbols = [str(x).zfill(6) for x in targets.get("symbol", pd.Series(dtype=str)).tolist()]
        if "target_weight" in targets.columns:
            target_weights = {
                str(row.symbol).zfill(6): float(row.target_weight)
                for row in targets.itertuples(index=False)
                if np.isfinite(float(row.target_weight)) and float(row.target_weight) > 0
            }
        else:
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
            if row is None or blocked_for_side(row, "sell", enforce_limits):
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
            if row is None or blocked_for_side(row, "buy", enforce_limits):
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

        # Mark the portfolio at T+1 close, then flatten all executable positions.
        # Only a failed close sale creates a forced overnight carry.
        mark_close_equity = self.cash
        close_missing = 0
        close_sell_notional = 0.0
        blocked_close_sell = 0
        close_sell_count = 0
        close_sell_symbols = []

        for symbol, shares in list(self.shares.items()):
            row = execution.loc[symbol] if symbol in execution.index else None
            close_price = row_value(
                execution,
                symbol,
                "close",
                self.last_close.get(symbol, np.nan),
            )
            if not valid_price(close_price):
                close_missing += 1
                continue

            close_price = float(close_price)
            self.last_close[symbol] = close_price
            mark_close_equity += shares * close_price

            if row is None or blocked_for_side(row, "sell", enforce_limits):
                blocked_close_sell += 1
                continue

            notional = float(shares) * close_price
            self.cash += notional * (1.0 - cost_rate)
            close_sell_notional += notional
            close_sell_count += 1
            close_sell_symbols.append(symbol)
            del self.shares[symbol]

        end_equity = self.cash
        for symbol, shares in self.shares.items():
            price = self.last_close.get(symbol, np.nan)
            if valid_price(price):
                end_equity += shares * float(price)

        if not valid_price(end_equity) or end_equity <= 0:
            raise ValueError("flattened execution produced invalid close equity")

        total_traded = sell_notional + buy_notional + close_sell_notional
        total_cost = total_traded * cost_rate
        net_return = end_equity / prior_equity - 1.0
        gross_return = net_return + total_cost

        turnover_value = total_traded / equity_open if equity_open > 0 else 0.0
        overnight_return = equity_open / prior_equity - 1.0
        intraday_return = mark_close_equity / equity_open - 1.0 if equity_open > 0 else 0.0

        self.prev_close_equity = float(end_equity)

        return {
            "net_return": float(net_return),
            "gross_return": float(gross_return),
            "intraday_return": float(intraday_return),
            "turnover": float(turnover_value),
            "cost": float(total_cost),
            "equity_open": float(equity_open),
            "equity_close": float(end_equity),
            "mark_close_equity": float(mark_close_equity),
            "overnight_return": float(overnight_return),
            "sell_count": len(sell_symbols),
            "buy_count": len(buy_symbols),
            "close_sell_count": close_sell_count,
            "blocked_buy_count": blocked_buy,
            "blocked_sell_count": blocked_sell,
            "blocked_close_sell_count": blocked_close_sell,
            "close_missing_count": close_missing,
            "cash": float(self.cash),
            "position_count": len(self.shares),
            "forced_overnight_positions": len(self.shares),
            "sold_symbols": sell_symbols,
            "bought_symbols": buy_symbols,
            "close_sold_symbols": close_sell_symbols,
        }
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
    trade_start = payload.get("trade_start")
    performance_daily = [row for row in daily if trade_start and row.get("date", "") >= trade_start] if trade_start else daily
    if int(audit.get("performance_sessions", -1)) != len(performance_daily):
        raise ValueError("performance_sessions does not match performance daily rows")
    if int(overall.get("trading_days", -1)) != len(performance_daily):
        raise ValueError("overall trading_days does not match performance sessions")

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
            "cash_floor",
        ):
            value = float(row[key])
            if not np.isfinite(value):
                raise ValueError(f"non-finite constrained field: {key}")
        if float(row["equity_close"]) <= 0 or float(row["equity_open"]) <= 0:
            raise ValueError("non-positive portfolio equity")
        if float(row["turnover"]) < -1e-12:
            raise ValueError("negative turnover")
        if float(row["turnover"]) > float(payload["turnover_cap"]) + 1e-8:
            raise ValueError("daily turnover cap violated")
        if float(row["cash"]) + 1e-9 < float(row["cash_floor"]):
            raise ValueError("cash floor violated")
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



class StatefulPortfolio:
    """Persistent long-only portfolio: rebalance at T+1 open and hold overnight."""

    def __init__(self, initial_cash: float = 1.0, cash_buffer: float = DEFAULT_CASH_BUFFER) -> None:
        if not 0 <= cash_buffer < 1:
            raise ValueError("cash_buffer must be in [0, 1)")
        self.cash = float(initial_cash)
        self.prev_close_equity = float(initial_cash)
        self.cash_buffer = float(cash_buffer)
        self.shares: dict[str, float] = {}
        self.last_close: dict[str, float] = {}

    def open_equity(self, execution: pd.DataFrame) -> float:
        value = self.cash
        for symbol, shares in self.shares.items():
            price = row_value(execution, symbol, "open", self.last_close.get(symbol, np.nan))
            if valid_price(price):
                value += shares * float(price)
            elif valid_price(self.last_close.get(symbol)):
                value += shares * float(self.last_close[symbol])
        return float(value)

    def rebalance(
        self,
        targets: pd.DataFrame,
        execution_df: pd.DataFrame,
        cost_rate: float,
        *,
        enforce_limits: bool = True,
        turnover_cap: float = DEFAULT_TURNOVER_CAP,
    ) -> dict:
        execution = execution_df.set_index("symbol", drop=False)
        target_symbols = [str(x).zfill(6) for x in targets.get("symbol", pd.Series(dtype=str)).tolist()]
        target_weights = (
            {str(r.symbol).zfill(6): float(r.target_weight) for r in targets.itertuples(index=False)
             if "target_weight" in targets.columns and np.isfinite(float(r.target_weight)) and float(r.target_weight) > 0}
            if "target_weight" in targets.columns else base.normalize_weights(target_symbols)
        )
        if not 0 < turnover_cap <= 1:
            raise ValueError("turnover_cap must be in (0, 1]")
        prior_equity = float(self.prev_close_equity)
        equity_open = self.open_equity(execution)
        if not valid_price(equity_open) or equity_open <= 0:
            equity_open = prior_equity

        def open_price(symbol: str) -> float:
            return row_value(execution, symbol, "open", self.last_close.get(symbol, np.nan))

        open_values = {s: sh * open_price(s) for s, sh in self.shares.items() if valid_price(open_price(s))}
        current_weights = {s: v / equity_open for s, v in open_values.items()} if equity_open > 0 else {}
        trade_keys = set(current_weights) | set(target_weights)
        requested_l1 = sum(abs(target_weights.get(s, 0.0) - current_weights.get(s, 0.0)) for s in trade_keys)
        trade_scale = min(1.0, turnover_cap / requested_l1) if requested_l1 > 0 else 1.0
        effective_targets = {
            s: current_weights.get(s, 0.0) + trade_scale * (target_weights.get(s, 0.0) - current_weights.get(s, 0.0))
            for s in trade_keys
        }
        sell_notional = 0.0
        blocked_sell = 0
        sold_symbols = []

        for symbol in list(self.shares):
            px = open_price(symbol)
            if not valid_price(px):
                blocked_sell += 1
                continue
            current_value = open_values.get(symbol, 0.0)
            desired_value = equity_open * effective_targets.get(symbol, 0.0)
            delta = current_value - desired_value
            if delta <= 1e-12:
                continue
            row = execution.loc[symbol] if symbol in execution.index else None
            if row is None or blocked_for_side(row, "sell", enforce_limits):
                blocked_sell += 1
                continue
            shares_to_sell = min(self.shares[symbol], round_lot(delta / px))
            notional = shares_to_sell * px
            if shares_to_sell <= 0:
                continue
            self.shares[symbol] -= shares_to_sell
            if self.shares[symbol] <= 1e-12:
                del self.shares[symbol]
            self.cash += notional * (1.0 - cost_rate)
            sell_notional += notional
            sold_symbols.append(symbol)

        current_open_values = {
            s: sh * open_price(s) for s, sh in self.shares.items() if valid_price(open_price(s))
        }
        buy_orders = {}
        blocked_buy = 0
        for symbol, weight in effective_targets.items():
            px = open_price(symbol)
            current_value = current_open_values.get(symbol, 0.0)
            desired_value = equity_open * weight
            deficit = desired_value - current_value
            if deficit <= 1e-12:
                continue
            row = execution.loc[symbol] if symbol in execution.index else None
            if row is None or blocked_for_side(row, "buy", enforce_limits):
                blocked_buy += 1
                continue
            buy_orders[symbol] = deficit

        total_buy = float(sum(buy_orders.values()))
        cash_floor = equity_open * self.cash_buffer
        affordable_cash = max(0.0, self.cash - cash_floor)
        affordable = affordable_cash / (1.0 + cost_rate)
        ratio = min(1.0, affordable / total_buy) if total_buy > 0 else 0.0
        buy_notional = 0.0
        bought_symbols = []
        for symbol, deficit in buy_orders.items():
            px = open_price(symbol)
            notional = deficit * ratio
            shares = round_lot(notional / px)
            if shares <= 0:
                continue
            notional = shares * px
            total_cash = notional * (1.0 + cost_rate)
            available_for_buy = max(0.0, self.cash - cash_floor)
            if total_cash > available_for_buy + 1e-12:
                shares = round_lot(available_for_buy / ((1.0 + cost_rate) * px))
                notional = shares * px
                total_cash = notional * (1.0 + cost_rate)
            if shares <= 0 or total_cash > max(0.0, self.cash - cash_floor) + 1e-12:
                continue
            self.shares[symbol] = self.shares.get(symbol, 0.0) + shares
            self.cash -= total_cash
            buy_notional += notional
            bought_symbols.append(symbol)

        # Clean small non-target residuals only with unused turnover capacity.
        residual_cleanup_count = 0
        turnover_room = max(0.0, equity_open * turnover_cap - sell_notional - buy_notional)
        for symbol in list(self.shares):
            if symbol in target_weights:
                continue
            px = open_price(symbol)
            if not valid_price(px) or self.shares[symbol] > 3 * LOT_SIZE:
                continue
            row = execution.loc[symbol] if symbol in execution.index else None
            if row is None or blocked_for_side(row, "sell", enforce_limits):
                continue
            shares_to_sell = min(round_lot(self.shares[symbol]), round_lot(turnover_room / px))
            if shares_to_sell <= 0:
                continue
            notional = shares_to_sell * px
            self.shares[symbol] -= shares_to_sell
            self.cash += notional * (1.0 - cost_rate)
            sell_notional += notional
            turnover_room -= notional
            residual_cleanup_count += 1
            sold_symbols.append(symbol)
            if self.shares[symbol] <= 1e-12:
                del self.shares[symbol]

        mark_close_equity = self.cash
        close_missing = 0
        for symbol, shares in self.shares.items():
            close_price = row_value(execution, symbol, "close", self.last_close.get(symbol, np.nan))
            if not valid_price(close_price):
                close_missing += 1
                continue
            self.last_close[symbol] = float(close_price)
            mark_close_equity += shares * float(close_price)

        end_equity = mark_close_equity
        traded = sell_notional + buy_notional
        total_cost = traded * cost_rate
        net_return = end_equity / prior_equity - 1.0
        gross_return = net_return + total_cost
        self.prev_close_equity = float(end_equity)
        return {
            "net_return": float(net_return),
            "gross_return": float(gross_return),
            "intraday_return": float(end_equity / equity_open - 1.0) if equity_open > 0 else 0.0,
            "turnover": float(traded / equity_open) if equity_open > 0 else 0.0,
            "turnover_cap": float(turnover_cap),
            "turnover_scale": float(trade_scale),
            "cash_buffer": float(self.cash_buffer),
            "cash_floor": float(cash_floor),
            "cash_floor_enforced": True,
            "cost": float(total_cost),
            "equity_open": float(equity_open),
            "equity_close": float(end_equity),
            "mark_close_equity": float(mark_close_equity),
            "overnight_return": float(equity_open / prior_equity - 1.0) if prior_equity > 0 else 0.0,
            "sell_count": len(sold_symbols),
            "buy_count": len(bought_symbols),
            "close_sell_count": 0,
            "blocked_buy_count": blocked_buy,
            "blocked_sell_count": blocked_sell,
            "blocked_close_sell_count": 0,
            "residual_cleanup_count": residual_cleanup_count,
            "close_missing_count": close_missing,
            "cash": float(self.cash),
            "position_count": len(self.shares),
            "forced_overnight_positions": len(self.shares),
            "sold_symbols": sold_symbols,
            "bought_symbols": bought_symbols,
            "close_sold_symbols": [],
        }

def run_constrained(
    *,
    start: str | None,
    end: str | None,
    top_n: int,
    cost_bps: float,
    slippage_bps: float,
    turnover_cap: float,
    output: Path = OUT_FILE,
) -> dict:
    files = base.history_files()
    selected_files, _ = base.iter_selected_dates(files, start, end)
    if len(selected_files) < 2:
        raise ValueError("Backtest needs at least two trading days")

    strategy_model, strategy_version, strategy_commit = base.load_strategy()
    states = base.RollingFeatureState()
    # Use realistic capital so A-share 100-share lots are representable.
    portfolio = StatefulPortfolio(initial_cash=1_000_000.0, cash_buffer=DEFAULT_CASH_BUFFER)
    unconstrained = StatefulPortfolio(initial_cash=1_000_000.0, cash_buffer=DEFAULT_CASH_BUFFER)

    daily_rows = []
    unconstrained_rows = []
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
        allocation = allocate_weights(targets, max_weight=0.05, cash_buffer=0.05)
        targets["target_weight"] = targets["symbol"].map(allocation).fillna(0.0)
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
            enforce_limits=True,
            turnover_cap=turnover_cap,
        )
        unconstrained_result = unconstrained.rebalance(
            targets,
            next_day,
            cost_rate,
            enforce_limits=False,
            turnover_cap=turnover_cap,
        )
        blocked_buy_total += result["blocked_buy_count"]
        blocked_sell_total += result["blocked_sell_count"]

        intraday_rows.append({"date": next_date, "net_return": result["intraday_return"], "turnover": result["turnover"]})
        unconstrained_rows.append({"date": next_date, "net_return": float(unconstrained_result["net_return"]), "turnover": unconstrained_result["turnover"]})

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
    active = daily.loc[daily["target_count"] > 0].reset_index(drop=True)
    performance = active
    trade_start = performance["date"].min() if not performance.empty else None

    overall = base.metrics(performance)
    annual = base.period_metrics(performance, "Y")
    monthly = base.period_metrics(performance, "M")
    rolling_252d = base.rolling_252d_metrics(performance)
    intraday_frame = pd.DataFrame(intraday_rows)
    if trade_start:
        intraday_frame = intraday_frame.loc[intraday_frame["date"] >= trade_start].reset_index(drop=True)
    intraday_metrics = base.metrics(intraday_frame)
    unconstrained_daily = pd.DataFrame(unconstrained_rows)
    if trade_start:
        unconstrained_daily = unconstrained_daily.loc[unconstrained_daily["date"] >= trade_start].reset_index(drop=True)
    unconstrained_overall = base.metrics(unconstrained_daily)
    unconstrained_annual = base.period_metrics(unconstrained_daily, "Y")
    unconstrained_monthly = base.period_metrics(unconstrained_daily, "M")
    unconstrained_rolling_252d = base.rolling_252d_metrics(unconstrained_daily)

    payload = {
        "schema_version": 1,
        "status": "ready",
        "method": "strict_point_in_time_stateful_execution_production_weights",
        "future_function": False,
        "market_scope": (
            "沪深主板：000001-004999.SZ（排除001001-001199 CDR）"
            "+ 600/601/603/605.SH"
        ),
        "selection_rule": (
            "T日收盘后使用历史截面和Aquant-Private/main打分，取Top-N；组合权重与生产portfolio.py完全一致。"
        ),
        "portfolio_rule": "shared scripts.portfolio.allocate_weights: inverse volatility, 5% single-name cap, 5% cash buffer",
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
        "turnover_cap": turnover_cap,
        "lot_size": LOT_SIZE,
        "cash_buffer": DEFAULT_CASH_BUFFER,
        "warmup_sessions": warmup_sessions,
        "backtest_start": start or selected_files[0].name[:10],
        "backtest_end": end or selected_files[-1].name[:10],
        "trade_start": trade_start,
        "trade_end": daily["date"].max() if not daily.empty else None,
        "strategy_source": "Aquant-Private/main",
        "strategy_version": strategy_version,
        "strategy_commit": strategy_commit,
        "overall": overall,
        "intraday_metrics": intraday_metrics,
        "unconstrained_persistent": {
            "overall": unconstrained_overall,
            "annual": unconstrained_annual,
            "monthly": unconstrained_monthly,
            "rolling_252d": unconstrained_rolling_252d,
        },
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
            "blocked_close_sell_orders": int(sum(r["blocked_close_sell_count"] for r in daily_rows)),
            "forced_overnight_sessions": int(sum(r["forced_overnight_positions"] > 0 for r in daily_rows)),
            "forced_overnight_position_observations": int(sum(r["forced_overnight_positions"] for r in daily_rows)),
            "persistent_positions": True,
            "overnight_gaps_included": True,
            "current_universe_not_used_for_history": True,
            "current_names_not_used_for_history": True,
            "future_adjusted_factor_not_used": True,
            "pit_fundamentals_required": False,
            "intraday_return_definition": "T+1 close equity / T+1 pre-trade open equity - 1",
            "constraint_comparison": "same persistent portfolio and PIT signals, price limits enabled vs disabled; paused/missing-price restrictions remain",
            "turnover_cap_enforced": True,
            "lot_size_enforced": True,
            "cash_floor_enforced": True,
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
    parser.add_argument("--turnover-cap", type=float, default=DEFAULT_TURNOVER_CAP)
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
        turnover_cap=args.turnover_cap,
        output=Path(args.output),
    )
    print(json.dumps(payload["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()