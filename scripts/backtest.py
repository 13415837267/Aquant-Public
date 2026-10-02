"""Point-in-time historical backtest for the public production pipeline.

The backtest deliberately keeps strategy logic in Aquant-Private. Public only
provides the historical market replay, execution assumptions, accounting, and
audit output.

Selection:
  - End-of-day T cross-section from the historical file for T.
  - Shanghai/Shenzhen main-board only.
  - Same investability filters as production.
  - 60 trading-day momentum, 20-day volatility, 20-day volume ratio.
  - Score using the strategy loaded from Aquant-Private/main.

Execution:
  - Rebalance at T+1 open into top N equal-weight names.
  - Hold through T+1 close.
  - Transaction cost + slippage are charged on target-weight turnover.
  - A missing/invalid T+1 execution price makes that target weight inactive for
    the day and is recorded in audit diagnostics.

This module is price/valuation based. It does not fabricate point-in-time
quarterly fundamentals that are not present in the current v0.2 strategy.
"""
from __future__ import annotations

import argparse
import gzip
import importlib
import json
import os
import subprocess
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.market_scope import is_main_board_symbol

DATA_DIR = ROOT / "data"
HISTORY_DIR = DATA_DIR / "history"
OUT_FILE = DATA_DIR / "backtest" / "latest.json"

REQUIRED_COLUMNS = {
    "symbol",
    "date",
    "open",
    "close",
    "volume",
    "amount",
    "pct_chg",
    "turnover_pct",
    "is_paused",
    "is_st",
    "pe_ratio",
    "pb_ratio",
}

DEFAULT_TOP_N = 30
DEFAULT_COST_BPS = 3.0
DEFAULT_SLIPPAGE_BPS = 2.0


def history_files() -> list[Path]:
    files = sorted(
        HISTORY_DIR.glob("????/*.csv.gz"),
        key=lambda p: p.name[:10],
    )
    if not files:
        raise RuntimeError("No historical daily files under data/history/YYYY/")
    return files


def read_daily(path: Path) -> pd.DataFrame:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            df = pd.read_csv(fh)
    except Exception as exc:
        raise RuntimeError(f"failed to read {path}: {exc}") from exc

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise RuntimeError(f"{path.name} missing required columns: {sorted(missing)}")

    df = df.loc[df["symbol"].map(is_main_board_symbol)].copy()
    if df.empty:
        raise RuntimeError(f"{path.name} has no main-board rows")

    df["symbol"] = (
        df["symbol"].astype(str).str.extract(r"(\d{6})")[0].str.zfill(6)
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    file_date = path.name[:10]
    if df["date"].ne(file_date).any():
        raise RuntimeError(f"{path.name} contains rows with date != filename date")

    numeric = [
        "close",
        "volume",
        "amount",
        "pct_chg",
        "turnover_pct",
        "pe_ratio",
        "pb_ratio",
        "is_paused",
        "is_st",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # There should be at most one row per security per day.
    dup = df.duplicated(["symbol", "date"])
    if dup.any():
        raise RuntimeError(f"{path.name} has {int(dup.sum())} duplicate symbol/date rows")

    return df


def load_strategy() -> tuple[object, str, str]:
    strategy_path = os.environ.get("AQUANT_PRIVATE_STRATEGY_PATH", "").strip()
    if not strategy_path:
        raise RuntimeError(
            "AQUANT_PRIVATE_STRATEGY_PATH is required; Public must not ship an independent strategy"
        )

    root = Path(strategy_path).resolve()
    model_path = root / "strategy" / "model.py"
    if not model_path.exists():
        raise RuntimeError(f"Private strategy missing: {model_path}")

    sys.path.insert(0, str(root))
    model = importlib.import_module("strategy.model")
    version_mod = importlib.import_module("strategy.version")
    version = str(getattr(version_mod, "STRATEGY_VERSION", "unknown"))

    commit = os.environ.get("AQUANT_PRIVATE_STRATEGY_COMMIT", "").strip()
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
            ).strip()
        except Exception:
            commit = "unknown"
    return model, version, commit


def iter_selected_dates(
    files: list[Path],
    start: str | None,
    end: str | None,
) -> tuple[list[Path], dict[str, int]]:
    dates = [p.name[:10] for p in files]
    index = {d: i for i, d in enumerate(dates)}

    start_i = 0 if start is None else index.get(start)
    end_i = len(dates) - 1 if end is None else index.get(end)
    if start_i is None:
        raise ValueError(f"start date {start} is not a trading date in the database")
    if end_i is None:
        raise ValueError(f"end date {end} is not a trading date in the database")
    if start_i > end_i:
        raise ValueError("start date must be <= end date")

    return files[start_i : end_i + 1], index


def build_strategy_frame(
    df: pd.DataFrame,
    states: dict[str, dict[str, deque[float]]],
) -> pd.DataFrame:
    """Append current observations and return today's scorable main-board frame."""
    rows = []

    for row in df.itertuples(index=False):
        symbol = str(row.symbol).zfill(6)
        state = states.setdefault(
            symbol,
            {
                "rets": deque(maxlen=60),
                "vol_rets": deque(maxlen=20),
                "volumes": deque(maxlen=20),
            },
        )

        daily_ret = float(row.pct_chg) / 100.0 if pd.notna(row.pct_chg) else np.nan
        volume = float(row.volume) if pd.notna(row.volume) else np.nan

        state["rets"].append(daily_ret)
        state["vol_rets"].append(daily_ret)
        state["volumes"].append(volume)

        ret60 = np.nan
        if len(state["rets"]) == 60 and all(np.isfinite(state["rets"])):
            gross = np.prod([1.0 + x for x in state["rets"]])
            ret60 = float(gross - 1.0)

        volatility_proxy = np.nan
        valid_rets = [x for x in state["vol_rets"] if np.isfinite(x)]
        if len(valid_rets) >= 10:
            volatility_proxy = float(np.std(valid_rets, ddof=1) * 100.0)

        volume_ratio = np.nan
        valid_volumes = [x for x in state["volumes"] if np.isfinite(x)]
        if len(valid_volumes) >= 10 and valid_volumes[-1] > 0:
            mean_volume = float(np.mean(valid_volumes))
            if mean_volume > 0:
                volume_ratio = float(valid_volumes[-1] / mean_volume)

        name_st = str(getattr(row, "name", "") or "")
        if "ST" in name_st.upper() or "退" in name_st:
            continue

        if not (
            pd.notna(row.is_paused)
            and float(row.is_paused) == 0
            and pd.notna(row.is_st)
            and float(row.is_st) == 0
            and pd.notna(row.close)
            and float(row.close) > 2.0
            and pd.notna(row.amount)
            and float(row.amount) >= 2e7
            and np.isfinite(ret60)
        ):
            continue

        rows.append(
            {
                "symbol": symbol,
                "name": name_st or f"股票{symbol}",
                "close": float(row.close),
                "amount": float(row.amount),
                "turnover_pct": (
                    float(row.turnover_pct) if pd.notna(row.turnover_pct) else np.nan
                ),
                "change_pct": (
                    float(row.pct_chg) if pd.notna(row.pct_chg) else np.nan
                ),
                "pe": float(row.pe_ratio) if pd.notna(row.pe_ratio) else np.nan,
                "pb": float(row.pb_ratio) if pd.notna(row.pb_ratio) else np.nan,
                "ret_60d": ret60,
                "momentum_60d": ret60 * 100.0,
                "volatility_proxy": volatility_proxy,
                "volume_ratio": volume_ratio,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "close",
                "amount",
                "turnover_pct",
                "change_pct",
                "pe",
                "pb",
                "ret_60d",
                "momentum_60d",
                "volatility_proxy",
                "volume_ratio",
            ]
        )

    return pd.DataFrame(rows)


def select_targets(frame: pd.DataFrame, strategy_model: object, top_n: int) -> pd.DataFrame:
    if frame.empty:
        return frame

    scored = strategy_model.score_universe(frame)
    scored = scored.sort_values(
        ["score", "amount", "symbol"],
        ascending=[False, False, True],
    )
    return scored.head(top_n).reset_index(drop=True)


def next_session_return(
    selected: pd.DataFrame,
    execution_df: pd.DataFrame,
) -> tuple[float, list[str], list[str]]:
    if selected.empty:
        return 0.0, [], []

    execution = execution_df.set_index("symbol", drop=False)
    returns = []
    executed = []
    missing = []

    for row in selected.itertuples(index=False):
        symbol = str(row.symbol).zfill(6)
        if symbol not in execution.index:
            missing.append(symbol)
            continue

        future = execution.loc[symbol]
        open_price = getattr(future, "open", np.nan)
        close_price = getattr(future, "close", np.nan)
        if not (pd.notna(open_price) and pd.notna(close_price)):
            missing.append(symbol)
            continue
        open_price = float(open_price)
        close_price = float(close_price)
        if open_price <= 0 or close_price <= 0:
            missing.append(symbol)
            continue

        returns.append(close_price / open_price - 1.0)
        executed.append(symbol)

    gross = float(np.mean(returns)) if returns else 0.0
    return gross, executed, missing


def normalize_weights(symbols: Iterable[str]) -> dict[str, float]:
    values = sorted(set(str(x).zfill(6) for x in symbols))
    if not values:
        return {}
    weight = 1.0 / len(values)
    return {symbol: weight for symbol in values}


def turnover(prev: dict[str, float], target: dict[str, float]) -> float:
    keys = set(prev) | set(target)
    return float(sum(abs(target.get(k, 0.0) - prev.get(k, 0.0)) for k in keys))


def metrics(daily: pd.DataFrame) -> dict:
    if daily.empty:
        return {
            "trading_days": 0,
            "total_return_pct": None,
            "annualized_return_pct": None,
            "annualized_volatility_pct": None,
            "sharpe": None,
            "max_drawdown_pct": None,
            "average_turnover_pct": None,
            "total_turnover_pct": None,
        }

    returns = pd.to_numeric(daily["net_return"], errors="coerce").fillna(0.0)
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = max(len(daily) / 252.0, 1.0 / 252.0)
    annualized = float((1.0 + total_return) ** (1.0 / years) - 1.0)

    std = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    sharpe = float(returns.mean() / std * np.sqrt(252.0)) if std > 0 else None

    drawdown = equity / equity.cummax() - 1.0
    return {
        "trading_days": int(len(daily)),
        "total_return_pct": total_return * 100.0,
        "annualized_return_pct": annualized * 100.0,
        "annualized_volatility_pct": std * np.sqrt(252.0) * 100.0 if np.isfinite(std) else None,
        "sharpe": sharpe,
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "average_turnover_pct": float(daily["turnover"].mean() * 100.0),
        "total_turnover_pct": float(daily["turnover"].sum() * 100.0),
    }


def period_metrics(daily: pd.DataFrame, freq: str) -> list[dict]:
    if daily.empty:
        return []
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    rows = []
    for period, group in frame.groupby(frame["date"].dt.to_period(freq)):
        item = metrics(group.reset_index(drop=True))
        item["period"] = str(period)
        item["start_date"] = group["date"].min().strftime("%Y-%m-%d")
        item["end_date"] = group["date"].max().strftime("%Y-%m-%d")
        rows.append(item)
    return rows


def run_backtest(
    *,
    start: str | None,
    end: str | None,
    top_n: int,
    cost_bps: float,
    slippage_bps: float,
) -> dict:
    files = history_files()
    selected_files, all_index = iter_selected_dates(files, start, end)
    if len(selected_files) < 2:
        raise ValueError("Backtest needs at least two trading days")

    strategy_model, strategy_version, strategy_commit = load_strategy()
    states: dict[str, dict[str, deque[float]]] = {}

    # The first 59 sessions are warm-up only; they cannot be traded until each
    # stock has a complete 60-observation return window.
    daily_rows = []
    selection_rows = []
    prev_target: dict[str, float] = {}
    missing_execution_total = 0
    selected_total = 0

    # We need T and T+1 for every scoring date.
    for i in range(len(selected_files) - 1):
        current_path = selected_files[i]
        next_path = selected_files[i + 1]
        current_date = current_path.name[:10]
        next_date = next_path.name[:10]

        current = read_daily(current_path)
        next_day = read_daily(next_path)

        frame = build_strategy_frame(current, states)
        targets = select_targets(frame, strategy_model, top_n)
        target_symbols = targets["symbol"].astype(str).tolist()
        target_weights = normalize_weights(target_symbols)

        selected_total += len(target_symbols)
        gross_return, executed, missing = next_session_return(targets, next_day)
        missing_execution_total += len(missing)

        # Use equal-weight intended targets for turnover. Missing T+1 execution
        # is a visible execution-quality event, not silently removed from the
        # portfolio definition.
        turn = turnover(prev_target, target_weights)
        total_cost = turn * (cost_bps + slippage_bps) / 10000.0
        net_return = (1.0 + gross_return) - 1.0 - total_cost

        daily_rows.append(
            {
                "date": next_date,
                "signal_date": current_date,
                "gross_return": gross_return,
                "turnover": turn,
                "cost": total_cost,
                "net_return": net_return,
                "target_count": len(target_symbols),
                "executed_count": len(executed),
                "missing_execution_count": len(missing),
                "missing_execution_symbols": ",".join(missing[:20]),
            }
        )
        selection_rows.append(
            {
                "signal_date": current_date,
                "execution_date": next_date,
                "candidate_count": len(target_symbols),
                "executed_count": len(executed),
                "missing_execution_count": len(missing),
                "symbols": target_symbols,
                "scores": [
                    round(float(x), 6)
                    for x in targets["score"].tolist()
                ],
            }
        )
        prev_target = target_weights

        if (i + 1) % 100 == 0:
            print(
                f"[backtest] {i + 1}/{len(selected_files) - 1} "
                f"signal={current_date} execution={next_date} "
                f"targets={len(target_symbols)}"
            )

    daily = pd.DataFrame(daily_rows)
    overall = metrics(daily)
    annual = period_metrics(daily, "Y")
    monthly = period_metrics(daily, "M")

    warmup_days = min(59, len(selected_files) - 1)
    trade_start = daily["date"].min() if not daily.empty else None
    trade_end = daily["date"].max() if not daily.empty else None

    payload = {
        "schema_version": 1,
        "status": "ready",
        "method": "strict_point_in_time",
        "future_function": False,
        "market_scope": (
            "沪深主板：000001-004999.SZ（排除001001-001199 CDR）"
            "+ 600/601/603/605.SH"
        ),
        "selection_rule": (
            "T日收盘后按历史截面计算60日动量、20日波动率、20日量比，"
            "执行与生产相同的基础过滤，再读取Aquant-Private/main打分，取Top-N。"
        ),
        "execution_rule": (
            "T+1开盘等权调仓，持有至T+1收盘；手续费与滑点按换手计提。"
        ),
        "return_definition": "T+1 close / T+1 open - 1",
        "lookback_trading_days": 60,
        "top_n": top_n,
        "transaction_cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "warmup_sessions": warmup_days,
        "backtest_start": start or selected_files[0].name[:10],
        "backtest_end": end or selected_files[-1].name[:10],
        "trade_start": trade_start,
        "trade_end": trade_end,
        "strategy_source": "Aquant-Private/main",
        "strategy_version": strategy_version,
        "strategy_commit": strategy_commit,
        "overall": overall,
        "annual": annual,
        "monthly": monthly,
        "audit": {
            "historical_files_used": len(selected_files),
            "selected_sessions": len(selected_files) - 1,
            "selected_stock_observations": selected_total,
            "missing_execution_observations": missing_execution_total,
            "production_filter_reused": True,
            "current_universe_not_used_for_history": True,
            "current_names_not_used_for_history": True,
            "future_adjusted_factor_not_used": True,
            "pit_fundamentals_required": False,
            "execution_model": "next_open_to_close",
        },
        "daily": daily.to_dict(orient="records"),
        "selection_audit": selection_rows,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Aquant historical backtest")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise SystemExit("--top-n must be positive")
    if args.cost_bps < 0 or args.slippage_bps < 0:
        raise SystemExit("--cost-bps and --slippage-bps must be >= 0")

    payload = run_backtest(
        start=args.start,
        end=args.end,
        top_n=args.top_n,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
    )
    print(json.dumps(payload["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
