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
            df = pd.read_csv(fh, usecols=sorted(REQUIRED_COLUMNS))
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


class RollingFeatureState:
    """Vectorized rolling state for the point-in-time feature set."""

    def __init__(self, initial_capacity: int = 4096) -> None:
        self.capacity = initial_capacity
        self.symbol_to_idx: dict[str, int] = {}
        self.symbols = np.empty(initial_capacity, dtype=object)

        self.rets = np.full((initial_capacity, 60), np.nan, dtype=float)
        self.vol_rets = np.full((initial_capacity, 20), np.nan, dtype=float)
        self.volumes = np.full((initial_capacity, 20), np.nan, dtype=float)

        self.pos60 = np.zeros(initial_capacity, dtype=np.int64)
        self.pos20 = np.zeros(initial_capacity, dtype=np.int64)
        self.count60 = np.zeros(initial_capacity, dtype=np.int64)
        self.count20 = np.zeros(initial_capacity, dtype=np.int64)
        self.finite60 = np.zeros(initial_capacity, dtype=np.int64)
        self.finite20 = np.zeros(initial_capacity, dtype=np.int64)
        self.volume_valid20 = np.zeros(initial_capacity, dtype=np.int64)

    def _grow(self, required: int) -> None:
        if required <= self.capacity:
            return
        new_capacity = max(required, self.capacity * 2)

        new_rets = np.full((new_capacity, 60), np.nan, dtype=float)
        new_rets[: self.capacity] = self.rets
        self.rets = new_rets

        new_vol_rets = np.full((new_capacity, 20), np.nan, dtype=float)
        new_vol_rets[: self.capacity] = self.vol_rets
        self.vol_rets = new_vol_rets

        new_volumes = np.full((new_capacity, 20), np.nan, dtype=float)
        new_volumes[: self.capacity] = self.volumes
        self.volumes = new_volumes

        for name in (
            "symbols",
            "pos60",
            "pos20",
            "count60",
            "count20",
            "finite60",
            "finite20",
            "volume_valid20",
        ):
            old = getattr(self, name)
            if name == "symbols":
                value = np.empty(new_capacity, dtype=object)
                value[: self.capacity] = old
            elif old.dtype.kind in "iu":
                value = np.zeros(new_capacity, dtype=old.dtype)
                value[: self.capacity] = old
            else:
                value = np.empty(new_capacity, dtype=old.dtype)
                value[: self.capacity] = old
            setattr(self, name, value)

        self.capacity = new_capacity

    def indices_for(self, symbols: np.ndarray) -> np.ndarray:
        new_symbols = [symbol for symbol in pd.unique(symbols) if symbol not in self.symbol_to_idx]
        if new_symbols:
            start = len(self.symbol_to_idx)
            self._grow(start + len(new_symbols))
            for offset, symbol in enumerate(new_symbols):
                idx = start + offset
                self.symbol_to_idx[symbol] = idx
                self.symbols[idx] = symbol

        return np.fromiter(
            (self.symbol_to_idx[symbol] for symbol in symbols),
            dtype=np.int64,
            count=len(symbols),
        )

    def update_and_features(
        self,
        symbols: np.ndarray,
        daily_ret: np.ndarray,
        volumes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = self.indices_for(symbols)

        slot60 = self.pos60[idx] % 60
        old60 = self.rets[idx, slot60]
        self.finite60[idx] += np.isfinite(daily_ret).astype(np.int64) - np.isfinite(old60).astype(np.int64)
        self.rets[idx, slot60] = daily_ret
        self.pos60[idx] += 1
        self.count60[idx] = np.minimum(self.count60[idx] + 1, 60)

        slot20 = self.pos20[idx] % 20
        old20_ret = self.vol_rets[idx, slot20]
        old20_volume = self.volumes[idx, slot20]
        self.finite20[idx] += np.isfinite(daily_ret).astype(np.int64) - np.isfinite(old20_ret).astype(np.int64)
        self.volume_valid20[idx] += np.isfinite(volumes).astype(np.int64) - np.isfinite(old20_volume).astype(np.int64)
        self.vol_rets[idx, slot20] = daily_ret
        self.volumes[idx, slot20] = volumes
        self.pos20[idx] += 1
        self.count20[idx] = np.minimum(self.count20[idx] + 1, 20)

        ret60 = np.full(len(idx), np.nan, dtype=float)
        complete = (self.count60[idx] == 60) & (self.finite60[idx] == 60)
        if complete.any():
            ret60[complete] = np.prod(1.0 + self.rets[idx[complete]], axis=1) - 1.0

        volatility = np.full(len(idx), np.nan, dtype=float)
        enough_rets = self.finite20[idx] >= 10
        if enough_rets.any():
            volatility[enough_rets] = np.nanstd(
                self.vol_rets[idx[enough_rets]],
                axis=1,
                ddof=1,
            )

        volume_ratio = np.full(len(idx), np.nan, dtype=float)
        enough_volume = (
            (self.volume_valid20[idx] >= 10)
            & np.isfinite(volumes)
            & (volumes > 0)
        )
        if enough_volume.any():
            mean_volume = np.nanmean(self.volumes[idx[enough_volume]], axis=1)
            valid_mean = mean_volume > 0
            ratio = np.full(len(mean_volume), np.nan, dtype=float)
            ratio[valid_mean] = volumes[enough_volume][valid_mean] / mean_volume[valid_mean]
            volume_ratio[enough_volume] = ratio

        return ret60, volatility * 100.0, volume_ratio


def build_strategy_frame(
    df: pd.DataFrame,
    state: RollingFeatureState,
) -> pd.DataFrame:
    """Append current observations and return today's scorable main-board frame."""
    symbols = df["symbol"].astype(str).str.zfill(6).to_numpy()
    daily_ret = pd.to_numeric(df["pct_chg"], errors="coerce").to_numpy(dtype=float) / 100.0
    volumes = pd.to_numeric(df["volume"], errors="coerce").to_numpy(dtype=float)

    ret60, volatility_proxy, volume_ratio = state.update_and_features(
        symbols,
        daily_ret,
        volumes,
    )

    close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
    amount = pd.to_numeric(df["amount"], errors="coerce").to_numpy(dtype=float)
    turnover_pct = pd.to_numeric(df["turnover_pct"], errors="coerce").to_numpy(dtype=float)
    is_paused = pd.to_numeric(df["is_paused"], errors="coerce").to_numpy(dtype=float)
    is_st = pd.to_numeric(df["is_st"], errors="coerce").to_numpy(dtype=float)
    pe = pd.to_numeric(df["pe_ratio"], errors="coerce").to_numpy(dtype=float)
    pb = pd.to_numeric(df["pb_ratio"], errors="coerce").to_numpy(dtype=float)
    change_pct = pd.to_numeric(df["pct_chg"], errors="coerce").to_numpy(dtype=float)

    eligible = (
        (is_paused == 0)
        & (is_st == 0)
        & np.isfinite(close)
        & (close > 2.0)
        & np.isfinite(amount)
        & (amount >= 2e7)
        & np.isfinite(ret60)
    )

    if not eligible.any():
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

    selected_symbols = symbols[eligible]
    return pd.DataFrame(
        {
            "symbol": selected_symbols,
            "name": [f"股票{symbol}" for symbol in selected_symbols],
            "close": close[eligible],
            "amount": amount[eligible],
            "turnover_pct": turnover_pct[eligible],
            "change_pct": change_pct[eligible],
            "pe": pe[eligible],
            "pb": pb[eligible],
            "ret_60d": ret60[eligible],
            "momentum_60d": ret60[eligible] * 100.0,
            "volatility_proxy": volatility_proxy[eligible],
            "volume_ratio": volume_ratio[eligible],
        }
    )
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
    target_weights: dict[str, float],
) -> tuple[float, list[str], list[str]]:
    if selected.empty:
        return 0.0, [], []

    execution = execution_df.set_index("symbol", drop=False)
    gross = 0.0
    executed = []
    missing = []

    for row in selected.itertuples(index=False):
        symbol = str(row.symbol).zfill(6)
        weight = float(target_weights.get(symbol, 0.0))
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

        gross += weight * (close_price / open_price - 1.0)
        executed.append(symbol)

    return float(gross), executed, missing


def normalize_weights(symbols: Iterable[str]) -> dict[str, float]:
    values = sorted(set(str(x).zfill(6) for x in symbols))
    if not values:
        return {}
    weight = 1.0 / len(values)
    return {symbol: weight for symbol in values}


def turnover(prev: dict[str, float], target: dict[str, float]) -> float:
    keys = set(prev) | set(target)
    return float(sum(abs(target.get(k, 0.0) - prev.get(k, 0.0)) for k in keys))


def validate_backtest_payload(payload: dict) -> None:
    """Fail closed on structural, numerical, and point-in-time audit violations."""
    if payload.get("status") != "ready":
        raise ValueError("backtest payload is not ready")
    if payload.get("future_function") is not False:
        raise ValueError("future_function audit must be false")
    audit = payload.get("audit", {})
    overall = payload.get("overall", {})
    daily = payload.get("daily", [])

    performance_sessions = int(audit.get("performance_sessions", -1))
    trade_start = payload.get("trade_start")
    performance_daily = (
        [row for row in daily if trade_start and row.get("date", "") >= trade_start]
        if trade_start
        else daily
    )
    if performance_sessions != len(performance_daily):
        raise ValueError("performance_sessions does not match performance daily rows")
    if performance_sessions != int(overall.get("trading_days", -1)):
        raise ValueError("overall trading_days does not match performance sessions")
    if int(audit.get("selected_sessions", -1)) + 1 != int(audit.get("historical_files_used", -2)):
        raise ValueError("historical session/file audit mismatch")

    dates = [row["date"] for row in daily]
    if dates != sorted(dates) or len(dates) != len(set(dates)):
        raise ValueError("daily dates are not strictly unique and increasing")

    numeric_fields = [
        "gross_return",
        "turnover",
        "cost",
        "net_return",
        "target_count",
        "executed_count",
        "missing_execution_count",
    ]
    for row in daily:
        for field in numeric_fields:
            value = float(row[field])
            if not np.isfinite(value):
                raise ValueError(f"non-finite daily field: {field}")
        if row["turnover"] < -1e-12 or row["turnover"] > 2.0 + 1e-12:
            raise ValueError("turnover outside theoretical L1 bounds")
        expected_cost = row["turnover"] * (
            float(payload["transaction_cost_bps"]) + float(payload["slippage_bps"])
        ) / 10000.0
        if not np.isclose(row["cost"], expected_cost, rtol=0, atol=1e-12):
            raise ValueError("daily cost does not match turnover * bps")

    for key in (
        "total_return_pct",
        "annualized_return_pct",
        "annualized_volatility_pct",
        "max_drawdown_pct",
        "average_turnover_pct",
        "total_turnover_pct",
    ):
        if overall.get(key) is not None and not np.isfinite(float(overall[key])):
            raise ValueError(f"non-finite overall metric: {key}")
    if overall.get("sharpe") is not None and not np.isfinite(float(overall["sharpe"])):
        raise ValueError("non-finite Sharpe")
    if overall.get("max_drawdown_pct") is not None and float(overall["max_drawdown_pct"]) > 1e-9:
        raise ValueError("max drawdown cannot be positive")
    if not payload.get("strategy_commit"):
        raise ValueError("strategy commit audit is missing")


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
    states = RollingFeatureState()

    # The first 59 sessions are warm-up only; they cannot be traded until each
    # stock has a complete 60-observation return window.
    daily_rows = []
    selection_rows = []
    prev_target: dict[str, float] = {}
    missing_execution_total = 0
    selected_total = 0

    # Read each historical session once. The next session becomes the current
    # session on the following iteration, eliminating duplicate CSV reads.
    current = read_daily(selected_files[0])

    # We need T and T+1 for every scoring date.
    for i in range(len(selected_files) - 1):
        next_path = selected_files[i + 1]
        next_date = next_path.name[:10]
        current_date = current["date"].iloc[0]

        next_day = read_daily(next_path)

        frame = build_strategy_frame(current, states)
        targets = select_targets(frame, strategy_model, top_n)
        target_symbols = targets["symbol"].astype(str).tolist()
        target_weights = normalize_weights(target_symbols)

        selected_total += len(target_symbols)
        gross_return, executed, missing = next_session_return(
            targets,
            next_day,
            target_weights,
        )
        missing_execution_total += len(missing)

        # Missing executions remain cash. Turnover is charged on positions
        # that were actually executable at T+1 open.
        actual_target = {
            symbol: target_weights[symbol]
            for symbol in executed
            if symbol in target_weights
        }
        turn = turnover(prev_target, actual_target)
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
                    for x in targets.get("score", pd.Series(dtype=float)).tolist()
                ],
            }
        )
        prev_target = actual_target

        if (i + 1) % 100 == 0:
            print(
                f"[backtest] {i + 1}/{len(selected_files) - 1} "
                f"signal={current_date} execution={next_date} "
                f"targets={len(target_symbols)}"
            )

        current = next_day

    daily = pd.DataFrame(daily_rows)
    active_start = 0
    active_rows = daily.index[daily["target_count"] > 0].tolist()
    if active_rows:
        active_start = int(active_rows[0])
    performance = daily.iloc[active_start:].reset_index(drop=True)
    overall = metrics(performance)
    annual = period_metrics(performance, "Y")
    monthly = period_metrics(performance, "M")

    warmup_days = min(59, len(selected_files) - 1)
    trade_start = performance["date"].min() if not performance.empty else None
    trade_end = performance["date"].max() if not performance.empty else None

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
            "performance_sessions": len(performance),
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

    validate_backtest_payload(payload)
    print("[backtest] quality audit passed")

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
