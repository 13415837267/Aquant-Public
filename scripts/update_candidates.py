from __future__ import annotations

import gzip
import numpy as np
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from scripts.market_scope import is_main_board_symbol

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HISTORY_DIR = DATA_DIR / "history"
DATA_FILE = DATA_DIR / "candidates.json"
PRIVATE_STRATEGY_PATH = os.environ.get("AQUANT_PRIVATE_STRATEGY_PATH")
PRIVATE_STRATEGY_COMMIT = os.environ.get("AQUANT_PRIVATE_STRATEGY_COMMIT")

REQUIRED_COLUMNS = {
    "symbol", "date", "close", "pre_close", "volume", "amount", "pct_chg",
    "turnover_pct", "amplitude_pct", "is_paused", "is_st", "market_cap",
    "circulating_market_cap", "pe_ratio", "pb_ratio",
}


def _load_current_names() -> dict[str, str]:
    path = DATA_DIR / "universe.json"
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(row.get("symbol", "")).zfill(6): str(row.get("name", "")).strip()
        for row in rows
        if is_main_board_symbol(row.get("ts_code"))
        and row.get("symbol")
        and row.get("name")
    }


def _load_private_strategy():
    if not PRIVATE_STRATEGY_PATH:
        raise RuntimeError("AQUANT_PRIVATE_STRATEGY_PATH is required; Public must not run an independent strategy")
    strategy_root = Path(PRIVATE_STRATEGY_PATH).resolve()
    if not (strategy_root / "strategy" / "model.py").exists():
        raise RuntimeError(f"Private strategy not found: {strategy_root / 'strategy' / 'model.py'}")
    sys.path.insert(0, str(strategy_root))
    model = importlib.import_module("strategy.model")
    version = importlib.import_module("strategy.version")
    commit = PRIVATE_STRATEGY_COMMIT
    if not commit:
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(strategy_root), "rev-parse", "HEAD"],
                text=True,
            ).strip()
        except Exception:
            commit = "unknown"
    return model, str(getattr(version, "STRATEGY_VERSION", "unknown")), commit


def _history_files() -> list[Path]:
    files = sorted(HISTORY_DIR.glob("????/*.csv.gz"), key=lambda path: path.name[:10], reverse=True)
    if not files:
        raise RuntimeError("No historical daily files found under data/history/YYYY/")
    return files


def _read_history_window(max_files: int = 61) -> tuple[pd.DataFrame, list[str]]:
    files = _history_files()[:max_files]
    frames, dates = [], []
    for path in files:
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                frame = pd.read_csv(fh)
        except Exception as exc:
            raise RuntimeError(f"Failed to read {path.name}: {exc}") from exc
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path.name} missing required columns: {sorted(missing)}")
        frame = frame.loc[frame["symbol"].map(is_main_board_symbol)].copy()
        if frame.empty:
            raise RuntimeError(f"{path.name} contains no Shanghai/Shenzhen main-board rows")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        file_date = path.name[:10]
        bad_dates = int(frame["date"].ne(file_date).sum())
        if bad_dates:
            raise RuntimeError(f"{path.name} contains {bad_dates} rows with a mismatched date")
        frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
        dates.append(file_date)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")
    return data.sort_values(["symbol", "date"]), dates


def _prepare_strategy_frame(history: pd.DataFrame, latest_date: str) -> pd.DataFrame:
    hist = history.copy()
    hist = hist.loc[hist["symbol"].map(is_main_board_symbol)].copy()
    if hist.empty:
        raise RuntimeError("No Shanghai/Shenzhen main-board rows in candidate input")
    numeric_cols = [
        "close", "pre_close", "volume", "amount", "turnover_pct", "pct_chg",
        "pe_ratio", "pb_ratio",
    ]
    for col in numeric_cols:
        hist[col] = pd.to_numeric(hist[col], errors="coerce")
    hist = hist.sort_values(["symbol", "date"])
    grouped = hist.groupby("symbol", sort=False)

    # Provider pct_chg is the economic daily return series. Compounding it
    # avoids raw-close discontinuities on corporate-action dates and does not
    # inject hindsight from a future-adjusted factor series.
    hist["daily_ret"] = hist["pct_chg"] / 100.0
    hist["gross_ret"] = 1.0 + hist["daily_ret"]
    hist["ret_1d"] = hist["daily_ret"]
    hist["ret_60d"] = (
        grouped["gross_ret"]
        .rolling(60, min_periods=60)
        .apply(np.prod, raw=True)
        .reset_index(level=0, drop=True)
        - 1.0
    )
    hist["volatility_proxy"] = (
        grouped["daily_ret"].rolling(20, min_periods=10).std()
        .reset_index(level=0, drop=True) * 100
    )
    avg_volume_20d = (
        grouped["volume"].rolling(20, min_periods=10).mean()
        .reset_index(level=0, drop=True)
    )
    hist["volume_ratio"] = hist["volume"] / avg_volume_20d.replace(0, float("nan"))

    latest = hist.loc[hist["date"].eq(latest_date)].copy()
    names = _load_current_names()
    latest["name"] = latest["symbol"].map(names).fillna(
        latest.get("name", latest["symbol"]).astype(str)
    )

    excluded_name = latest["name"].str.contains(r"ST|退", case=False, na=False)
    for col in ["is_paused", "is_st", "close", "amount"]:
        latest[col] = pd.to_numeric(latest[col], errors="coerce")
    latest["is_paused"] = latest["is_paused"].fillna(0)
    latest["is_st"] = latest["is_st"].fillna(0)

    usable = (
        ~excluded_name
        & latest["is_st"].eq(0)
        & latest["is_paused"].eq(0)
        & latest["close"].gt(2)
        & latest["amount"].ge(2e7)
        & latest["ret_60d"].notna()
    )
    latest = latest.loc[usable].copy()

    strategy_frame = latest.rename(columns={
        "pe_ratio": "pe",
        "pb_ratio": "pb",
        "pct_chg": "change_pct",
    })[
        ["symbol", "name", "close", "amount", "turnover_pct", "change_pct",
         "pe", "pb", "ret_60d", "volatility_proxy", "volume_ratio"]
    ].copy()
    strategy_frame["momentum_60d"] = strategy_frame["ret_60d"] * 100
    return strategy_frame


def build_candidates(history: pd.DataFrame, strategy_model, strategy_version: str, strategy_commit: str) -> dict:
    latest_date = history["date"].dropna().max()
    if not latest_date:
        raise RuntimeError("History has no valid date")

    strategy_frame = _prepare_strategy_frame(history, latest_date)
    if strategy_frame.empty:
        raise RuntimeError("No usable stocks after strategy universe filters")

    scored = strategy_model.score_universe(strategy_frame)
    scored = scored.head(30).reset_index(drop=True)

    rows = []
    for idx, row in scored.iterrows():
        flags = []
        if pd.notna(row["volatility_proxy"]) and row["volatility_proxy"] > 8:
            flags.append("高波动")
        if pd.notna(row["change_pct"]) and row["change_pct"] < -7:
            flags.append("当日跌幅<-7%")
        rows.append({
            "rank": idx + 1,
            "symbol": row["symbol"],
            "name": row["name"],
            "price": round(float(row["close"]), 3),
            "change_pct": round(float(row["change_pct"]), 3) if pd.notna(row["change_pct"]) else None,
            "momentum_60d": round(float(row["momentum_60d"]) * 1.0, 3) if pd.notna(row["momentum_60d"]) else None,
            "turnover_pct": round(float(row["turnover_pct"]), 3) if pd.notna(row["turnover_pct"]) else None,
            "pb": round(float(row["pb"]), 3) if pd.notna(row["pb"]) and row["pb"] > 0 else None,
            "pe": round(float(row["pe"]), 3) if pd.notna(row["pe"]) and row["pe"] > 0 else None,
            "amount": round(float(row["amount"]), 2) if pd.notna(row["amount"]) else None,
            "volatility_proxy": round(float(row["volatility_proxy"]), 3) if pd.notna(row["volatility_proxy"]) else None,
            "score": round(float(row["score"]), 3),
            "flags": flags,
        })

    weights = getattr(strategy_model, "WEIGHTS", None)
    return {
        "as_of": f"{latest_date}T18:00:00+08:00",
        "timezone": "Asia/Shanghai",
        "source": "Aquant-Public data/history",
        "status": "ready",
        "strategy_source": "Aquant-Private/main",
        "strategy_version": strategy_version,
        "strategy_commit": strategy_commit,
        "market_scope": "沪深主板：000001-004999.SZ（排除001001-001199 CDR）+ 600/601/603/605.SH",
        "universe": "沪深主板；排除 ST/退市相关标的、停牌、价格≤2元、最近交易日成交额<2000万元；60日动量必须有完整窗口",
        "lookback_trading_days": 60,
        "candidates": rows,
        "factor_weights": weights,
    }


def main() -> None:
    strategy_model, strategy_version, strategy_commit = _load_private_strategy()
    history, dates = _read_history_window(61)
    snapshot = build_candidates(history, strategy_model, strategy_version, strategy_commit)
    previous = None
    if DATA_FILE.exists():
        try:
            previous = json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({
        "changed": previous != snapshot,
        "rows": len(snapshot["candidates"]),
        "as_of": snapshot["as_of"],
        "strategy_version": strategy_version,
        "strategy_commit": strategy_commit,
        "history_files_used": len(dates),
        "oldest_file_used": dates[-1],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
