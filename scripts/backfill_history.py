from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from zzshare.client import DataApi

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY = DATA / "history"
STATE_FILE = HISTORY / "_BACKFILL_STATE.json"
VALIDATION_FILE = HISTORY / "_ZZSHARE_VALIDATION.json"
TZ = ZoneInfo("Asia/Shanghai")

TARGET_YEARS = 5
BULK_LIMIT = 6000
VALIDATION_TRADING_DAYS = 3
REQUEST_RETRIES = 4
MIN_VALIDATION_ROWS = 4500
MIN_REQUIRED_COLUMNS = {
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pct_chg",
    "vol",
    "amount",
}


def api_client() -> DataApi:
    token = os.getenv("ZZSHARE_TOKEN", "").strip()
    if token:
        print("ZZSHARE: using configured free token")
        return DataApi(token=token)
    print("ZZSHARE: using anonymous mode")
    return DataApi()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError("empty dataframe")

    out = df.copy()
    rename = {
        "ts_code": "symbol",
        "trade_date": "date",
        "vol": "volume",
        "amount": "amount",
        "pct_chg": "pct_chg",
        "quote_rate": "pct_chg",
        "turnover": "turnover_pct",
        "turnover_rate": "turnover_pct",
        "amp_rate": "amplitude_pct",
        "factor": "factor",
        "pre_close": "pre_close",
        "change": "change",
        "high_limit": "high_limit",
        "low_limit": "low_limit",
        "is_paused": "is_paused",
        "is_st": "is_st",
    }
    for src, dst in rename.items():
        if src in out.columns and src != dst:
            out = out.rename(columns={src: dst})

    if "symbol" not in out.columns or "date" not in out.columns:
        raise RuntimeError(f"missing identity columns: {list(out.columns)}")

    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "volume",
        "amount",
        "turnover_pct",
        "amplitude_pct",
        "factor",
        "high_limit",
        "low_limit",
    ]
    for c in numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out.dropna(subset=["symbol", "date", "close"])


def load_universe(api: DataApi) -> pd.DataFrame:
    df = api.stock_basic(
        exchange="ALL",
        list_status="L",
        fields="ts_code,symbol,name,exchange,list_status",
    )
    if df is None or df.empty:
        raise RuntimeError("zzshare stock_basic returned empty universe")

    df = df.copy()
    df["ts_code"] = df["ts_code"].astype(str).str.upper()
    if "name" in df.columns:
        name = df["name"].astype(str).str.upper()
        before = len(df)
        df = df[~name.str.contains(r"ST|退", regex=True, na=False)].copy()
        print(f"UNIVERSE FILTER: removed {before - len(df)} ST/delisted-related names")

    df = df.drop_duplicates("ts_code").sort_values("ts_code")
    if len(df) < 4000:
        raise RuntimeError(f"universe unexpectedly small: {len(df)}")

    (DATA).mkdir(parents=True, exist_ok=True)
    df.to_json(
        DATA / "universe.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    print(f"UNIVERSE: {len(df)} active non-ST symbols")
    return df


def load_trade_days(api: DataApi, start: str | None = None, end: str | None = None) -> list[str]:
    kwargs = {}
    if start:
        kwargs["day_start"] = start
    if end:
        kwargs["day_end"] = end
    if not kwargs:
        kwargs["days"] = 20

    raw = api.trade_days(**kwargs)
    if raw is None or len(raw) == 0:
        raise RuntimeError("zzshare trade_days returned empty calendar")

    df = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    candidate = next((c for c in ["trade_date", "cal_date", "date", "日期"] if c in df.columns), None)
    if candidate is None:
        raise RuntimeError(f"cannot identify trade-date column: {list(df.columns)}")

    dates = pd.to_datetime(df[candidate], errors="coerce").dropna().dt.strftime("%Y-%m-%d")
    if "is_open" in df.columns:
        open_flag = pd.to_numeric(df["is_open"], errors="coerce")
        dates = dates[open_flag.reindex(df.index).fillna(1).astype(bool)]

    return sorted(set(dates.tolist()), reverse=True)


def request_bulk_day(api: DataApi, trade_date: str) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            df = api.daily(
                trade_date=trade_date.replace("-", ""),
                offset=0,
                limit=BULK_LIMIT,
                export_all=True,
            )
            out = normalize_columns(df)
            if out.empty:
                raise RuntimeError("bulk daily returned empty")
            return out
        except Exception as exc:
            last_exc = exc
            wait = min(60.0, (2 ** (attempt - 1)) + random.random())
            print(f"WARN zzshare {trade_date} attempt {attempt}: {exc}; sleep={wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"zzshare bulk failed for {trade_date}: {last_exc}")


def apply_strategy_universe(df: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    allowed = set(universe["ts_code"].astype(str).str.upper())
    out = df[df["symbol"].isin(allowed)].copy()
    if "is_st" in out.columns:
        out = out[~out["is_st"].astype(str).str.lower().isin({"1", "true", "y", "yes"})].copy()
    return out


def quality_check(df: pd.DataFrame, trade_date: str, minimum_rows: int | None = None) -> dict:
    missing = sorted(MIN_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise RuntimeError(f"{trade_date}: missing columns {missing}")

    unique_symbols = df["symbol"].nunique()
    duplicate_rows = int(df.duplicated(["symbol", "date"]).sum())
    wrong_date = int((df["date"] != trade_date).sum())

    if unique_symbols == 0:
        raise RuntimeError(f"{trade_date}: no symbols")
    if duplicate_rows:
        raise RuntimeError(f"{trade_date}: duplicate symbol/date rows={duplicate_rows}")
    if wrong_date:
        raise RuntimeError(f"{trade_date}: wrong-date rows={wrong_date}")
    if minimum_rows is not None and unique_symbols < minimum_rows:
        raise RuntimeError(
            f"{trade_date}: only {unique_symbols} symbols, below validation minimum {minimum_rows}"
        )

    null_close = int(df["close"].isna().sum())
    if null_close:
        raise RuntimeError(f"{trade_date}: null close rows={null_close}")

    exchanges = {}
    for suffix in (".SH", ".SZ", ".BJ"):
        exchanges[suffix] = int(df["symbol"].str.endswith(suffix).sum())

    return {
        "date": trade_date,
        "rows": len(df),
        "unique_symbols": unique_symbols,
        "duplicates": duplicate_rows,
        "wrong_date": wrong_date,
        "null_close": null_close,
        "markets": exchanges,
    }


def write_daily_file(df: pd.DataFrame, trade_date: str) -> Path:
    HISTORY.mkdir(parents=True, exist_ok=True)
    path = HISTORY / f"{trade_date}.csv.gz"
    ordered = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "volume",
        "amount",
        "pct_chg",
        "change",
        "turnover_pct",
        "amplitude_pct",
        "factor",
        "high_limit",
        "low_limit",
        "is_paused",
        "is_st",
    ]
    cols = [c for c in ordered if c in df.columns]
    df[cols].sort_values("symbol").to_csv(path, index=False, compression="gzip")
    return path


def git_checkpoint(message: str) -> None:
    subprocess.run(["git", "add", "data/history", "data/universe.json"], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        return
    subprocess.run(["git", "config", "user.name", "aquant-bot"], check=True)
    subprocess.run(
        ["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"],
        check=True,
    )
    subprocess.run(["git", "commit", "-m", message], check=True)
    subprocess.run(["git", "push"], check=True)


def validate_bulk(days: int = VALIDATION_TRADING_DAYS) -> None:
    api = api_client()
    universe = load_universe(api)
    recent = load_trade_days(api)[:days]
    if not recent:
        raise RuntimeError("no recent trading days returned")

    results = []
    for trade_date in recent:
        raw = request_bulk_day(api, trade_date)
        filtered = apply_strategy_universe(raw, universe)
        result = quality_check(filtered, trade_date, MIN_VALIDATION_ROWS)
        result["raw_rows"] = len(raw)
        result["filtered_rows"] = len(filtered)
        results.append(result)
        print(f"VALIDATED {trade_date}: raw={len(raw)} filtered={len(filtered)} markets={result['markets']}")

    payload = {
        "validated_at": datetime.now(TZ).isoformat(),
        "source": "zzshare bulk daily",
        "days": results,
        "universe_size": len(universe),
        "bulk_limit": BULK_LIMIT,
        "anonymous_mode_allowed": True,
    }
    HISTORY.mkdir(parents=True, exist_ok=True)
    VALIDATION_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    git_checkpoint("test: validate zzshare daily bulk source")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    HISTORY.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def backfill() -> None:
    api = api_client()
    universe = load_universe(api)

    today = datetime.now(TZ).date()
    try:
        target_start = today.replace(year=today.year - TARGET_YEARS)
    except ValueError:
        target_start = today.replace(year=today.year - TARGET_YEARS, day=28)

    trade_days = load_trade_days(
        api,
        start=target_start.strftime("%Y%m%d"),
        end=today.strftime("%Y%m%d"),
    )
    if not trade_days:
        raise RuntimeError("no trading days in target range")

    state = load_state()
    completed = set(state.get("completed_dates", [])) if state else set()

    print(
        f"BACKFILL: {len(trade_days)} trading days, "
        f"{target_start} -> {today}, completed={len(completed)}"
    )

    for trade_date in trade_days:
        if trade_date in completed:
            continue

        raw = request_bulk_day(api, trade_date)
        filtered = apply_strategy_universe(raw, universe)
        result = quality_check(filtered, trade_date)

        path = write_daily_file(filtered, trade_date)

        state = state or {
            "status": "running",
            "started_at": datetime.now(TZ).isoformat(),
            "target_start": str(target_start),
            "initial_end": str(today),
            "direction": "near_to_far",
            "source": "zzshare bulk daily",
            "bulk_limit": BULK_LIMIT,
            "completed_dates": [],
            "days_completed": 0,
            "rows_written": 0,
        }
        state["completed_dates"].append(trade_date)
        state["completed_dates"] = sorted(set(state["completed_dates"]), reverse=True)
        state["days_completed"] = len(state["completed_dates"])
        state["rows_written"] += len(filtered)
        state["last_day"] = result
        state["last_file"] = str(path.relative_to(ROOT))
        save_state(state)

        git_checkpoint(f"data: checkpoint full-market {trade_date}")
        print(
            f"CHECKPOINT {trade_date}: rows={len(filtered)} "
            f"symbols={result['unique_symbols']} days={state['days_completed']}/{len(trade_days)}"
        )

    state["status"] = "complete"
    state["completed_at"] = datetime.now(TZ).isoformat()
    save_state(state)

    (HISTORY / "_BACKFILL_COMPLETE").write_text(
        json.dumps(
            {
                "completed_at": state["completed_at"],
                "target_start": str(target_start),
                "end": str(today),
                "years": TARGET_YEARS,
                "trading_days": len(trade_days),
                "days_completed": state["days_completed"],
                "symbols_universe": len(universe),
                "source": "zzshare bulk daily",
                "format": "daily CSV gzip",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    git_checkpoint("data: complete five-year full-market history")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "backfill"], default="validate")
    parser.add_argument("--days", type=int, default=VALIDATION_TRADING_DAYS)
    args = parser.parse_args()

    if args.mode == "validate":
        validate_bulk(args.days)
    else:
        backfill()


if __name__ == "__main__":
    main()
