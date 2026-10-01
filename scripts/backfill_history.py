from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from zzshare.client import DataApi

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY = DATA / "history"
FUNDAMENTALS = DATA / "fundamentals"
STATE_FILE = HISTORY / "_BACKFILL_STATE.json"
VALIDATION_FILE = HISTORY / "_ZZSHARE_VALIDATION.json"
FUND_STATE_FILE = FUNDAMENTALS / "_FUNDAMENTALS_STATE.json"
FUND_VALIDATION_FILE = FUNDAMENTALS / "_FUNDAMENTALS_VALIDATION.json"
TZ = ZoneInfo("Asia/Shanghai")

TARGET_YEARS = 5
BULK_LIMIT = 6000
VALIDATION_TRADING_DAYS = 3
REQUEST_RETRIES = 4
MIN_VALIDATION_ROWS = 4500
FINANCE_LIMIT = 40000
FINANCE_CHUNK_DAYS = 7
FINANCIAL_START_BUFFER_YEARS = 1

DAILY_REQUIRED = {
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "pct_chg",
    "volume",
    "amount",
    "factor",
    "high_limit",
    "low_limit",
    "turnover_pct",
    "amplitude_pct",
    "is_paused",
    "is_st",
}

FINANCE_TABLES = {
    "indicator": "finance_indicator",
    "income": "finance_income",
    "balance": "finance_balance",
    "cash_flow": "finance_cash_flow",
}


def api_client() -> DataApi:
    token = os.getenv("ZZSHARE_TOKEN", "").strip()
    if token:
        print("ZZSHARE: using configured token")
        return DataApi(token=token, timeout=30)
    print("ZZSHARE: using anonymous mode")
    return DataApi(timeout=30)


def first_series(df: pd.DataFrame, name: str):
    value = df[name]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return value


def normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError("empty daily dataframe")

    out = df.copy()

    # zzshare all-fields mode currently exposes both native and Tushare-compatible
    # names. Normalize them into one stable internal schema.
    aliases = {
        "ts_code": "symbol",
        "trade_date": "date",
        "prev_close": "pre_close",
        "quote_rate": "pct_chg",
        "vol": "volume",
        "turnover": "amount",
        "turnover_rate": "turnover_pct",
        "amp_rate": "amplitude_pct",
    }
    for src, dst in aliases.items():
        if src in out.columns and dst not in out.columns:
            out = out.rename(columns={src: dst})

    if "symbol" not in out.columns or "date" not in out.columns:
        raise RuntimeError(f"daily missing identity columns: {list(out.columns)}")

    out["symbol"] = first_series(out, "symbol").astype(str).str.strip().str.upper()
    out["date"] = pd.to_datetime(first_series(out, "date"), errors="coerce").dt.strftime("%Y-%m-%d")

    numeric_cols = [
        "open", "high", "low", "close", "pre_close", "change", "pct_chg",
        "volume", "amount", "turnover_pct", "amplitude_pct", "factor",
        "high_limit", "low_limit", "avg_price",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(first_series(out, col), errors="coerce")

    for col in ["is_paused", "is_st"]:
        if col in out.columns:
            out[col] = first_series(out, col)

    return out.dropna(subset=["symbol", "date", "close"]).reset_index(drop=True)


def normalize_finance(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise RuntimeError("empty finance dataframe")

    out = df.copy()

    code_col = next((c for c in ["code", "ts_code", "symbol"] if c in out.columns), None)
    if code_col is None:
        raise RuntimeError(f"finance missing code column: {list(out.columns)}")
    if code_col != "symbol":
        out = out.rename(columns={code_col: "symbol"})

    report_col = next(
        (c for c in ["statDate", "stat_date", "report_date", "trade_date"] if c in out.columns),
        None,
    )
    pub_col = next(
        (c for c in ["pubDate", "pub_date", "publish_date"] if c in out.columns),
        None,
    )

    out["symbol"] = first_series(out, "symbol").astype(str).str.strip().str.upper()
    if report_col:
        out["report_date"] = pd.to_datetime(
            first_series(out, report_col), errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    else:
        out["report_date"] = None

    if pub_col:
        out["pub_date"] = pd.to_datetime(
            first_series(out, pub_col), errors="coerce"
        ).dt.strftime("%Y-%m-%d")
    else:
        out["pub_date"] = None

    # Convert every non-identity column where possible. This preserves vendor
    # column names while making the database numeric-query friendly.
    identity = {"symbol", "report_date", "pub_date"}
    for col in out.columns:
        if col not in identity:
            try:
                series = first_series(out, col)
                try:
                    out[col] = pd.to_numeric(series, errors="raise")
                except (TypeError, ValueError):
                    out[col] = series
            except Exception:
                pass

    return out.drop_duplicates().reset_index(drop=True)


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
        print(f"STRATEGY UNIVERSE: removed {before - len(df)} ST/delisted-related names")

    df = df.drop_duplicates("ts_code").sort_values("ts_code")
    if len(df) < 4000:
        raise RuntimeError(f"universe unexpectedly small: {len(df)}")

    DATA.mkdir(parents=True, exist_ok=True)
    df.to_json(
        DATA / "universe.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    print(f"STRATEGY UNIVERSE: {len(df)} active non-ST symbols")
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
    if candidate is None and len(df.columns) == 1:
        candidate = df.columns[0]
    if candidate is None:
        raise RuntimeError(f"cannot identify trade-date column: {list(df.columns)}")

    dates = pd.to_datetime(df[candidate], errors="coerce")
    if "is_open" in df.columns:
        open_flag = pd.to_numeric(df["is_open"], errors="coerce")
        dates = dates.where(open_flag.reindex(df.index).fillna(1).astype(bool))

    return sorted(set(dates.dropna().dt.strftime("%Y-%m-%d").tolist()), reverse=True)


def request_bulk_day(api: DataApi, trade_date: str) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            df = api.daily(
                trade_date=trade_date.replace("-", ""),
                offset=0,
                limit=BULK_LIMIT,
                fields="all",
            )
            out = normalize_daily(df)
            if out.empty:
                raise RuntimeError("bulk daily returned empty")
            return out
        except Exception as exc:
            last_exc = exc
            wait = min(90.0, (2 ** (attempt - 1)) + random.random())
            print(
                f"WARN daily {trade_date} attempt {attempt}: "
                f"{type(exc).__name__}: {exc}; sleep={wait:.1f}s"
            )
            time.sleep(wait)
    raise RuntimeError(f"zzshare bulk failed for {trade_date}: {last_exc}")


def quality_check_daily(df: pd.DataFrame, trade_date: str, minimum_rows: int | None = None) -> dict:
    missing = sorted(DAILY_REQUIRED - set(df.columns))
    if missing:
        raise RuntimeError(f"{trade_date}: missing daily columns {missing}")

    unique_symbols = int(df["symbol"].nunique())
    duplicate_rows = int(df.duplicated(["symbol", "date"]).sum())
    wrong_date = int((df["date"] != trade_date).sum())
    null_close = int(df["close"].isna().sum())

    if unique_symbols == 0:
        raise RuntimeError(f"{trade_date}: no symbols")
    if duplicate_rows:
        raise RuntimeError(f"{trade_date}: duplicate symbol/date rows={duplicate_rows}")
    if wrong_date:
        raise RuntimeError(f"{trade_date}: wrong-date rows={wrong_date}")
    if null_close:
        raise RuntimeError(f"{trade_date}: null close rows={null_close}")
    if minimum_rows is not None and unique_symbols < minimum_rows:
        raise RuntimeError(
            f"{trade_date}: only {unique_symbols} symbols, below validation minimum {minimum_rows}"
        )

    markets = {}
    for suffix in (".SH", ".SZ", ".BJ"):
        markets[suffix] = int(df["symbol"].str.endswith(suffix).sum())

    completeness = {}
    for c in sorted(DAILY_REQUIRED):
        completeness[c] = round(float(df[c].notna().mean()), 6)

    return {
        "date": trade_date,
        "rows": len(df),
        "unique_symbols": unique_symbols,
        "duplicates": duplicate_rows,
        "wrong_date": wrong_date,
        "null_close": null_close,
        "markets": markets,
        "completeness": completeness,
    }


def write_daily_file(df: pd.DataFrame, trade_date: str) -> Path:
    HISTORY.mkdir(parents=True, exist_ok=True)
    path = HISTORY / f"{trade_date}.csv.gz"
    ordered = [
        "date", "symbol", "open", "high", "low", "close", "pre_close",
        "volume", "amount", "pct_chg", "change", "turnover_pct",
        "amplitude_pct", "factor", "avg_price", "high_limit", "low_limit",
        "is_paused", "is_st",
    ]
    cols = [c for c in ordered if c in df.columns]
    df[cols].sort_values("symbol").to_csv(
        path, index=False, compression="gzip"
    )
    return path


def git_checkpoint(paths: list[str], message: str) -> None:
    subprocess.run(["git", "add", "--", *paths], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        return
    subprocess.run(["git", "config", "user.name", "aquant-bot"], check=True)
    subprocess.run(
        [
            "git", "config", "user.email",
            "41898282+github-actions[bot]@users.noreply.github.com",
        ],
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
        result = quality_check_daily(raw, trade_date, MIN_VALIDATION_ROWS)
        result["raw_rows"] = len(raw)
        result["strategy_universe_overlap"] = int(
            raw["symbol"].isin(set(universe["ts_code"])).sum()
        )
        results.append(result)
        print(
            f"VALIDATED {trade_date}: raw={len(raw)} "
            f"symbols={result['unique_symbols']} "
            f"markets={result['markets']} "
            f"factor_nonnull={result['completeness']['factor']}"
        )

    payload = {
        "validated_at": datetime.now(TZ).isoformat(),
        "source": "zzshare daily all-fields bulk",
        "days": results,
        "universe_size": len(universe),
        "bulk_limit": BULK_LIMIT,
        "advanced_fields": sorted(DAILY_REQUIRED),
    }
    HISTORY.mkdir(parents=True, exist_ok=True)
    VALIDATION_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    git_checkpoint(
        ["data/history/_ZZSHARE_VALIDATION.json", "data/universe.json"],
        "test: validate full-field daily bulk source",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    HISTORY.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def backfill_daily() -> None:
    api = api_client()
    universe = load_universe(api)

    today = datetime.now(TZ).date()
    try:
        target_start = today.replace(year=today.year - TARGET_YEARS)
    except ValueError:
        target_start = today.replace(
            year=today.year - TARGET_YEARS,
            day=28,
        )

    trade_days = load_trade_days(
        api,
        start=target_start.strftime("%Y%m%d"),
        end=today.strftime("%Y%m%d"),
    )
    if not trade_days:
        raise RuntimeError("no trading days in target range")

    state = load_state()
    completed = set(state.get("completed_dates", [])) if state else set()

    state = state or {
        "status": "running",
        "started_at": datetime.now(TZ).isoformat(),
        "target_start": str(target_start),
        "initial_end": str(today),
        "direction": "near_to_far",
        "source": "zzshare daily all-fields bulk",
        "bulk_limit": BULK_LIMIT,
        "completed_dates": [],
        "days_completed": 0,
        "rows_written": 0,
    }

    print(
        f"DAILY BACKFILL: {len(trade_days)} trading days, "
        f"{target_start} -> {today}, completed={len(completed)}"
    )

    for trade_date in trade_days:
        if trade_date in completed:
            continue

        raw = request_bulk_day(api, trade_date)
        result = quality_check_daily(raw, trade_date)
        path = write_daily_file(raw, trade_date)

        state["completed_dates"].append(trade_date)
        state["completed_dates"] = sorted(
            set(state["completed_dates"]), reverse=True
        )
        state["days_completed"] = len(state["completed_dates"])
        state["rows_written"] += len(raw)
        state["last_day"] = result
        state["last_file"] = str(path.relative_to(ROOT))
        save_state(state)

        git_checkpoint(
            ["data/history", "data/universe.json"],
            f"data: checkpoint full-market {trade_date}",
        )
        print(
            f"CHECKPOINT {trade_date}: rows={len(raw)} "
            f"symbols={result['unique_symbols']} "
            f"days={state['days_completed']}/{len(trade_days)}"
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
                "symbols_strategy_universe": len(universe),
                "source": "zzshare daily all-fields bulk",
                "format": "daily CSV gzip",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    git_checkpoint(
        ["data/history/_BACKFILL_STATE.json", "data/history/_BACKFILL_COMPLETE"],
        "data: complete five-year full-market daily history",
    )


def quarter_end_for(year: int, quarter: int) -> date:
    return {
        1: date(year, 3, 31),
        2: date(year, 6, 30),
        3: date(year, 9, 30),
        4: date(year, 12, 31),
    }[quarter]


def iter_quarters(start: date, end: date):
    year = start.year
    q = (start.month - 1) // 3 + 1
    while (year, q) <= (end.year, (end.month - 1) // 3 + 1):
        yield year, q, quarter_end_for(year, q)
        q += 1
        if q == 5:
            q = 1
            year += 1


def iter_calendar_chunks(start: date, end: date, days: int = FINANCE_CHUNK_DAYS):
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def request_finance_range(
    api: DataApi,
    table: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    last_exc: Exception | None = None
    method = {
        "valuation": "finance_range",
        "indicator": "finance_range",
        "income": "finance_range",
        "balance": "finance_range",
        "cash_flow": "finance_range",
    }[table]

    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            df = getattr(api, method)(
                table=table,
                start_date=start_date,
                end_date=end_date,
                limit=FINANCE_LIMIT,
            )
            out = normalize_finance(df)
            if out.empty:
                raise RuntimeError(
                    f"finance_range returned empty table={table} "
                    f"range={start_date}:{end_date}"
                )
            return out
        except Exception as exc:
            last_exc = exc
            wait = min(120.0, (2 ** (attempt - 1)) + random.random())
            print(
                f"WARN finance_range {table} {start_date}:{end_date} "
                f"attempt {attempt}: {type(exc).__name__}: {exc}; "
                f"sleep={wait:.1f}s"
            )
            time.sleep(wait)
    raise RuntimeError(
        f"finance_range failed table={table} range={start_date}:{end_date}: {last_exc}"
    )


def request_finance_quarter(
    api: DataApi,
    table: str,
    quarter_value: str,
) -> pd.DataFrame:
    last_exc: Exception | None = None
    method_name = FINANCE_TABLES[table]

    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            df = getattr(api, method_name)(quarter_value)
            out = normalize_finance(df)
            if out.empty:
                raise RuntimeError(
                    f"{method_name} returned empty for {quarter_value}"
                )
            return out
        except Exception as exc:
            last_exc = exc
            wait = min(120.0, (2 ** (attempt - 1)) + random.random())
            print(
                f"WARN {method_name} {quarter_value} attempt {attempt}: "
                f"{type(exc).__name__}: {exc}; sleep={wait:.1f}s"
            )
            time.sleep(wait)
    raise RuntimeError(
        f"{method_name} failed for {quarter_value}: {last_exc}"
    )


def write_finance_file(
    df: pd.DataFrame,
    path: Path,
    key_cols: list[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in key_cols:
        if col not in out.columns:
            raise RuntimeError(f"missing finance key column {col} for {path}")
    out = out.drop_duplicates(key_cols)
    sort_cols = [c for c in ["trade_date", "report_date", "pub_date", "symbol"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    out.to_csv(path, index=False, compression="gzip")
    return path


def load_fund_state() -> dict:
    if FUND_STATE_FILE.exists():
        return json.loads(FUND_STATE_FILE.read_text(encoding="utf-8"))
    return {
        "status": "running",
        "started_at": datetime.now(TZ).isoformat(),
        "completed_valuation_chunks": [],
        "completed_quarters": [],
        "rows_written": {},
    }


def save_fund_state(state: dict) -> None:
    FUNDAMENTALS.mkdir(parents=True, exist_ok=True)
    FUND_STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def backfill_fundamentals() -> None:
    api = api_client()
    FUNDAMENTALS.mkdir(parents=True, exist_ok=True)

    today = datetime.now(TZ).date()
    try:
        target_start = today.replace(year=today.year - TARGET_YEARS)
    except ValueError:
        target_start = today.replace(
            year=today.year - TARGET_YEARS,
            day=28,
        )
    finance_start = target_start.replace(
        year=target_start.year - FINANCIAL_START_BUFFER_YEARS
    )

    state = load_fund_state()
    done_chunks = set(state["completed_valuation_chunks"])
    done_quarters = set(state["completed_quarters"])

    print(
        f"FUNDAMENTALS: valuation {target_start} -> {today}; "
        f"statements {finance_start} -> {today}"
    )

    # 1) Daily valuation history. Seven calendar days keeps the response below
    # the provider's practical range-limit and avoids silent 40k-row truncation.
    valuation_dir = FUNDAMENTALS / "valuation"
    for chunk_start, chunk_end in iter_calendar_chunks(
        target_start, today, FINANCE_CHUNK_DAYS
    ):
        key = f"{chunk_start}:{chunk_end}"
        if key in done_chunks:
            continue
        df = request_finance_range(
            api,
            "valuation",
            str(chunk_start),
            str(chunk_end),
        )

        # finance_range returns descending data; make the database canonical.
        df = df.rename(
            columns={
                "trade_date": "date",
                "market_cap": "market_cap",
                "circulating_market_cap": "circulating_market_cap",
            }
        )
        if "date" not in df.columns and "report_date" in df.columns:
            df = df.rename(columns={"report_date": "date"})
        if "date" in df.columns:
            df["date"] = pd.to_datetime(
                df["date"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")

        path = valuation_dir / f"{chunk_start}_{chunk_end}.csv.gz"
        write_finance_file(df, path, ["symbol", "date"])
        state["completed_valuation_chunks"].append(key)
        state["completed_valuation_chunks"] = sorted(
            set(state["completed_valuation_chunks"])
        )
        state["rows_written"]["valuation"] = (
            int(state["rows_written"].get("valuation", 0)) + len(df)
        )
        save_fund_state(state)

        git_checkpoint(
            ["data/fundamentals/valuation", "data/fundamentals/_FUNDAMENTALS_STATE.json"],
            f"data: valuation checkpoint {chunk_start} {chunk_end}",
        )
        print(
            f"VALUATION CHECKPOINT {chunk_start}:{chunk_end}: rows={len(df)}"
        )

    # 2) Quarterly statements/indicators. Keep raw vendor fields plus
    # report_date/pub_date so point-in-time joins can be rebuilt without
    # look-ahead bias.
    for year, quarter, report_end in iter_quarters(finance_start, today):
        q = f"{year}q{quarter}"
        for table in ["indicator", "income", "balance", "cash_flow"]:
            unit = f"{table}:{q}"
            if unit in done_quarters:
                continue

            df = request_finance_quarter(api, table, q)
            table_dir = FUNDAMENTALS / table
            path = table_dir / f"{q}.csv.gz"
            write_finance_file(df, path, ["symbol", "report_date", "pub_date"])

            state["completed_quarters"].append(unit)
            state["completed_quarters"] = sorted(
                set(state["completed_quarters"])
            )
            state["rows_written"][table] = (
                int(state["rows_written"].get(table, 0)) + len(df)
            )
            save_fund_state(state)

            git_checkpoint(
                [f"data/fundamentals/{table}", "data/fundamentals/_FUNDAMENTALS_STATE.json"],
                f"data: {table} checkpoint {q}",
            )
            print(
                f"FINANCE CHECKPOINT {table} {q}: rows={len(df)}"
            )

    state["status"] = "complete"
    state["completed_at"] = datetime.now(TZ).isoformat()
    state["target_start"] = str(target_start)
    state["financial_start"] = str(finance_start)
    save_fund_state(state)

    (FUNDAMENTALS / "_FUNDAMENTALS_COMPLETE").write_text(
        json.dumps(
            {
                "completed_at": state["completed_at"],
                "valuation_start": str(target_start),
                "valuation_end": str(today),
                "financial_start": str(finance_start),
                "financial_end": str(today),
                "tables": [
                    "valuation",
                    "indicator",
                    "income",
                    "balance",
                    "cash_flow",
                ],
                "source": "zzshare fundamentals",
                "point_in_time_fields": ["report_date", "pub_date"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    git_checkpoint(
        [
            "data/fundamentals/_FUNDAMENTALS_STATE.json",
            "data/fundamentals/_FUNDAMENTALS_COMPLETE",
        ],
        "data: complete five-year fundamentals database",
    )


def validate_fundamentals() -> None:
    api = api_client()
    checks = {}

    val = normalize_finance(api.finance_valuation("2026-09-30"))
    checks["valuation"] = {
        "rows": len(val),
        "columns": list(val.columns),
        "unique_symbols": int(val["symbol"].nunique()),
    }
    range_df = normalize_finance(
        api.finance_range(
            table="valuation",
            start_date="2026-09-21",
            end_date="2026-09-30",
            limit=40000,
        )
    )
    if "trade_date" not in range_df.columns:
        raise RuntimeError("valuation range missing trade_date")
    range_dates = pd.to_datetime(range_df["trade_date"], errors="coerce")
    checks["valuation_range_10d"] = {
        "rows": len(range_df),
        "unique_symbols": int(range_df["symbol"].nunique()),
        "min_date": range_dates.min().strftime("%Y-%m-%d") if range_dates.notna().any() else None,
        "max_date": range_dates.max().strftime("%Y-%m-%d") if range_dates.notna().any() else None,
    }

    for table, method in FINANCE_TABLES.items():
        df = normalize_finance(getattr(api, method)("2026q2"))
        checks[table] = {
            "rows": len(df),
            "columns": list(df.columns)[:30],
            "unique_symbols": int(df["symbol"].nunique()),
            "with_pub_date": int(df["pub_date"].notna().sum()),
        }

    payload = {
        "validated_at": datetime.now(TZ).isoformat(),
        "source": "zzshare fundamentals",
        "checks": checks,
    }
    FUNDAMENTALS.mkdir(parents=True, exist_ok=True)
    FUND_VALIDATION_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    git_checkpoint(
        ["data/fundamentals/_FUNDAMENTALS_VALIDATION.json"],
        "test: validate zzshare fundamentals source",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["validate", "backfill", "validate_fundamentals", "fundamentals", "full"],
        default="validate",
    )
    parser.add_argument("--days", type=int, default=VALIDATION_TRADING_DAYS)
    args = parser.parse_args()

    if args.mode == "validate":
        validate_bulk(args.days)
    elif args.mode == "backfill":
        backfill_daily()
    elif args.mode == "validate_fundamentals":
        validate_fundamentals()
    elif args.mode == "fundamentals":
        backfill_fundamentals()
    elif args.mode == "full":
        backfill_daily()
        backfill_fundamentals()


if __name__ == "__main__":
    main()
