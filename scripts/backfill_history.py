from __future__ import annotations
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HISTORY = DATA / "history"
TZ = ZoneInfo("Asia/Shanghai")
TARGET_YEARS = 5
CHUNK_DAYS = 90
WORKERS = 4


def fetch_spot():
    for fn in (ak.stock_zh_a_spot, ak.stock_zh_a_spot_em):
        for attempt in range(3):
            try:
                df = fn()
                if df is not None and not df.empty:
                    df = df.copy()
                    df["代码"] = df["代码"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
                    df = df[df["代码"].str.len().eq(6)]
                    if "名称" in df.columns:
                        name = df["名称"].astype(str).str.upper()
                        before = len(df)
                        df = df[~name.str.contains(r"ST|退", regex=True, na=False)].copy()
                        print(f"UNIVERSE FILTER: removed {before-len(df)} ST/delisted-related symbols; remaining={len(df)}")
                    return df.drop_duplicates("代码")
            except Exception as exc:
                print(f"WARN universe {getattr(fn, '__name__', fn)} attempt {attempt+1}: {exc}")
                time.sleep(2 ** attempt)
    raise RuntimeError("No A-share universe source is reachable")


def market_symbol(code):
    code = str(code).zfill(6)
    return ("sh" if code.startswith("6") else "sz") + code


def normalize_history(df, symbol):
    if df is None or df.empty:
        return None
    aliases = {
        "date": ["日期", "date"], "open": ["开盘", "open"], "high": ["最高", "high"],
        "low": ["最低", "low"], "close": ["收盘", "close"], "volume": ["成交量", "volume"],
        "amount": ["成交额", "amount"], "turnover_pct": ["换手率", "turnover"],
    }
    def col(key):
        for name in aliases[key]:
            if name in df.columns:
                return df[name]
        return pd.Series(index=df.index, dtype="float64")
    dates = col("date")
    if dates.isna().all():
        return None
    return pd.DataFrame({
        "date": pd.to_datetime(dates, errors="coerce").dt.strftime("%Y-%m-%d"),
        "symbol": str(symbol).zfill(6),
        "open": pd.to_numeric(col("open"), errors="coerce"),
        "high": pd.to_numeric(col("high"), errors="coerce"),
        "low": pd.to_numeric(col("low"), errors="coerce"),
        "close": pd.to_numeric(col("close"), errors="coerce"),
        "volume": pd.to_numeric(col("volume"), errors="coerce"),
        "amount": pd.to_numeric(col("amount"), errors="coerce"),
        "turnover_pct": pd.to_numeric(col("turnover_pct"), errors="coerce"),
    }).dropna(subset=["date", "close"])


def history_one(symbol, start, end):
    for attempt in range(3):
        try:
            return normalize_history(
                ak.stock_zh_a_daily(
                    symbol=market_symbol(symbol),
                    start_date=start,
                    end_date=end,
                    adjust="qfq",
                ),
                symbol,
            )
        except Exception as exc:
            if attempt == 2:
                print(f"WARN {symbol} {start}-{end}: {exc}")
            time.sleep(1.5 * (attempt + 1))
    return None


def write_daily_gzip(all_df):
    HISTORY.mkdir(parents=True, exist_ok=True)
    for day, group in all_df.groupby("date", sort=True):
        group.sort_values("symbol").to_csv(
            HISTORY / f"{day}.csv.gz", index=False, compression="gzip"
        )
    return int(all_df["date"].nunique())


def write_universe(raw):
    cols = [c for c in ["代码", "名称", "最新价", "成交额"] if c in raw.columns]
    raw[cols].sort_values("代码").to_json(
        DATA / "universe.json", orient="records", force_ascii=False, indent=2
    )


def load_state():
    path = HISTORY / "_BACKFILL_STATE.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_state(state):
    (HISTORY / "_BACKFILL_STATE.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def git_checkpoint(message):
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


def five_year_start(end):
    try:
        return end.replace(year=end.year - TARGET_YEARS)
    except ValueError:
        return end.replace(year=end.year - TARGET_YEARS, day=28)


def initial_backfill(raw):
    HISTORY.mkdir(parents=True, exist_ok=True)
    end = datetime.now(TZ).date()
    target_start = five_year_start(end)
    symbols = raw["代码"].dropna().astype(str).str.zfill(6).drop_duplicates().tolist()
    state = load_state()

    if state:
        cursor_end = datetime.strptime(state["cursor_end"], "%Y-%m-%d").date()
        print(f"RESUME: next range ending {cursor_end}")
    else:
        cursor_end = end
        state = {
            "status": "running",
            "started_at": datetime.now(TZ).isoformat(),
            "target_start": str(target_start),
            "initial_end": str(end),
            "symbols_requested": len(symbols),
            "chunks_completed": 0,
            "rows_written": 0,
            "trading_days_written": 0,
            "failed_requests": {},
            "cursor_end": str(cursor_end),
            "direction": "near_to_far",
            "chunk_days": CHUNK_DAYS,
        }
        save_state(state)
        git_checkpoint("data: initialize five-year history backfill")

    while cursor_end >= target_start:
        cursor_start = max(target_start, cursor_end - timedelta(days=CHUNK_DAYS - 1))
        print(f"CHUNK: {cursor_start} -> {cursor_end} (near to far)")
        frames, failed = [], []

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {
                pool.submit(
                    history_one, symbol,
                    cursor_start.strftime("%Y%m%d"),
                    cursor_end.strftime("%Y%m%d"),
                ): symbol
                for symbol in symbols
            }
            for i, future in enumerate(as_completed(futures), 1):
                symbol = futures[future]
                try:
                    df = future.result()
                except Exception as exc:
                    print(f"WARN future {symbol}: {exc}")
                    df = None
                if df is not None and not df.empty:
                    frames.append(df)
                else:
                    failed.append(symbol)
                if i % 200 == 0:
                    print(f"CHUNK PROGRESS {i}/{len(symbols)} success={len(frames)} failed={len(failed)}")

        if not frames:
            raise RuntimeError(f"No history returned for chunk {cursor_start} -> {cursor_end}")

        chunk_df = pd.concat(frames, ignore_index=True)
        days = write_daily_gzip(chunk_df)
        write_universe(raw)

        state["chunks_completed"] += 1
        state["rows_written"] += len(chunk_df)
        state["trading_days_written"] += days
        state["cursor_end"] = str(cursor_start - timedelta(days=1))
        state["last_chunk"] = {
            "start": str(cursor_start), "end": str(cursor_end),
            "symbols_with_data": len(frames), "symbols_without_data": len(failed),
            "rows": len(chunk_df), "trading_days": days,
        }
        state["failed_requests"] = {str(cursor_start): failed[:500]} if failed else {}
        save_state(state)

        git_checkpoint(f"data: checkpoint five-year history {cursor_start} to {cursor_end}")
        cursor_end = cursor_start - timedelta(days=1)

    state["status"] = "complete"
    state["completed_at"] = datetime.now(TZ).isoformat()
    save_state(state)
    (HISTORY / "_BACKFILL_COMPLETE").write_text(json.dumps({
        "completed_at": state["completed_at"],
        "symbols_requested": len(symbols),
        "target_start": str(target_start),
        "end": str(end),
        "years": TARGET_YEARS,
        "direction": "near_to_far",
        "format": "daily CSV gzip",
        "source": "AKShare / Sina stock_zh_a_daily",
        "universe_filter": "exclude ST/*ST and delisted-related names",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    git_checkpoint("data: complete five-year history backfill")


def incremental(raw, today):
    path = HISTORY / f"{today}.csv.gz"
    symbols = raw["代码"].dropna().astype(str).str.zfill(6).drop_duplicates().tolist()
    frames = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(history_one, symbol, today.strftime("%Y%m%d"), today.strftime("%Y%m%d")): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            df = future.result()
            if df is not None and not df.empty:
                frames.append(df)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(path, index=False, compression="gzip")
        write_universe(raw)


def main():
    raw = fetch_spot()
    marker = HISTORY / "_BACKFILL_COMPLETE"
    if not marker.exists():
        initial_backfill(raw)
        return
    today = datetime.now(TZ).date()
    try:
        dates = set(pd.to_datetime(ak.tool_trade_date_hist_sina()["trade_date"]).dt.date)
        if today not in dates:
            print(json.dumps({"mode": "incremental", "skipped": True, "date": str(today)}, ensure_ascii=False))
            return
    except Exception as exc:
        print(f"WARN trade calendar: {exc}")
    incremental(raw, today)


if __name__ == "__main__":
    main()
