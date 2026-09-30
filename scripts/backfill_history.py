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
STATE_FILE = HISTORY / "_BACKFILL_STATE.json"
TZ = ZoneInfo("Asia/Shanghai")

TARGET_YEARS = 5
CHUNK_DAYS = 20
WORKERS = 6
MIN_SUCCESS_RATIO = 0.80


def fetch_universe():
    for fn in (ak.stock_zh_a_spot_em, ak.stock_zh_a_spot):
        for attempt in range(3):
            try:
                df = fn()
                if df is None or df.empty:
                    raise RuntimeError("empty universe")
                df = df.copy()
                df["代码"] = (
                    df["代码"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
                )
                df = df[df["代码"].str.len().eq(6)]
                if "名称" in df.columns:
                    name = df["名称"].astype(str).str.upper()
                    before = len(df)
                    df = df[~name.str.contains(r"ST|退", regex=True, na=False)].copy()
                    print(
                        f"UNIVERSE FILTER: removed {before-len(df)} "
                        f"ST/delisted-related symbols; remaining={len(df)}"
                    )
                return df.drop_duplicates("代码").sort_values("代码")
            except Exception as exc:
                print(
                    f"WARN universe {getattr(fn, '__name__', fn)} "
                    f"attempt {attempt + 1}: {exc}"
                )
                time.sleep(2 ** attempt)
    raise RuntimeError("No free public A-share universe source is reachable")


def five_year_start(end):
    try:
        return end.replace(year=end.year - TARGET_YEARS)
    except ValueError:
        return end.replace(year=end.year - TARGET_YEARS, day=28)


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return None


def save_state(state):
    HISTORY.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def git_checkpoint(message):
    subprocess.run(
        ["git", "add", "data/history", "data/universe.json"], check=True
    )
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        return
    subprocess.run(["git", "config", "user.name", "aquant-bot"], check=True)
    subprocess.run(
        [
            "git",
            "config",
            "user.email",
            "41898282+github-actions[bot]@users.noreply.github.com",
        ],
        check=True,
    )
    subprocess.run(["git", "commit", "-m", message], check=True)
    subprocess.run(["git", "push"], check=True)


def fetch_stock_window(symbol, start, end):
    for attempt in range(3):
        try:
            df = ak.stock_zh_a_hist(
                symbol=str(symbol).zfill(6),
                period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="",
            )
            if df is None or df.empty:
                return None

            aliases = {
                "date": ["日期"],
                "open": ["开盘"],
                "close": ["收盘"],
                "high": ["最高"],
                "low": ["最低"],
                "volume": ["成交量"],
                "amount": ["成交额"],
                "amplitude": ["振幅"],
                "pct_chg": ["涨跌幅"],
                "change": ["涨跌额"],
                "turnover_pct": ["换手率"],
            }

            def col(key):
                for name in aliases[key]:
                    if name in df.columns:
                        return df[name]
                return pd.Series(index=df.index, dtype="float64")

            out = pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        col("date"), errors="coerce"
                    ).dt.strftime("%Y-%m-%d"),
                    "symbol": str(symbol).zfill(6),
                    "open": pd.to_numeric(col("open"), errors="coerce"),
                    "high": pd.to_numeric(col("high"), errors="coerce"),
                    "low": pd.to_numeric(col("low"), errors="coerce"),
                    "close": pd.to_numeric(col("close"), errors="coerce"),
                    "volume": pd.to_numeric(col("volume"), errors="coerce"),
                    "amount": pd.to_numeric(col("amount"), errors="coerce"),
                    "amplitude_pct": pd.to_numeric(
                        col("amplitude"), errors="coerce"
                    ),
                    "pct_chg": pd.to_numeric(col("pct_chg"), errors="coerce"),
                    "change": pd.to_numeric(col("change"), errors="coerce"),
                    "turnover_pct": pd.to_numeric(
                        col("turnover_pct"), errors="coerce"
                    ),
                }
            ).dropna(subset=["date", "close"])

            return out if not out.empty else None
        except Exception as exc:
            if attempt == 2:
                print(f"WARN {symbol} {start} -> {end}: {exc}")
            time.sleep(1.5 * (attempt + 1))
    return None


def write_daily_files(df):
    written = []
    for day, group in df.groupby("date", sort=True):
        path = HISTORY / f"{day}.csv.gz"
        group.sort_values("symbol").to_csv(
            path, index=False, compression="gzip"
        )
        written.append(day)
    return written


def write_universe(raw):
    cols = [c for c in ["代码", "名称", "最新价", "成交额"] if c in raw.columns]
    raw[cols].sort_values("代码").to_json(
        DATA / "universe.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )


def initial_backfill(raw):
    HISTORY.mkdir(parents=True, exist_ok=True)

    end = datetime.now(TZ).date()
    target_start = five_year_start(end)
    symbols = (
        raw["代码"].dropna().astype(str).str.zfill(6).drop_duplicates().tolist()
    )

    state = load_state()
    if state and state.get("status") == "running":
        cursor_end = datetime.strptime(
            state["cursor_end"], "%Y-%m-%d"
        ).date()
        print(f"RESUME: next date range ends {cursor_end}")
    else:
        cursor_end = end
        state = {
            "status": "running",
            "started_at": datetime.now(TZ).isoformat(),
            "target_start": str(target_start),
            "initial_end": str(end),
            "symbols_requested": len(symbols),
            "direction": "near_to_far",
            "chunk_days": CHUNK_DAYS,
            "cursor_end": str(cursor_end),
            "chunks_completed": 0,
            "days_completed": 0,
            "rows_written": 0,
            "failed_symbols": {},
        }
        save_state(state)
        write_universe(raw)
        git_checkpoint("data: initialize daily five-year market collection")

    while cursor_end >= target_start:
        cursor_start = max(
            target_start, cursor_end - timedelta(days=CHUNK_DAYS - 1)
        )
        print(f"COLLECT: {cursor_start} -> {cursor_end}")

        frames = []
        failed = []

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {
                pool.submit(
                    fetch_stock_window, symbol, cursor_start, cursor_end
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

                if i % 250 == 0:
                    print(
                        f"PROGRESS {i}/{len(symbols)} "
                        f"success={len(frames)} failed={len(failed)}"
                    )

        success_ratio = len(frames) / max(len(symbols), 1)
        if success_ratio < MIN_SUCCESS_RATIO:
            raise RuntimeError(
                f"Chunk coverage too low: {success_ratio:.2%}; "
                f"refusing to checkpoint incomplete market data"
            )

        chunk_df = pd.concat(frames, ignore_index=True)
        written_days = write_daily_files(chunk_df)
        write_universe(raw)

        state["chunks_completed"] += 1
        state["days_completed"] += len(written_days)
        state["rows_written"] += len(chunk_df)
        state["cursor_end"] = str(cursor_start - timedelta(days=1))
        state["last_chunk"] = {
            "start": str(cursor_start),
            "end": str(cursor_end),
            "days_written": written_days,
            "symbols_with_data": len(frames),
            "symbols_failed": len(failed),
            "coverage": success_ratio,
            "rows": len(chunk_df),
        }
        state["failed_symbols"] = {
            "range": f"{cursor_start}/{cursor_end}",
            "symbols": failed[:1000],
        }
        save_state(state)

        # Each completed date range is durable before moving farther back.
        git_checkpoint(
            f"data: checkpoint market history {cursor_start} to {cursor_end}"
        )
        cursor_end = cursor_start - timedelta(days=1)

    state["status"] = "complete"
    state["completed_at"] = datetime.now(TZ).isoformat()
    save_state(state)

    (HISTORY / "_BACKFILL_COMPLETE").write_text(
        json.dumps(
            {
                "completed_at": state["completed_at"],
                "target_start": str(target_start),
                "end": str(end),
                "years": TARGET_YEARS,
                "direction": "near_to_far",
                "symbols_requested": len(symbols),
                "format": "daily CSV gzip",
                "source": "AKShare / Eastmoney stock_zh_a_hist",
                "free_public_source": True,
                "universe_filter": "exclude ST/*ST and delisted-related names",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    git_checkpoint("data: complete five-year daily market history")


def incremental(raw, today):
    # Daily maintenance uses the same free public historical interface.
    path = HISTORY / f"{today}.csv.gz"
    symbols = (
        raw["代码"].dropna().astype(str).str.zfill(6).drop_duplicates().tolist()
    )
    frames = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(fetch_stock_window, symbol, today, today): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            df = future.result()
            if df is not None and not df.empty:
                frames.append(df)

    if frames:
        day_df = pd.concat(frames, ignore_index=True)
        coverage = len(day_df["symbol"].unique()) / max(len(symbols), 1)
        if coverage < MIN_SUCCESS_RATIO:
            raise RuntimeError(f"Daily coverage too low: {coverage:.2%}")
        day_df.sort_values("symbol").to_csv(
            path, index=False, compression="gzip"
        )
        write_universe(raw)
        print(
            json.dumps(
                {
                    "mode": "incremental",
                    "date": str(today),
                    "symbols": len(day_df["symbol"].unique()),
                    "rows": len(day_df),
                    "coverage": coverage,
                },
                ensure_ascii=False,
            )
        )


def main():
    raw = fetch_universe()
    marker = HISTORY / "_BACKFILL_COMPLETE"

    if not marker.exists():
        initial_backfill(raw)
        return

    today = datetime.now(TZ).date()
    try:
        dates = set(
            pd.to_datetime(
                ak.tool_trade_date_hist_sina()["trade_date"]
            ).dt.date
        )
        if today not in dates:
            print(
                json.dumps(
                    {
                        "mode": "incremental",
                        "skipped": True,
                        "date": str(today),
                    },
                    ensure_ascii=False,
                )
            )
            return
    except Exception as exc:
        print(f"WARN trade calendar: {exc}")

    incremental(raw, today)


if __name__ == "__main__":
    main()
