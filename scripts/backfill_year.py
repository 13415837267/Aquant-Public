from __future__ import annotations

import argparse
import subprocess
import time
from datetime import date

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backfill_history import (
    api_client,
    filter_strategy_universe,
    load_trade_days,
    quality_check_daily,
    request_bulk_day,
    request_valuation_day,
    write_daily_file,
)


def load_universe_no_write(api):
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
        df = df[~name.str.contains(r"ST|退", regex=True, na=False)].copy()
    df = df.drop_duplicates("ts_code").sort_values("ts_code")
    if len(df) < 4000:
        raise RuntimeError(f"universe unexpectedly small: {len(df)}")
    return df


def git_push_year(year: int, batch: int | None = None) -> None:
    subprocess.run(["git", "config", "user.name", "aquant-bot"], check=True)
    subprocess.run(["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], check=True)
    subprocess.run(["git", "add", "--", f"data/history/{year}"], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        print(f"YEAR {year}: no new files to commit")
        return
    message = (
        f"data: backfill historical year {year} batch {batch}"
        if batch is not None
        else f"data: backfill historical year {year}"
    )
    subprocess.run(["git", "commit", "-m", message], check=True)
    for attempt in range(1, 6):
        subprocess.run(["git", "fetch", "origin", "main"], check=True)
        try:
            subprocess.run(["git", "rebase", "origin/main"], check=True)
            subprocess.run(["git", "push", "origin", "HEAD:main"], check=True)
            print(f"YEAR {year}: pushed successfully")
            return
        except subprocess.CalledProcessError:
            subprocess.run(["git", "rebase", "--abort"], check=False)
            if attempt == 5:
                raise
            wait = attempt * 3
            print(f"YEAR {year}: push race, retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"YEAR {year}: failed to push after retries")


def run_year(year: int) -> None:
    start = date(year, 1, 1).strftime("%Y%m%d")
    end = date(year, 12, 31).strftime("%Y%m%d")
    api = api_client()
    universe = load_universe_no_write(api)
    trade_days = load_trade_days(api, start=start, end=end)
    if not trade_days:
        raise RuntimeError(f"no trading days for {year}")

    completed = 0
    skipped = 0
    batch_completed = 0
    batch = 0
    for trade_date in reversed(trade_days):
        path = __import__('pathlib').Path("data/history") / str(year) / f"{trade_date}.csv.gz"
        if path.exists():
            skipped += 1
            continue
        raw = request_bulk_day(api, trade_date)
        market = filter_strategy_universe(raw, universe)
        valuation = request_valuation_day(api, trade_date)
        combined = market.merge(valuation, on=["symbol", "date"], how="left", validate="one_to_one")
        result = quality_check_daily(combined, trade_date)
        write_daily_file(combined, trade_date)
        completed += 1
        batch_completed += 1
        print(f"YEAR {year}: {trade_date} rows={result['rows']} symbols={result['unique_symbols']} progress={completed}/{len(trade_days)}")

        # Persist every 5 newly downloaded trading days.
        if batch_completed >= 5:
            batch += 1
            git_push_year(year, batch=batch)
            batch_completed = 0

    if batch_completed:
        batch += 1
        git_push_year(year, batch=batch)

    print(f"YEAR {year}: trading_days={len(trade_days)} new={completed} skipped={skipped} commits={batch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    if args.year < 2015 or args.year > 2026:
        raise SystemExit("year must be between 2015 and 2026")
    run_year(args.year)
