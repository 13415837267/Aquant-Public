from __future__ import annotations
import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.backfill_history import (api_client, filter_strategy_universe, git_checkpoint, load_trade_days, quality_check_daily, request_bulk_day, request_valuation_day, write_daily_file)

def load_universe_no_write(api):
    df = api.stock_basic(exchange="ALL", list_status="L", fields="ts_code,symbol,name,exchange,list_status")
    if df is None or df.empty:
        raise RuntimeError("zzshare stock_basic returned empty universe")
    df = df.copy()
    df["ts_code"] = df["ts_code"].astype(str).str.upper()
    if "name" in df.columns:
        name = df["name"].astype(str).str.upper()
        df = df[~name.str.contains(r"ST|\u9000", regex=True, na=False)].copy()
    df = df.drop_duplicates("ts_code").sort_values("ts_code")
    if len(df) < 4000:
        raise RuntimeError(f"universe unexpectedly small: {len(df)}")
    return df

def run_year(year: int) -> None:
    start = date(year, 1, 1).strftime("%Y%m%d")
    # Never request future trading dates for the current year.
    # The remote calendar may contain pre-published future open dates.
    today_bj = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    end_date = min(date(year, 12, 31), today_bj)
    end = end_date.strftime("%Y%m%d")
    api = api_client()
    universe = load_universe_no_write(api)
    trade_days = load_trade_days(api, start=start, end=end)
    if not trade_days:
        raise RuntimeError(f"no trading days for {year}")
    completed = 0
    skipped = 0
    days_since_remote_checkpoint = 0
    for trade_date in reversed(trade_days):
        path = ROOT / "data" / "history" / str(year) / f"{trade_date}.csv.gz"
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
        days_since_remote_checkpoint += 1
        print(f"CHECKPOINT {trade_date}: rows={len(combined)} symbols={result['unique_symbols']} year={year} new={completed}/{len(trade_days)}")
        if days_since_remote_checkpoint >= 5:
            git_checkpoint([f"data/history/{year}"], f"data: checkpoint historical year {year} through {trade_date}")
            days_since_remote_checkpoint = 0
            print(f"REMOTE CHECKPOINT {trade_date}: year={year} new={completed}/{len(trade_days)}")
    if days_since_remote_checkpoint:
        git_checkpoint([f"data/history/{year}"], f"data: checkpoint historical year {year} final batch")
    print(f"YEAR {year}: trading_days={len(trade_days)} new={completed} skipped={skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()
    if args.year < 2015 or args.year > 2026:
        raise SystemExit("year must be between 2015 and 2026")
    run_year(args.year)

