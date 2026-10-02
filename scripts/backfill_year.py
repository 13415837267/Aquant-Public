from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backfill_history import (
    api_client,
    filter_historical_main_board,
    git_checkpoint,
    load_trade_days,
    quality_check_daily,
    request_bulk_day,
    request_valuation_day,
    write_daily_file,
)


def run_year(year: int, rebuild: bool = False) -> None:
    start = date(year, 1, 1).strftime("%Y%m%d")
    today_bj = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    end_date = min(date(year, 12, 31), today_bj)
    end = end_date.strftime("%Y%m%d")
    api = api_client()
    trade_days = load_trade_days(api, start=start, end=end)
    if not trade_days:
        raise RuntimeError(f"no trading days for {year}")

    completed = 0
    skipped = 0
    days_since_remote_checkpoint = 0

    for trade_date in reversed(trade_days):
        path = ROOT / "data" / "history" / str(year) / f"{trade_date}.csv.gz"
        if path.exists() and not rebuild:
            skipped += 1
            continue

        raw = request_bulk_day(api, trade_date)
        market = filter_historical_main_board(raw)
        valuation = request_valuation_day(api, trade_date)
        combined = market.merge(
            valuation,
            on=["symbol", "date"],
            how="left",
            validate="one_to_one",
        )
        result = quality_check_daily(combined, trade_date)
        write_daily_file(combined, trade_date)
        completed += 1
        days_since_remote_checkpoint += 1

        print(
            f"CHECKPOINT {trade_date}: rows={len(combined)} "
            f"symbols={result['unique_symbols']} year={year} "
            f"new={completed}/{len(trade_days)} rebuild={rebuild}"
        )

        if days_since_remote_checkpoint >= 5:
            git_checkpoint(
                [f"data/history/{year}"],
                f"data: checkpoint historical main-board year {year} through {trade_date}",
            )
            days_since_remote_checkpoint = 0
            print(
                f"REMOTE CHECKPOINT {trade_date}: year={year} "
                f"new={completed}/{len(trade_days)} rebuild={rebuild}"
            )

    if days_since_remote_checkpoint:
        git_checkpoint(
            [f"data/history/{year}"],
            f"data: checkpoint historical main-board year {year} final batch",
        )

    print(
        f"YEAR {year}: trading_days={len(trade_days)} "
        f"new={completed} skipped={skipped} rebuild={rebuild}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="re-download and overwrite existing daily files",
    )
    args = parser.parse_args()
    if args.year < 2015 or args.year > 2026:
        raise SystemExit("year must be between 2015 and 2026")
    run_year(args.year, rebuild=args.rebuild)
