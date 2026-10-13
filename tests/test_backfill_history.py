from pathlib import Path

import pandas as pd

from scripts.backfill_history import (
    daily_file_has_basic_schema,
    history_file_date,
    merge_verified_trading_day,
)


def test_history_file_date_parses_gzip_suffix():
    path = Path("data/history/2026/2026-09-30.csv.gz")
    assert history_file_date(path).isoformat() == "2026-09-30"


def test_daily_file_has_basic_schema_does_not_require_valuation(tmp_path):
    path = tmp_path / "2026-09-30.csv.gz"
    pd.DataFrame(
        [
            {
                "symbol": "000001.SZ",
                "date": "2026-09-30",
                "open": 10.0,
                "high": 10.5,
                "low": 9.9,
                "close": 10.2,
                "pre_close": 10.0,
                "pct_chg": 2.0,
                "volume": 1000000,
                "amount": 10000000,
            }
        ]
    ).to_csv(path, index=False, compression="gzip")

    ok, rows = daily_file_has_basic_schema(path)
    assert ok is True
    assert rows == 1



def test_merge_verified_trading_day_keeps_current_date_missing_from_calendar():
    calendar_dates = ["2026-10-12", "2026-10-09", "2026-10-08"]

    merged = merge_verified_trading_day(calendar_dates, "2026-10-13")

    assert merged == [
        "2026-10-13",
        "2026-10-12",
        "2026-10-09",
        "2026-10-08",
    ]


def test_merge_verified_trading_day_deduplicates_verified_date():
    calendar_dates = ["2026-10-13", "2026-10-12"]

    merged = merge_verified_trading_day(calendar_dates, "2026-10-13")

    assert merged == ["2026-10-13", "2026-10-12"]
