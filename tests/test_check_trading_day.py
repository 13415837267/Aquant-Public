from datetime import date

from scripts.check_trading_day import calendar_contains_date


def test_calendar_uses_iso_dates_returned_by_data_source():
    trading_days = {"2026-10-13", "2026-10-14"}
    assert calendar_contains_date(date(2026, 10, 14), trading_days)


def test_calendar_keeps_dates_missing_from_open_days_closed():
    trading_days = {"2026-10-09", "2026-10-12"}
    assert not calendar_contains_date(date(2026, 10, 10), trading_days)
