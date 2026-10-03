import pandas as pd

from scripts.short_term_research import managed_trade


def _day(symbol, open_, high, low, close, high_limit=20.0):
    return pd.DataFrame([{
        "symbol": symbol,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "high_limit": high_limit,
    }])


def test_t1_position_cannot_exit_on_entry_session():
    future = [
        _day("600000", 10.0, 10.6, 9.4, 10.5),
        _day("600000", 10.5, 10.8, 10.4, 10.7),
        _day("600000", 10.7, 10.8, 10.6, 10.75),
    ]
    result = managed_trade("600000", future)
    assert result is not None
    assert result["holding_days"] >= 2


def test_entry_at_upper_limit_is_not_executable():
    future = [
        _day("600000", 20.0, 20.0, 19.5, 20.0, high_limit=20.0),
        _day("600000", 19.8, 20.2, 19.7, 20.0, high_limit=22.0),
        _day("600000", 20.0, 20.5, 19.9, 20.4, high_limit=22.0),
    ]
    assert managed_trade("600000", future) is None
