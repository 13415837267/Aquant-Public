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


def test_t1_position_can_exit_when_net_profit_reaches_1pct():
    future = [
        _day("600000", 10.0, 10.6, 9.8, 10.5),
        _day("600000", 10.5, 10.8, 10.4, 10.7),
        _day("600000", 10.7, 10.8, 10.6, 10.75),
    ]
    result = managed_trade("600000", future)
    assert result is not None
    assert result["holding_days"] == 1
    assert result["exit_reason"] == "target"


def test_entry_at_upper_limit_is_not_executable():
    future = [
        _day("600000", 20.0, 20.0, 19.5, 20.0, high_limit=20.0),
        _day("600000", 19.8, 20.2, 19.7, 20.0, high_limit=22.0),
        _day("600000", 20.0, 20.5, 19.9, 20.4, high_limit=22.0),
    ]
    assert managed_trade("600000", future) is None


def test_same_day_stop_takes_priority_over_target():
    future = [
        _day("600000", 10.0, 10.2, 9.6, 10.1),
        _day("600000", 10.1, 10.3, 10.0, 10.2),
        _day("600000", 10.2, 10.3, 10.1, 10.25),
    ]
    result = managed_trade("600000", future)
    assert result is not None
    assert result["exit_reason"] == "stop"
    assert result["holding_days"] == 1
