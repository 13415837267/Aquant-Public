import pandas as pd

from scripts.market_scope import is_main_board_symbol
from scripts.update_candidates import build_candidates


class FakeStrategy:
    WEIGHTS = {"momentum": 1.0}

    @staticmethod
    def score_universe(frame):
        out = frame.copy()
        out["score"] = out["momentum_60d"]
        return out.sort_values("score", ascending=False)


def _history():
    dates = pd.date_range("2026-07-01", periods=61, freq="B")
    rows = []
    for symbol, start, step in [
        ("000001", 10.0, 0.15),
        ("000002", 10.0, 0.0),
        ("688001.SH", 10.0, 0.20),
        ("300001.SZ", 10.0, 0.25),
        ("920001.BJ", 10.0, 0.30),
    ]:
        for i, dt in enumerate(dates):
            close = start + step * i
            rows.append(
                {
                    "symbol": symbol,
                    "date": dt.strftime("%Y-%m-%d"),
                    "close": close,
                    "pre_close": close,
                    "volume": 1_000_000,
                    "amount": 50_000_000,
                    "pct_chg": 0.5,
                    "turnover_pct": 2.0,
                    "amplitude_pct": 1.0,
                    "is_paused": 0,
                    "is_st": 0,
                    "market_cap": 1e9,
                    "circulating_market_cap": 8e8,
                    "pe_ratio": 15.0,
                    "pb_ratio": 1.5,
                }
            )
    return pd.DataFrame(rows)


def test_main_board_scope():
    accepted = [
        "000001.SZ",
        "001200.SZ",
        "002594.SZ",
        "003816.SZ",
        "004001.SZ",
        "600000.SH",
        "601398.SH",
        "603019.SH",
        "605499.SH",
    ]
    rejected = [
        "001001.SZ",  # main-board CDR range
        "300001.SZ",  # ChiNext
        "688001.SH",  # STAR Market
        "920001.BJ",  # Beijing
        "900901.SH",  # Shanghai B-share
    ]
    assert all(is_main_board_symbol(code) for code in accepted)
    assert not any(is_main_board_symbol(code) for code in rejected)


def test_build_candidates_filters_to_main_board_and_metadata():
    snapshot = build_candidates(
        _history(),
        FakeStrategy,
        "test",
        "abc123",
    )

    assert len(snapshot["candidates"]) == 2
    assert {row["symbol"] for row in snapshot["candidates"]} == {"000001", "000002"}
    assert snapshot["strategy_source"] == "Aquant-Private/main"
    assert snapshot["strategy_version"] == "test"
    assert snapshot["strategy_commit"] == "abc123"
    assert snapshot["lookback_trading_days"] == 60
    assert snapshot["market_scope"].startswith("沪深主板")
    assert snapshot["diagnostics"]["history_rows"] == len(_history())
    assert snapshot["diagnostics"]["latest_main_board_rows"] == 2
    assert snapshot["diagnostics"]["scorable_rows"] == 2
    assert snapshot["diagnostics"]["candidate_count"] == 2

