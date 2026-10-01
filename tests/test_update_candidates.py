import pandas as pd

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
                    "name": "测试A" if symbol == "000001" else "测试B",
                }
            )
    return pd.DataFrame(rows)


def test_build_candidates_uses_database_history_and_strategy_output():
    snapshot = build_candidates(
        _history(),
        FakeStrategy,
        "test",
        "abc123",
    )

    assert len(snapshot["candidates"]) == 2
    assert snapshot["candidates"][0]["symbol"] == "000001"
    assert snapshot["strategy_source"] == "Aquant-Private/main"
    assert snapshot["strategy_version"] == "test"
    assert snapshot["strategy_commit"] == "abc123"
    assert snapshot["lookback_trading_days"] == 60
