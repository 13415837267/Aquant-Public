import gzip
import json

import numpy as np
import pandas as pd

from scripts.backtest import execution_limit_diagnostics, metrics, normalize_weights, rolling_252d_metrics, turnover


def test_metrics_simple_path():
    daily = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05", "2026-01-06"],
            "net_return": [0.01, -0.005, 0.02],
            "turnover": [1.0, 0.2, 0.1],
        }
    )
    result = metrics(daily)

    expected = (1.01 * 0.995 * 1.02) - 1
    assert np.isclose(result["total_return_pct"], expected * 100)
    assert result["trading_days"] == 3
    assert result["average_turnover_pct"] > 0
    assert 0 <= result["win_rate_pct"] <= 100
    assert result["best_day_pct"] >= result["worst_day_pct"]
    assert result["max_consecutive_losses"] >= 0
    assert result["max_consecutive_gains"] >= 0




def test_execution_limit_diagnostics_counts_limit_opens():
    selected = pd.DataFrame({"symbol": ["000001", "000002", "000003"]})
    execution = pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000003"],
            "open": [10.0, 9.0, 8.0],
            "high_limit": [10.0, 10.0, 8.0],
            "low_limit": [9.0, 9.0, 7.0],
        }
    )
    assert execution_limit_diagnostics(selected, execution) == (2, 1)

def test_weights_are_equal_and_normalized():
    weights = normalize_weights(["000002", "000001", "000001"])
    assert weights == {"000001": 0.5, "000002": 0.5}
    assert np.isclose(sum(weights.values()), 1.0)


def test_turnover_is_l1_target_weight_change():
    prev = {"000001": 0.5, "000002": 0.5}
    target = {"000002": 0.5, "000003": 0.5}
    assert np.isclose(turnover(prev, target), 1.0)


def test_run_backtest_executes_full_loop_on_tiny_history(tmp_path, monkeypatch):
    import scripts.backtest as backtest

    dates = pd.date_range("2026-01-05", periods=62, freq="B")
    files = []
    for idx, dt in enumerate(dates):
        rows = []
        for symbol, base in [("000001.SZ", 10.0), ("000002.SZ", 12.0)]:
            close = base + idx * (0.10 if symbol == "000001.SZ" else 0.02)
            rows.append(
                {
                    "symbol": symbol,
                    "date": dt.strftime("%Y-%m-%d"),
                    "open": close,
                    "close": close * 1.01,
                    "volume": 1_000_000,
                    "amount": 50_000_000,
                    "pct_chg": 0.5,
                    "turnover_pct": 2.0,
                    "is_paused": 0,
                    "is_st": 0,
                    "pe_ratio": 15.0,
                    "pb_ratio": 1.5,
                    "high_limit": close * 1.10,
                    "low_limit": close * 0.90,
                }
            )
        frame = pd.DataFrame(rows)
        path = tmp_path / f"{dt.strftime('%Y-%m-%d')}.csv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            frame.to_csv(fh, index=False)
        files.append(path)

    class FakeStrategy:
        @staticmethod
        def score_universe(frame):
            out = frame.copy()
            out["score"] = out["momentum_60d"]
            return out.sort_values("score", ascending=False)

    monkeypatch.setattr(backtest, "history_files", lambda: files)
    monkeypatch.setattr(
        backtest,
        "load_strategy",
        lambda: (FakeStrategy, "test", "tiny-commit"),
    )
    output_path = tmp_path / "latest.json"
    monkeypatch.setattr(backtest, "OUT_FILE", output_path)

    payload = backtest.run_backtest(
        start=dates[0].strftime("%Y-%m-%d"),
        end=dates[-1].strftime("%Y-%m-%d"),
        top_n=1,
        cost_bps=3.0,
        slippage_bps=2.0,
    )

    assert payload["status"] == "ready"
    assert payload["strategy_version"] == "test"
    assert payload["overall"]["trading_days"] >= 1
    assert payload["audit"]["performance_sessions"] == payload["overall"]["trading_days"]
    assert output_path.exists()


def test_rolling_252d_metrics_samples_final_window():
    dates = pd.date_range("2020-01-02", periods=300, freq="B")
    daily = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "net_return": np.full(300, 0.001),
            "turnover": np.zeros(300),
        }
    )
    rows = rolling_252d_metrics(daily, step=63)
    assert rows
    assert rows[-1]["end_date"] == dates[-1].strftime("%Y-%m-%d")
    assert rows[-1]["window_sessions"] == 252
