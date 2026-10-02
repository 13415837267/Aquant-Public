import gzip
import json

import numpy as np
import pandas as pd

from scripts.backtest import execution_limit_diagnostics, metrics, normalize_weights, rolling_252d_metrics, turnover
from scripts.backtest_constrained import FlattenedIntradayPortfolio
from scripts.walk_forward import build_folds
from scripts.sensitivity import run_one


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



def test_stateful_portfolio_blocks_limit_up_buy():
    portfolio = FlattenedIntradayPortfolio(initial_cash=1.0)
    targets = pd.DataFrame({"symbol": ["000001"]})
    execution = pd.DataFrame(
        {
            "symbol": ["000001"],
            "open": [10.0],
            "close": [11.0],
            "high_limit": [10.0],
            "low_limit": [9.0],
            "is_paused": [0],
        }
    )
    result = portfolio.rebalance(targets, execution, cost_rate=0.0005)
    assert result["blocked_buy_count"] == 1
    assert result["buy_count"] == 0
    assert result["position_count"] == 0
    assert np.isclose(result["equity_close"], 1.0)


def test_stateful_portfolio_keeps_limit_down_holding_and_buys_available_target():
    portfolio = FlattenedIntradayPortfolio(initial_cash=0.5)
    portfolio.shares["000001"] = 0.05
    portfolio.last_close["000001"] = 10.0
    portfolio.prev_close_equity = 1.0

    targets = pd.DataFrame({"symbol": ["000002"]})
    execution = pd.DataFrame(
        {
            "symbol": ["000001", "000002"],
            "open": [9.0, 10.0],
            "close": [9.0, 10.0],
            "high_limit": [9.9, 11.0],
            "low_limit": [9.0, 10.0],
            "is_paused": [0, 0],
        }
    )
    result = portfolio.rebalance(targets, execution, cost_rate=0.0005)
    assert result["blocked_sell_count"] == 1
    assert result["buy_count"] == 1
    assert "000001" in portfolio.shares
    assert "000002" in portfolio.shares
    assert result["position_count"] == 2



def test_stateful_portfolio_control_disables_price_limits_only():
    portfolio = FlattenedIntradayPortfolio(initial_cash=1.0)
    targets = pd.DataFrame({"symbol": ["000001"]})
    execution = pd.DataFrame(
        {
            "symbol": ["000001"],
            "open": [10.0],
            "close": [10.0],
            "high_limit": [10.0],
            "low_limit": [9.0],
            "is_paused": [0],
        }
    )
    result = portfolio.rebalance(
        targets,
        execution,
        cost_rate=0.0005,
        enforce_limits=False,
    )
    assert result["blocked_buy_count"] == 0
    assert result["buy_count"] == 1



def test_walk_forward_fold_snaps_calendar_date_to_next_trading_day():
    dates = [
        x.strftime("%Y-%m-%d")
        for x in pd.date_range("2015-01-05", "2020-12-31", freq="B")
        if x.strftime("%Y-%m-%d") != "2018-01-05"
    ]
    folds = build_folds(dates, train_years=3, test_years=1)
    assert folds
    assert folds[0]["oos_start"] == "2018-01-08"
    assert folds[0]["train_end"] == "2018-01-07"


def test_sensitivity_metrics_exclude_warmup_rows():
    payload = {
        "daily": [
            {"date": "2015-01-01", "gross_return": 0.0, "turnover": 0.0},
            {"date": "2015-01-02", "gross_return": 0.01, "turnover": 1.0},
            {"date": "2015-01-03", "gross_return": -0.005, "turnover": 0.5},
        ],
        "trade_start": "2015-01-02",
        "strategy_version": "test",
        "strategy_commit": "tiny-commit",
        "future_function": False,
    }
    result = run_one(30, 3.0, 2.0, payload)
    assert result["performance_sessions"] == 2
    assert np.isclose(result["total_return_pct"], 0.4200125)
