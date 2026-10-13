import numpy as np

from scripts.train_executable_return_ranker import (
    ExecutableReturnMLPRegressor,
    build_executable_net_returns,
    simulate_executable_net_return,
)
import scripts.train_executable_return_ranker as return_ranker


def _bar(open_price, high, low, close):
    return {"open": open_price, "high": high, "low": low, "close": close}


def test_regressor_learns_direction_of_realized_net_return():
    features = np.asarray(
        [
            [0.95, 0.05],
            [0.85, 0.15],
            [0.05, 0.95],
            [0.15, 0.85],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([2.0, 1.5, -2.0, -1.5], dtype=np.float64)
    model = ExecutableReturnMLPRegressor(
        n_features=2,
        hidden_units=6,
        learning_rate=0.05,
        l2=0.0,
        seed=17,
    )
    before = float(np.mean((model.predict(features) - targets) ** 2))
    for _ in range(80):
        model.update(features, targets)
    predictions = model.predict(features)
    after = float(np.mean((predictions - targets) ** 2))

    assert after < before
    assert predictions[:2].mean() > predictions[2:].mean()


def test_t_plus_one_intraday_touch_cannot_exit_same_day():
    bars = [
        _bar(10.0, 10.5, 9.5, 10.1),
        _bar(10.0, 10.05, 9.9, 10.0),
        _bar(10.0, 10.05, 9.9, 10.0),
        _bar(10.0, 10.05, 9.9, 10.0),
        _bar(10.0, 10.05, 9.9, 10.0),
    ]
    result = simulate_executable_net_return(10.0, bars)
    assert result is not None
    assert result < 0.1


def test_same_day_stop_precedes_target_and_gap_fill_is_conservative():
    bars = [
        _bar(10.0, 10.5, 9.5, 10.1),
        _bar(9.0, 10.5, 8.9, 9.1),
        _bar(9.0, 9.2, 8.9, 9.0),
        _bar(9.0, 9.2, 8.9, 9.0),
        _bar(9.0, 9.2, 8.9, 9.0),
    ]
    result = simulate_executable_net_return(10.0, bars)
    assert result is not None
    # 跳空开盘价低于止损价，按开盘价成交并扣除往返成本。
    assert abs(result - (-10.1)) < 1e-6


def test_target_gap_uses_open_price_when_better_than_target():
    bars = [
        _bar(10.0, 10.5, 9.5, 10.1),
        _bar(10.2, 10.3, 10.1, 10.25),
        _bar(10.2, 10.3, 10.1, 10.25),
        _bar(10.2, 10.3, 10.1, 10.25),
        _bar(10.2, 10.3, 10.1, 10.25),
    ]
    result = simulate_executable_net_return(10.0, bars)
    assert result is not None
    assert abs(result - 1.9) < 1e-6


def test_training_labels_exclude_unexecutable_t_plus_one_entries(monkeypatch):
    symbols = ["000001", "000002"]
    monkeypatch.setattr(
        return_ranker,
        "build_targets",
        lambda _symbols, _future_days: (
            np.zeros(2, dtype=np.uint8),
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.ones(2, dtype=bool),
        ),
    )
    future_days = []
    for day_index in range(5):
        future_days.append(
            __import__("pandas").DataFrame(
                [
                    {
                        "symbol": "000001",
                        "open": 11.0 if day_index == 0 else 10.0,
                        "high": 11.2,
                        "low": 9.8,
                        "close": 10.5,
                        "high_limit": 11.0,
                        "is_paused": 0,
                    },
                    {
                        "symbol": "000002",
                        "open": 10.0,
                        "high": 10.2,
                        "low": 9.8,
                        "close": 10.0,
                        "high_limit": 11.0,
                        "is_paused": 0,
                    },
                ]
            )
        )

    returns, usable = build_executable_net_returns(symbols, future_days)

    assert not usable[0]
    assert np.isnan(returns[0])
    assert usable[1]
    assert np.isfinite(returns[1])
