import numpy as np

from scripts.train_executable_return_ranker import (
    ExecutableReturnMLPRegressor,
    simulate_executable_net_return,
)


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
