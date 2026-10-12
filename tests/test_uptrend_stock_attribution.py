import numpy as np

from scripts.uptrend_stock_attribution import (
    FeatureMoments,
    simulate_target_path,
    standardized_effect,
)


def bar(date, open_, high, low, close, paused=0):
    return {
        "date": date,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "is_paused": paused,
    }


def test_t_plus_one_price_move_does_not_trigger_exit():
    bars = [
        bar("2026-01-02", 100, 120, 80, 105),
        bar("2026-01-05", 101, 102, 99, 101),
        bar("2026-01-06", 101, 102, 99, 101),
        bar("2026-01-07", 101, 102, 99, 101),
        bar("2026-01-08", 101, 102, 99, 101),
    ]
    result = simulate_target_path(100, bars, 3, 3, 10, earliest_exit_day=2)
    assert result["resolved"] is True
    assert result["reason"] == "到期"
    assert result["exit_date"] == "2026-01-08"


def test_same_day_stop_is_prioritized_over_target():
    bars = [
        bar("2026-01-02", 100, 105, 98, 101),
        bar("2026-01-05", 100, 106, 96, 103),
        bar("2026-01-06", 101, 102, 99, 101),
        bar("2026-01-07", 101, 102, 99, 101),
        bar("2026-01-08", 101, 102, 99, 101),
    ]
    result = simulate_target_path(100, bars, 3, 3, 10, earliest_exit_day=2)
    assert result["resolved"] is True
    assert result["reason"] == "止损"
    assert result["net_return_pct"] < 0


def test_gap_up_target_uses_open_fill_and_transaction_costs():
    bars = [
        bar("2026-01-02", 100, 104, 99, 103),
        bar("2026-01-05", 104, 108, 103.5, 107),
        bar("2026-01-06", 107, 108, 106, 107),
        bar("2026-01-07", 107, 108, 106, 107),
        bar("2026-01-08", 107, 108, 106, 107),
    ]
    result = simulate_target_path(100, bars, 3, 3, 10, earliest_exit_day=2)
    assert result["resolved"] is True
    assert result["reason"] == "止盈"
    assert result["exit_price"] == 104
    assert result["net_return_pct"] == 3.9


def test_suspended_future_path_is_not_given_a_fake_exit():
    bars = [
        bar("2026-01-02", 100, 101, 99, 100),
        bar("2026-01-05", 100, 101, 99, 100, paused=1),
        bar("2026-01-06", 100, 101, 99, 100, paused=1),
        bar("2026-01-07", 100, 101, 99, 100, paused=1),
        bar("2026-01-08", 100, 101, 99, 100, paused=1),
    ]
    result = simulate_target_path(100, bars, 3, 3, 10, earliest_exit_day=2)
    assert result["resolved"] is False
    assert result["reason"] == "期末仍持仓，按最后有效收盘估值"


def test_standardized_factor_difference():
    winners = FeatureMoments(2)
    controls = FeatureMoments(2)
    winners.update(np.array([[3.0, 10.0], [4.0, 10.0]]))
    controls.update(np.array([[1.0, 10.0], [2.0, 10.0]]))
    effect = standardized_effect(winners.summary(), controls.summary(), 0)
    assert effect["上涨样本均值"] == 3.5
    assert effect["对照样本均值"] == 1.5
    assert effect["标准化均值差"] > 0
