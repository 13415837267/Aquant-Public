import math

import pandas as pd

from research.external_strategy_tests.runner import (
    CONFIG,
    MAX_HOLDING_SESSIONS,
    STOP_PCT,
    WIN_THRESHOLD_PCT,
    STRATEGIES,
    Portfolio,
    exit_decision,
    score_candidates,
)


def test_public_picker_approx_filters_and_ranks_by_disclosed_conditions():
    frame = pd.DataFrame(
        [
            {
                "symbol": "600002",
                "ma5_vs_ma10_pct": 1.0,
                "ma10_vs_ma20_pct": 1.0,
                "limit_up_10d_count": 1,
                "turnover_pct": 10.0,
                "volume_ratio_5d": 1.2,
                "amount_20d": 60_000_000,
                "change_pct": 1.0,
            },
            {
                "symbol": "600001",
                "ma5_vs_ma10_pct": 2.0,
                "ma10_vs_ma20_pct": 2.0,
                "limit_up_10d_count": 2,
                "turnover_pct": 12.0,
                "volume_ratio_5d": 1.5,
                "amount_20d": 90_000_000,
                "change_pct": 2.0,
            },
            {
                "symbol": "600003",
                "ma5_vs_ma10_pct": -1.0,
                "ma10_vs_ma20_pct": 1.0,
                "limit_up_10d_count": 2,
                "turnover_pct": 12.0,
                "volume_ratio_5d": 1.5,
                "amount_20d": 90_000_000,
                "change_pct": 4.0,
            },
        ]
    )

    result = score_candidates("public_picker_approx", frame, CONFIG)

    assert result["symbol"].tolist() == ["600001", "600002"]
    assert result["score"].is_monotonic_decreasing


def test_pullback_proxy_requires_uptrend_pullback_and_volume_contraction():
    frame = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "ma20_vs_ma60_pct": 3.0,
                "close_vs_ma20_pct": 1.0,
                "close_vs_ma5_pct": -1.0,
                "distance_from_high_20d_pct": -6.0,
                "volume_ratio_5d": 0.7,
                "turnover_pct": 8.0,
            },
            {
                "symbol": "600002",
                "ma20_vs_ma60_pct": 3.0,
                "close_vs_ma20_pct": 1.0,
                "close_vs_ma5_pct": -1.0,
                "distance_from_high_20d_pct": -6.0,
                "volume_ratio_5d": 1.2,
                "turnover_pct": 8.0,
            },
        ]
    )

    result = score_candidates("khunter_pullback_proxy", frame, CONFIG)

    assert result["symbol"].tolist() == ["600001"]


def test_momentum_baseline_uses_only_past_and_current_close_features():
    frame = pd.DataFrame(
        [
            {
                "symbol": "600001",
                "return_20d_pct": 8.0,
                "close_vs_ma20_pct": 2.0,
                "ma20_vs_ma60_pct": 3.0,
                "volatility_10d_pct": 2.0,
            },
            {
                "symbol": "600002",
                "return_20d_pct": -2.0,
                "close_vs_ma20_pct": 2.0,
                "ma20_vs_ma60_pct": 3.0,
                "volatility_10d_pct": 2.0,
            },
        ]
    )

    result = score_candidates("momentum20_baseline", frame, CONFIG)

    assert result["symbol"].tolist() == ["600001"]


def test_entry_day_cannot_exit_and_same_day_stop_wins_over_target():
    position = {"stop_price": 97.0, "target_price": 101.0}
    bar = {"low": 90.0, "high": 105.0, "close": 100.0, "is_paused": 0}

    assert exit_decision(position, bar, days_since_entry=0) is None
    decision = exit_decision(position, bar, days_since_entry=1)
    assert decision is not None
    assert decision["exit_reason"] == "止损"
    assert decision["exit_price"] == position["stop_price"]


def test_paused_position_is_not_falsely_marked_as_executable_exit():
    position = {"stop_price": 97.0, "target_price": 101.0}
    bar = {"low": 90.0, "high": 105.0, "close": 100.0, "is_paused": 1}

    assert exit_decision(position, bar, days_since_entry=1) is None


def test_maximum_holding_window_exits_at_last_close_if_no_barrier_was_hit():
    position = {"stop_price": 97.0, "target_price": 101.0}
    bar = {"low": 99.0, "high": 100.5, "close": 100.2, "is_paused": 0}

    decision = exit_decision(position, bar, days_since_entry=MAX_HOLDING_SESSIONS - 1)

    assert decision is not None
    assert decision["exit_reason"] == "持有期满"
    assert decision["exit_price"] == 100.2


def test_metrics_report_total_return_and_drawdown_without_optimistic_annualization():
    period = {"id": "validation", "label": "验证期", "start": "2023-01-03", "end": "2024-12-31"}
    portfolio = Portfolio(
        strategy=STRATEGIES[0],
        period=period,
        start_index=0,
        end_index=3,
        config=CONFIG,
        one_way_cost_rate=0.0005,
        initial_equity=1.0,
    )
    portfolio.equity_curve = [
        {"日期": "2023-01-03", "净值": 1.0},
        {"日期": "2023-01-04", "净值": 1.1},
        {"日期": "2023-01-05", "净值": 0.9},
        {"日期": "2023-01-06", "净值": 1.05},
    ]

    result = portfolio.result(annualization_days=252)

    assert math.isclose(result["累计收益百分比"], 5.0, rel_tol=0, abs_tol=1e-8)
    assert math.isclose(result["最大回撤百分比"], (0.9 / 1.1 - 1.0) * 100.0, rel_tol=0, abs_tol=1e-6)
    assert math.isfinite(result["年化复利百分比"])
    assert WIN_THRESHOLD_PCT > 0
    assert STOP_PCT > 0
