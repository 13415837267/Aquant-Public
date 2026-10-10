import pytest

from scripts.compound_portfolio_backtest import 解析退出


def test_t_plus_one_entry_day_cannot_exit_even_when_target_touched():
    reason, net_return, mark_price = 解析退出(
        entry=10.0,
        high=10.5,
        low=9.9,
        close=10.2,
        持有日序号=0,
        目标净收益百分比=1.0,
        止损百分比=3.0,
        成本基点=10.0,
    )
    assert reason is None
    assert net_return is None
    assert mark_price == pytest.approx(10.2)


def test_t_plus_two_target_hit_returns_one_percent_net_of_costs():
    reason, net_return, exit_price = 解析退出(
        entry=10.0,
        high=10.12,
        low=10.0,
        close=10.05,
        持有日序号=1,
        目标净收益百分比=1.0,
        止损百分比=3.0,
        成本基点=10.0,
    )
    assert reason == "止盈"
    assert net_return == pytest.approx(1.0)
    assert exit_price == pytest.approx(10.11)


def test_stop_loss_has_priority_when_stop_and_target_are_both_touched():
    reason, net_return, exit_price = 解析退出(
        entry=10.0,
        high=10.2,
        low=9.6,
        close=10.0,
        持有日序号=1,
        目标净收益百分比=1.0,
        止损百分比=3.0,
        成本基点=10.0,
    )
    assert reason == "止损"
    assert net_return == pytest.approx(-3.1)
    assert exit_price == pytest.approx(9.7)


def test_fifth_entry_session_closes_at_market_price_when_no_barrier_hit():
    reason, net_return, exit_price = 解析退出(
        entry=10.0,
        high=10.05,
        low=9.95,
        close=10.2,
        持有日序号=4,
        目标净收益百分比=1.0,
        止损百分比=3.0,
        成本基点=10.0,
    )
    assert reason == "到期"
    assert net_return == pytest.approx(1.9)
    assert exit_price == pytest.approx(10.2)


def test_stop_gap_uses_open_price_instead_of_assumed_stop_fill():
    reason, net_return, exit_price = 解析退出(
        entry=10.0,
        high=10.0,
        low=9.4,
        close=9.6,
        持有日序号=1,
        目标净收益百分比=1.0,
        止损百分比=3.0,
        成本基点=10.0,
        开盘价=9.5,
    )
    assert reason == "止损"
    assert exit_price == pytest.approx(9.5)
    assert net_return == pytest.approx(-5.1)


def test_target_gap_uses_better_open_price():
    reason, net_return, exit_price = 解析退出(
        entry=10.0,
        high=10.4,
        low=10.2,
        close=10.3,
        持有日序号=1,
        目标净收益百分比=1.0,
        止损百分比=3.0,
        成本基点=10.0,
        开盘价=10.2,
    )
    assert reason == "止盈"
    assert exit_price == pytest.approx(10.2)
    assert net_return == pytest.approx(1.9)


def test_exit_policy_grid_selects_only_risk_eligible_validation_candidate():
    from scripts.compound_portfolio_backtest import 选择退出参数
    grid = [
        {"target_net_profit_pct": 1.0, "stop_loss_pct": 3.0, "completed_trades": 480,
         "compound_return_pct": -42.0, "annualized_compound_return_pct": -24.0,
         "max_drawdown_pct": -44.0},
        {"target_net_profit_pct": 1.5, "stop_loss_pct": 1.5, "completed_trades": 450,
         "compound_return_pct": 8.0, "annualized_compound_return_pct": 4.0,
         "max_drawdown_pct": -22.0},
        {"target_net_profit_pct": 2.0, "stop_loss_pct": 1.0, "completed_trades": 460,
         "compound_return_pct": 50.0, "annualized_compound_return_pct": 20.0,
         "max_drawdown_pct": -40.0},
        {"target_net_profit_pct": 2.0, "stop_loss_pct": 2.0, "completed_trades": 40,
         "compound_return_pct": 99.0, "annualized_compound_return_pct": 30.0,
         "max_drawdown_pct": -5.0},
    ]
    picked = 选择退出参数(grid, minimum_trades=100, max_drawdown_floor_pct=-30.0)
    assert picked["target_net_profit_pct"] == 1.5
    assert picked["stop_loss_pct"] == 1.5
    assert picked["selection_status"] == "验证集正收益候选"


def test_exit_policy_grid_returns_none_when_no_policy_meets_validation_gates():
    from scripts.compound_portfolio_backtest import 选择退出参数
    grid = [
        {"target_net_profit_pct": 1.0, "stop_loss_pct": 3.0, "completed_trades": 90,
         "compound_return_pct": 2.0, "annualized_compound_return_pct": 1.0,
         "max_drawdown_pct": -10.0},
        {"target_net_profit_pct": 1.5, "stop_loss_pct": 2.0, "completed_trades": 400,
         "compound_return_pct": -12.0, "annualized_compound_return_pct": -6.0,
         "max_drawdown_pct": -31.0},
    ]
    assert 选择退出参数(grid, minimum_trades=100, max_drawdown_floor_pct=-30.0) is None



def test_probability_threshold_selection_uses_only_eligible_validation_rows():
    from scripts.compound_portfolio_backtest import 选择概率阈值
    grid = [
        {"probability_threshold": 0.55, "completed_trades": 300,
         "compound_return_pct": -20.0, "annualized_compound_return_pct": -10.0,
         "max_drawdown_pct": -35.0},
        {"probability_threshold": 0.60, "completed_trades": 240,
         "compound_return_pct": 3.0, "annualized_compound_return_pct": 1.5,
         "max_drawdown_pct": -20.0},
        {"probability_threshold": 0.65, "completed_trades": 80,
         "compound_return_pct": 30.0, "annualized_compound_return_pct": 15.0,
         "max_drawdown_pct": -10.0},
        {"probability_threshold": 0.70, "completed_trades": 180,
         "compound_return_pct": 2.0, "annualized_compound_return_pct": 1.0,
         "max_drawdown_pct": -18.0},
    ]
    picked = 选择概率阈值(grid, minimum_trades=100, max_drawdown_floor_pct=-30.0)
    assert picked["probability_threshold"] == 0.60
    assert picked["selection_status"] == "验证集正收益候选"


def test_probability_threshold_selection_returns_none_when_validation_has_no_positive_compounding():
    from scripts.compound_portfolio_backtest import 选择概率阈值
    grid = [
        {"probability_threshold": 0.60, "completed_trades": 220,
         "compound_return_pct": -5.0, "annualized_compound_return_pct": -2.0,
         "max_drawdown_pct": -20.0},
        {"probability_threshold": 0.65, "completed_trades": 180,
         "compound_return_pct": -10.0, "annualized_compound_return_pct": -5.0,
         "max_drawdown_pct": -15.0},
    ]
    assert 选择概率阈值(grid, minimum_trades=100, max_drawdown_floor_pct=-30.0) is None


def test_probability_threshold_selection_returns_none_if_no_candidate_passes_risk_gates():
    from scripts.compound_portfolio_backtest import 选择概率阈值
    grid = [
        {"probability_threshold": 0.60, "completed_trades": 90,
         "compound_return_pct": 2.0, "annualized_compound_return_pct": 1.0,
         "max_drawdown_pct": -10.0},
        {"probability_threshold": 0.65, "completed_trades": 200,
         "compound_return_pct": 4.0, "annualized_compound_return_pct": 2.0,
         "max_drawdown_pct": -31.0},
    ]
    assert 选择概率阈值(grid, minimum_trades=100, max_drawdown_floor_pct=-30.0) is None



def test_candidate_rank_selection_uses_t_plus_one_execution_data_only():
    import numpy as np
    import pandas as pd
    from scripts.compound_portfolio_backtest import select_executable_rank_indices

    entry_day = pd.DataFrame([
        {"symbol": "600000", "open": 10.0, "high_limit": 11.0, "is_paused": 0},
        {"symbol": "000001", "open": 10.0, "high_limit": 11.0, "is_paused": 0},
        {"symbol": "300001", "open": 10.0, "high_limit": 10.0, "is_paused": 0},
    ])
    probs = np.asarray([0.70, 0.90, 0.99])

    selected, executable = select_executable_rank_indices(
        probs, ["600000", "000001", "300001"], entry_day, top_k=2
    )

    assert executable.tolist() == [True, True, False]
    assert selected.tolist() == [1, 0]


def test_candidate_bar_path_keeps_missing_future_bar_as_unavailable():
    import numpy as np
    import pandas as pd
    from scripts.compound_portfolio_backtest import candidate_bar_path

    entry_day = pd.DataFrame([{
        "symbol": "600000", "open": 10.0, "high": 10.1,
        "low": 9.9, "close": 10.0, "high_limit": 11.0, "is_paused": 0
    }])
    missing_day = pd.DataFrame([{
        "symbol": "000001", "open": 10.0, "high": 10.1,
        "low": 9.9, "close": 10.0, "high_limit": 11.0, "is_paused": 0
    }])

    bars = candidate_bar_path(
        "600000", [entry_day, missing_day], ["2026-01-02", "2026-01-05"], 10.0
    )

    assert bars[0]["bar_available"] is True
    assert bars[1]["bar_available"] is False
    assert np.isnan(bars[1]["close"])
