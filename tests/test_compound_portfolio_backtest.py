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
