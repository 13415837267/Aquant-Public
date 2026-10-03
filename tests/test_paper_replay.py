import pytest

from scripts.paper_replay import ReplayError, run_replay


def plan(reference_date, side, order_id, shares=100):
    return {
        "status": "ready_for_next_open_recheck",
        "reference_date": reference_date,
        "equity_reference": 200000,
        "starting_cash_reference": 200000,
        "cash_buffer": 0.05,
        "turnover_cap": 0.30,
        "lot_size": 100,
        "minimum_order_notional": 1,
        "strategy_version": "1.1.0",
        "strategy_commit": "test",
        "cost_assumptions_bps": {
            "commission": 3.0,
            "stamp_duty_sell": 5.0,
            "slippage": 2.0,
        },
        "summary": {"turnover": 0.001},
        "orders": [
            {
                "order_id": order_id,
                "symbol": "600000",
                "name": "示例",
                "side": side,
                "shares": shares,
                "requested_shares": shares,
            }
        ],
    }


def snapshot(execution_date, settlement_date, price):
    return {
        "execution_date": execution_date,
        "settlement_date": settlement_date,
        "symbols": {
            "600000": {
                "symbol": "600000",
                "date": execution_date,
                "open": price,
                "high_limit": price * 1.1,
                "low_limit": price * 0.9,
                "is_paused": False,
                "is_st": False,
            }
        },
    }


def test_two_day_replay_carries_state_and_t1():
    cycles = [
        {
            "plan": plan("2026-09-30", "buy", "buy-1"),
            "snapshot": snapshot("2026-10-01", "2026-10-02", 10.0),
        },
        {
            "plan": plan("2026-10-01", "sell", "sell-1"),
            "snapshot": snapshot("2026-10-02", "2026-10-05", 11.0),
        },
    ]
    final_state, curve = run_replay(cycles)
    assert len(curve) == 2
    assert curve[0]["position_count"] == 1
    assert curve[1]["position_count"] == 0
    assert final_state["positions"] == {}
    assert final_state["realized_pnl"] > 0
    assert final_state["replay"]["cycle_count"] == 2
    assert final_state["audit"]["broker_api_used"] is False


def test_replay_rejects_non_monotonic_execution_dates():
    cycles = [
        {
            "plan": plan("2026-09-30", "buy", "buy-1"),
            "snapshot": snapshot("2026-10-02", "2026-10-05", 10.0),
        },
        {
            "plan": plan("2026-10-01", "sell", "sell-1"),
            "snapshot": snapshot("2026-10-01", "2026-10-02", 11.0),
        },
    ]
    with pytest.raises(ReplayError, match="strictly increasing"):
        run_replay(cycles)


def test_replay_rejects_strategy_change():
    first = plan("2026-09-30", "buy", "buy-1")
    second = plan("2026-10-01", "sell", "sell-1")
    second["strategy_version"] = "9.9.9"
    with pytest.raises(ReplayError, match="strategy version changed"):
        run_replay(
            [
                {
                    "plan": first,
                    "snapshot": snapshot("2026-10-01", "2026-10-02", 10.0),
                },
                {
                    "plan": second,
                    "snapshot": snapshot("2026-10-02", "2026-10-05", 11.0),
                },
            ]
        )


def test_replay_requires_cycles():
    with pytest.raises(ReplayError, match="at least one"):
        run_replay([])
