from pathlib import Path

from scripts.paper_execution_cycle import reconcile_ledger, run_cycle
from scripts.paper_portfolio import new_paper_state


def base_plan(side="buy", shares=100):
    return {
        "status": "ready_for_next_open_recheck",
        "reference_date": "2026-09-30",
        "starting_cash_reference": 100000,
        "equity_reference": 100000,
        "cash_buffer": 0.05,
        "turnover_cap": 0.30,
        "lot_size": 100,
        "strategy_version": "1.1.0",
        "strategy_commit": "test",
        "cost_assumptions_bps": {
            "commission": 3.0,
            "stamp_duty_sell": 5.0,
            "slippage": 2.0,
        },
        "orders": [
            {
                "order_id": "o1",
                "symbol": "600000",
                "name": "示例",
                "side": side,
                "shares": shares,
            }
        ],
        "summary": {"turnover": 0.01},
    }


def snapshot(execution_date="2026-10-01", settlement_date="2026-10-02", price=10.0):
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


def test_full_cycle_creates_state_and_is_idempotent():
    state, entries, gate = run_cycle(base_plan(), snapshot(), None)
    assert gate["broker_submission"] is False
    assert len(entries) == 1
    assert state["positions"]["600000"]["shares"] == 100
    assert state["positions"]["600000"]["available_shares"] == 0

    state2, entries2, _ = run_cycle(base_plan(), snapshot(), state)
    assert state2 == state
    assert entries2 == []


def test_reconcile_ledger_recovers_missing_audit_lines(tmp_path: Path):
    state = new_paper_state(
        initial_cash=100000,
        as_of="2026-09-30",
        strategy_version="1.1.0",
        strategy_commit="test",
        cash_floor=5000,
    )
    state, entries, _ = run_cycle(base_plan(), snapshot(), state)
    ledger = tmp_path / "fills.jsonl"
    assert reconcile_ledger(state, ledger) == 1
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1
    assert reconcile_ledger(state, ledger) == 0


def test_cycle_requires_valuation_snapshot_for_remaining_position():
    state, _, _ = run_cycle(base_plan(), snapshot(), None)
    no_orders_plan = base_plan()
    no_orders_plan["orders"] = []
    no_orders_plan["summary"] = {"turnover": 0.0}
    no_orders_plan["equity_reference"] = state["equity"]
    no_orders_plan["starting_cash_reference"] = state["cash"]
    incomplete = snapshot()
    incomplete["symbols"] = {}
    try:
        run_cycle(no_orders_plan, incomplete, state)
    except RuntimeError as exc:
        assert "missing next-open valuation snapshots" in str(exc)
    else:
        raise AssertionError("missing valuation snapshot must fail closed")
