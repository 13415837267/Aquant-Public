import pandas as pd

from scripts.paper_rebalance import build_paper_plan, derive_paper_inputs
from scripts.paper_portfolio import new_paper_state, apply_paper_fills


def market(date="2026-10-01"):
    return pd.DataFrame(
        [
            {
                "symbol": "600000",
                "date": date,
                "open": 10.0,
                "close": 10.0,
                "high_limit": 11.0,
                "low_limit": 9.0,
                "is_paused": 0,
                "is_st": 0,
            },
            {
                "symbol": "600001",
                "date": date,
                "open": 20.0,
                "close": 20.0,
                "high_limit": 22.0,
                "low_limit": 18.0,
                "is_paused": 0,
                "is_st": 0,
            },
        ]
    ).set_index("symbol", drop=False)


def portfolio():
    return {
        "status": "ready",
        "strategy_source": "Aquant-Private/main",
        "strategy_version": "1.1.0",
        "strategy_commit": "test",
        "cash_buffer": 0.05,
        "positions": [
            {
                "rank": 1,
                "symbol": "600001",
                "name": "新目标",
                "score": 90,
                "volatility_proxy": 1.0,
                "target_weight": 0.05,
            }
        ],
    }


def test_paper_inputs_use_dynamic_equity_and_t1_inventory():
    state = new_paper_state(
        initial_cash=100000,
        as_of="2026-09-30",
        strategy_version="1.1.0",
        strategy_commit="test",
        cash_floor=5000,
    )
    fill = {
        "fill_id": "f1",
        "order_id": "o1",
        "symbol": "600000",
        "name": "旧仓",
        "side": "buy",
        "shares": 100,
        "price": 10,
        "execution_date": "2026-10-01",
        "settlement_date": "2026-10-02",
    }
    state, _ = apply_paper_fills(state, [fill], execution_date="2026-10-01")
    holdings, equity, provenance = derive_paper_inputs(state, market("2026-10-02"))
    assert holdings["600000"]["shares"] == 100
    assert holdings["600000"]["available_shares"] == 100
    assert equity < 100000
    assert provenance["broker_api_used"] is False


def test_next_plan_contains_exit_for_non_target_paper_holding():
    state = new_paper_state(
        initial_cash=100000,
        as_of="2026-09-30",
        strategy_version="1.1.0",
        strategy_commit="test",
        cash_floor=5000,
    )
    fill = {
        "fill_id": "f1",
        "order_id": "o1",
        "symbol": "600000",
        "name": "旧仓",
        "side": "buy",
        "shares": 100,
        "price": 10,
        "execution_date": "2026-10-01",
        "settlement_date": "2026-10-02",
    }
    state, _ = apply_paper_fills(state, [fill], execution_date="2026-10-01")
    plan = build_paper_plan(portfolio(), state, market("2026-10-02"), min_notional=1)
    exits = [o for o in plan["orders"] if o["symbol"] == "600000" and o["side"] == "sell"]
    entries = [o for o in plan["orders"] if o["symbol"] == "600001" and o["side"] == "buy"]
    assert exits
    assert exits[0]["shares"] == 100
    assert entries
    assert plan["mode"] == "paper"
    assert plan["audit"]["live_order_submission"] is False
