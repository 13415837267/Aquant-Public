from scripts.paper_portfolio import PortfolioError, apply_paper_fills, mark_to_market, new_paper_state


def buy_fill(fill_id="f1", shares=100, price=10.0, execution_date="2026-10-01", settlement_date="2026-10-02"):
    return {
        "fill_id": fill_id,
        "order_id": "o1",
        "symbol": "600000",
        "name": "示例",
        "side": "buy",
        "shares": shares,
        "price": price,
        "execution_date": execution_date,
        "settlement_date": settlement_date,
    }


def sell_fill(fill_id="f2", shares=100, price=11.0, execution_date="2026-10-02", settlement_date="2026-10-05"):
    return {
        "fill_id": fill_id,
        "order_id": "o2",
        "symbol": "600000",
        "name": "示例",
        "side": "sell",
        "shares": shares,
        "price": price,
        "execution_date": execution_date,
        "settlement_date": settlement_date,
    }


def test_buy_updates_cash_and_t1_inventory():
    state = new_paper_state(
        initial_cash=100000,
        as_of="2026-09-30",
        strategy_version="1.1.0",
        strategy_commit="test",
        cash_floor=5000,
    )
    next_state, entries = apply_paper_fills(state, [buy_fill()], execution_date="2026-10-01")
    position = next_state["positions"]["600000"]
    assert position["shares"] == 100
    assert position["available_shares"] == 0
    assert len(next_state["pending_settlements"]) == 1
    assert next_state["cash"] < 99000
    assert len(entries) == 1
    assert entries[0]["fill_id"] == "f1"


def test_duplicate_fill_is_idempotent():
    state = new_paper_state(
        initial_cash=100000,
        as_of="2026-09-30",
        strategy_version="1.1.0",
        strategy_commit="test",
        cash_floor=5000,
    )
    first, entries1 = apply_paper_fills(state, [buy_fill()], execution_date="2026-10-01")
    second, entries2 = apply_paper_fills(first, [buy_fill()], execution_date="2026-10-01")
    assert second == first
    assert len(entries1) == 1
    assert entries2 == []


def test_t1_prevents_same_day_sell():
    state = new_paper_state(
        initial_cash=100000,
        as_of="2026-09-30",
        strategy_version="1.1.0",
        strategy_commit="test",
        cash_floor=5000,
    )
    state, _ = apply_paper_fills(state, [buy_fill()], execution_date="2026-10-01")
    same_day_sell = sell_fill(
        fill_id="f2-same-day",
        execution_date="2026-10-01",
        settlement_date="2026-10-02",
    )
    try:
        apply_paper_fills(state, [same_day_sell], execution_date="2026-10-01")
    except PortfolioError as exc:
        assert "insufficient T+1 sellable shares" in str(exc)
    else:
        raise AssertionError("same-day buy must not be sellable")


def test_settled_shares_can_be_sold_and_realized_pnl_is_recorded():
    state = new_paper_state(
        initial_cash=100000,
        as_of="2026-09-30",
        strategy_version="1.1.0",
        strategy_commit="test",
        cash_floor=5000,
    )
    state, _ = apply_paper_fills(state, [buy_fill()], execution_date="2026-10-01")
    state, entries = apply_paper_fills(state, [sell_fill()], execution_date="2026-10-02")
    assert "600000" not in state["positions"]
    assert state["cash"] > 100000
    assert state["realized_pnl"] > 0
    assert entries[0]["side"] == "sell"


def test_cash_floor_blocks_buy_atomically():
    state = new_paper_state(
        initial_cash=10000,
        as_of="2026-09-30",
        strategy_version="1.1.0",
        strategy_commit="test",
        cash_floor=9900,
    )
    try:
        apply_paper_fills(state, [buy_fill(shares=1000, price=10.0)], execution_date="2026-10-01")
    except PortfolioError as exc:
        assert "cash floor" in str(exc)
    else:
        raise AssertionError("cash floor must block the fill")
    assert state["cash"] == 10000


def test_mark_to_market_updates_equity_without_changing_cash():
    state = new_paper_state(
        initial_cash=100000,
        as_of="2026-09-30",
        strategy_version="1.1.0",
        strategy_commit="test",
        cash_floor=5000,
    )
    state, _ = apply_paper_fills(state, [buy_fill()], execution_date="2026-10-01")
    cash_before = state["cash"]
    valued = mark_to_market(
        state,
        {"600000": {"symbol": "600000", "open": 11.0}},
    )
    assert valued["cash"] == cash_before
    assert valued["market_value"] == 1100.0
    assert valued["equity"] > cash_before
