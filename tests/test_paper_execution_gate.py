from scripts.paper_execution_gate import GateError, build_paper_decisions


def plan():
    return {
        "status": "ready_for_next_open_recheck",
        "reference_date": "2026-09-30",
        "strategy_version": "1.1.0",
        "strategy_commit": "test",
        "lot_size": 100,
        "turnover_cap": 0.30,
        "summary": {"turnover": 0.149082},
        "orders": [
            {"symbol": "601988", "side": "buy", "shares": 7400},
        ],
    }


def snapshot(open_price=6.80, high_limit=7.40, low_limit=6.06):
    return {
        "symbol": "601988",
        "open": open_price,
        "high_limit": high_limit,
        "low_limit": low_limit,
        "is_paused": False,
        "is_st": False,
    }


def test_releases_paper_order_when_gate_passes():
    result = build_paper_decisions(plan(), {"601988": snapshot()})
    assert result["status"] == "paper_released"
    assert result["broker_submission"] is False
    assert result["orders"][0]["shares"] == 7400
    assert result["orders"][0]["reference_price"] == 6.8


def test_fails_closed_on_missing_snapshot():
    try:
        build_paper_decisions(plan(), {})
    except GateError as exc:
        assert "missing next-open snapshot" in str(exc)
    else:
        raise AssertionError("missing snapshot must fail closed")


def test_blocks_buy_at_upper_limit():
    try:
        build_paper_decisions(plan(), {"601988": snapshot(open_price=7.40)})
    except GateError as exc:
        assert "upper limit" in str(exc)
    else:
        raise AssertionError("upper-limit buy must be blocked")


def test_blocks_paused_security():
    data = snapshot()
    data["is_paused"] = True
    try:
        build_paper_decisions(plan(), {"601988": data})
    except GateError as exc:
        assert "paused" in str(exc)
    else:
        raise AssertionError("paused security must be blocked")
