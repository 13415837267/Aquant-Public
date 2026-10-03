import pandas as pd
import pytest

from scripts.paper_snapshot_from_history import build_snapshot


def market(date="2026-10-01"):
    return pd.DataFrame(
        [
            {
                "symbol": "600000",
                "date": date,
                "open": 10.0,
                "high_limit": 11.0,
                "low_limit": 9.0,
                "is_paused": 0,
                "is_st": 0,
            }
        ]
    ).set_index("symbol", drop=False)


def plan():
    return {
        "reference_date": "2026-09-30",
        "orders": [{"symbol": "600000", "shares": 100}],
    }


def test_builds_historical_next_open_snapshot():
    result = build_snapshot(plan(), market(), settlement_date="2026-10-02")
    row = result["symbols"]["600000"]
    assert result["execution_date"] == "2026-10-01"
    assert row["open"] == 10.0
    assert row["symbol"] == "600000"
    assert result["audit"]["broker_api_used"] is False


def test_includes_remaining_state_positions_for_valuation():
    state = {"positions": {"600000": {"shares": 100}}}
    result = build_snapshot(
        {"reference_date": "2026-09-30", "orders": []},
        market(),
        settlement_date="2026-10-02",
        state=state,
    )
    assert "600000" in result["symbols"]


def test_rejects_stale_execution_date():
    with pytest.raises(ValueError, match="must be after plan reference date"):
        build_snapshot(plan(), market("2026-09-30"), settlement_date="2026-10-01")


def test_rejects_invalid_settlement_date():
    with pytest.raises(ValueError, match="settlement_date"):
        build_snapshot(plan(), market(), settlement_date="2026-10-01")


def test_rejects_missing_symbol():
    bad_plan = {
        "reference_date": "2026-09-30",
        "orders": [{"symbol": "600001", "shares": 100}],
    }
    with pytest.raises(ValueError, match="missing historical row"):
        build_snapshot(bad_plan, market(), settlement_date="2026-10-02")
