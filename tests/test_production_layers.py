import json

import pandas as pd

from scripts.execution_plan import build_plan, validate
from scripts.portfolio import build_portfolio


def test_build_portfolio_is_ready_and_respects_risk_caps():
    snapshot = {
        "status": "ready",
        "as_of": "2026-09-30T18:00:00+08:00",
        "strategy_source": "Aquant-Private/main",
        "strategy_version": "0.2.0",
        "strategy_commit": "test-commit",
        "candidates": [
            {"rank": 1, "symbol": "000001", "name": "A", "score": 90, "volatility_proxy": 2},
            {"rank": 2, "symbol": "000002", "name": "B", "score": 80, "volatility_proxy": 4},
            {"rank": 3, "symbol": "000003", "name": "C", "score": 70, "volatility_proxy": 8},
        ],
    }
    payload = build_portfolio(snapshot, max_weight=0.05, cash_buffer=0.05)

    assert payload["status"] == "ready"
    assert payload["strategy_commit"] == "test-commit"
    assert payload["audit"]["long_only"] is True
    assert payload["audit"]["leverage"] is False
    assert all(0 < p["target_weight"] <= 0.05 + 1e-12 for p in payload["positions"])


def test_execution_plan_enforces_lots_and_cash_floor(tmp_path):
    portfolio = {
        "status": "ready",
        "strategy_source": "Aquant-Private/main",
        "strategy_version": "0.2.0",
        "strategy_commit": "test-commit",
        "cash_buffer": 0.05,
        "positions": [
            {"rank": 1, "symbol": "000001", "name": "A", "target_weight": 0.05},
        ],
    }
    market = pd.DataFrame(
        {
            "symbol": ["000001"],
            "date": ["2026-09-30"],
            "open": [10.0],
            "close": [10.0],
            "high_limit": [11.0],
            "low_limit": [9.0],
            "is_paused": [0],
            "is_st": [0],
        }
    )
    holdings = tmp_path / "holdings.json"
    holdings.write_text(json.dumps({"holdings": {}}), encoding="utf-8")

    payload = build_plan(
        portfolio,
        market.set_index("symbol", drop=False),
        holdings,
        equity=1000000,
        turnover_cap=0.30,
    )
    validate(payload)

    assert payload["status"] == "ready_for_next_open_recheck"
    assert payload["summary"]["turnover"] <= 0.30 + 1e-8
    assert all(order["shares"] % 100 == 0 for order in payload["orders"])
    assert payload["summary"]["estimated_cash_after"] >= 50000 - 1e-6
