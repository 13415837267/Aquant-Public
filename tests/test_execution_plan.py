import json

import pandas as pd
import pytest

from scripts.execution_plan import build_plan, validate


def market():
    return pd.DataFrame(
        [
            {
                "symbol": "600000",
                "date": "2026-10-01",
                "open": 10.0,
                "close": 10.0,
                "high_limit": 11.0,
                "low_limit": 9.0,
                "is_paused": 0,
                "is_st": 0,
            },
            {
                "symbol": "600001",
                "date": "2026-10-01",
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
                "name": "目标",
                "score": 90,
                "volatility_proxy": 1.0,
                "target_weight": 0.05,
            }
        ],
    }


def holdings_file(tmp_path, rows):
    path = tmp_path / "holdings.json"
    path.write_text(json.dumps({"holdings": rows}), encoding="utf-8")
    return path


def test_build_plan_creates_residual_exit_and_plan_id(tmp_path):
    path = holdings_file(
        tmp_path,
        {"600000": {"shares": 100, "available_shares": 100}},
    )
    payload = build_plan(
        portfolio(),
        market(),
        path,
        equity=100000,
        min_notional=1,
    )
    exits = [o for o in payload["orders"] if o["symbol"] == "600000"]
    buys = [o for o in payload["orders"] if o["symbol"] == "600001"]
    assert exits[0]["side"] == "sell"
    assert exits[0]["shares"] == 100
    assert buys[0]["side"] == "buy"
    assert payload["plan_id"]
    assert len({o["order_id"] for o in payload["orders"]}) == len(payload["orders"])
    validate(payload)


def test_plan_rejects_holdings_without_market_close(tmp_path):
    path = holdings_file(
        tmp_path,
        {"600999": {"shares": 100, "available_shares": 100}},
    )
    with pytest.raises(ValueError, match="missing/invalid market close"):
        build_plan(
            portfolio(),
            market(),
            path,
            equity=100000,
            min_notional=1,
        )


def test_plan_identity_changes_when_orders_change(tmp_path):
    path_a = holdings_file(
        tmp_path,
        {"600000": {"shares": 100, "available_shares": 100}},
    )
    first = build_plan(portfolio(), market(), path_a, equity=100000, min_notional=1)

    path_b = holdings_file(
        tmp_path,
        {"600000": {"shares": 200, "available_shares": 200}},
    )
    second = build_plan(portfolio(), market(), path_b, equity=100000, min_notional=1)
    assert first["plan_id"] != second["plan_id"]
