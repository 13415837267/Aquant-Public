import copy

import pytest

from scripts.validate_candidates import validate_candidates


def _payload():
    return {
        "as_of": "2026-09-30T18:00:00+08:00",
        "status": "ready",
        "strategy_source": "Aquant-Private/main",
        "strategy_version": "1.1.0",
        "strategy_commit": "abc123",
        "market_scope": "沪深主板",
        "universe": "沪深主板；排除 ST/退市相关标的",
        "history_window_start": "2026-03-31",
        "history_window_end": "2026-09-30",
        "history_files_used": 126,
        "factor_weights": {
            "momentum": 0.35,
            "liquidity": 0.15,
            "value": 0.30,
            "safety": 0.20,
        },
        "candidates": [
            {
                "rank": 1,
                "symbol": "600000",
                "name": "浦发银行",
                "price": 10.0,
                "change_pct": 1.0,
                "momentum_60d": 12.0,
                "turnover_pct": 1.0,
                "amount": 100000000.0,
                "volatility_proxy": 1.2,
                "score": 92.0,
                "flags": [],
            },
            {
                "rank": 2,
                "symbol": "600001",
                "name": "示例银行",
                "price": 9.0,
                "change_pct": 0.5,
                "momentum_60d": 10.0,
                "turnover_pct": 0.8,
                "amount": 90000000.0,
                "volatility_proxy": 1.4,
                "score": 90.0,
                "flags": [],
            },
        ],
        "candidate_admission_policy": "top_score_3_max",
        "diagnostics": {"candidate_count": 2},
        "future_function": False,
        "audit": {
            "hard_eligibility_applied_before_scoring": True,
            "strategy_source_locked_to_private": True,
            "top_n_is_not_a_score_threshold": True,
        },
    }


def test_validate_passes():
    result = validate_candidates(_payload())
    assert result["status"] == "pass"
    assert result["candidate_count"] == 2


@pytest.mark.parametrize(
    "mutator",
    [
        lambda p: p.update({"as_of": "2026-09-29T18:00:00+08:00"}),
        lambda p: p.update({"future_function": True}),
        lambda p: p.update({"history_files_used": 125}),
        lambda p: p["factor_weights"].update({"momentum": 0.4}),
        lambda p: p["candidates"].append(copy.deepcopy(p["candidates"][0])),
        lambda p: p["candidates"].reverse(),
        lambda p: p["candidates"][0].update({"symbol": "300001"}),
    ],
)
def test_validate_rejects_inconsistent_snapshot(mutator):
    payload = _payload()
    mutator(payload)
    with pytest.raises(RuntimeError):
        validate_candidates(payload)


def test_private_provenance_must_match_when_supplied():
    payload = _payload()
    with pytest.raises(RuntimeError, match="version mismatch"):
        validate_candidates(payload, private_version="1.2.0", private_commit="abc123")

    with pytest.raises(RuntimeError, match="commit mismatch"):
        validate_candidates(payload, private_version="1.1.0", private_commit="different")

def test_validate_rejects_non_production_timestamp():
    payload = _payload()
    payload["as_of"] = "2026-09-30T17:59:59+08:00"
    with pytest.raises(RuntimeError, match="18:00"):
        validate_candidates(payload)


def test_validate_rejects_non_production_factor_weights():
    payload = _payload()
    payload["factor_weights"]["momentum"] = 0.34
    payload["factor_weights"]["liquidity"] = 0.16
    with pytest.raises(RuntimeError, match="momentum"):
        validate_candidates(payload)
