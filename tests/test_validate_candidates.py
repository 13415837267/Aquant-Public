import copy
import pytest
from scripts.validate_candidates import validate_candidates

def _payload():
    return {
        "as_of":"2026-09-30T18:00:00+08:00","status":"ready",
        "strategy_source":"Aquant-Private/main","strategy_version":"2.4.0","strategy_commit":"abc123",
        "market_scope":"沪深主板","universe":"沪深主板；排除 ST/退市相关标的",
        "history_window_start":"2026-09-03","history_window_end":"2026-09-30","history_files_used":20,
        "lookback_trading_days":20,"signal_horizon":"T收盘信号 → T+1开盘买入 → T+2起最早卖出 → 最长5个交易日",
        "factor_weights":{"momentum_short":0.25,"overnight_structure":0.15,"volume_activity":0.20,"price_strength":0.10,"liquidity":0.15,"safety":0.15},
        "market":{"breadth_pct":58.0,"median_return_pct":0.4,"regime":"risk_on"},
        "candidates":[
            {"rank":1,"symbol":"600000","name":"浦发银行","price":10.0,"change_pct":1.0,"return_3d_pct":2.0,"return_5d_pct":4.0,"return_10d_pct":6.0,"volume_ratio_5d":1.5,"turnover_pct":2.0,"amount":1e8,"volatility_10d_pct":3.0,"close_strength":0.9,"score":92.0,"precision_probability":0.92,"admission_tier":"primary_089"},
            {"rank":2,"symbol":"600001","name":"示例银行","price":9.0,"change_pct":0.5,"return_3d_pct":1.5,"return_5d_pct":3.0,"return_10d_pct":4.0,"volume_ratio_5d":1.2,"turnover_pct":1.5,"amount":9e7,"volatility_10d_pct":4.0,"close_strength":0.8,"score":90.0,"precision_probability":0.90,"admission_tier":"primary_089"},
        ],
        "candidate_admission_policy":"precision_top_2_with_089_primary_088_fallback_daily_top1_rescue_085_floor","diagnostics":{"candidate_count":2,"primary_candidate_count":2,"coverage_fallback_used":False,"daily_top1_rescue_used":False,"risk_off_no_trade":False},
        "future_function":False,
        "audit":{"hard_eligibility_applied_before_scoring":True,"strategy_source_locked_to_private":True,"short_term_features_only":True,"market_gate_applied":True},
    }

def test_validate_passes():
    result=validate_candidates(_payload())
    assert result["status"]=="pass" and result["candidate_count"]==2

@pytest.mark.parametrize("mutator",[
    lambda p:p.update({"as_of":"2026-09-29T18:00:00+08:00"}),
    lambda p:p.update({"future_function":True}),
    lambda p:p.update({"history_files_used":19}),
    lambda p:p["factor_weights"].update({"momentum_short":0.40}),
    lambda p:p["candidates"].append(copy.deepcopy(p["candidates"][0])),
    lambda p:p["candidates"].reverse(),
    lambda p:p["candidates"][0].update({"symbol":"300001"}),
    lambda p:p.update({"lookback_trading_days":126}),
])
def test_validate_rejects_inconsistent_snapshot(mutator):
    payload=_payload(); mutator(payload)
    with pytest.raises(RuntimeError): validate_candidates(payload)

def test_private_provenance_must_match_when_supplied():
    payload=_payload()
    with pytest.raises(RuntimeError,match="version mismatch"): validate_candidates(payload,private_version="1.9.0",private_commit="abc123")
    with pytest.raises(RuntimeError,match="commit mismatch"): validate_candidates(payload,private_version="2.4.0",private_commit="different")

def test_validate_rejects_non_production_timestamp():
    payload=_payload(); payload["as_of"]="2026-09-30T17:59:59+08:00"
    with pytest.raises(RuntimeError,match="invalid production timestamp"): validate_candidates(payload)

def test_validate_rejects_non_production_factor_weights():
    payload=_payload(); payload["factor_weights"]["momentum_short"]=0.34
    with pytest.raises(RuntimeError,match="factor weights"): validate_candidates(payload)


def test_validate_rejects_daily_rescue_below_085():
    payload = _payload()
    payload["candidates"] = [copy.deepcopy(payload["candidates"][0])]
    row = payload["candidates"][0]
    row["rank"] = 1
    row["score"] = 84.9
    row["precision_probability"] = 0.849
    row["admission_tier"] = "daily_top1_rescue"
    payload["diagnostics"]["candidate_count"] = 1
    payload["diagnostics"]["primary_candidate_count"] = 0
    payload["diagnostics"]["coverage_fallback_used"] = False
    payload["diagnostics"]["daily_top1_rescue_used"] = True
    with pytest.raises(RuntimeError, match="daily rescue probability floor"):
        validate_candidates(payload)
