from types import SimpleNamespace

import pytest

from scripts.private_strategy_replay_backtest import (
    assert_reconstruction_contract,
    assert_replay_versions,
    evaluate_candidate,
    quality_gate,
)


def _metrics(*, trades=120, compound=5.0, mean=0.1, profit_factor=1.3, drawdown=-8.0):
    return {
        "completed_trades": trades,
        "compound_return_pct": compound,
        "mean_trade_net_return_pct": mean,
        "profit_factor": profit_factor,
        "max_drawdown_pct": drawdown,
    }


def test_quality_gate_requires_trades_positive_net_profit_and_risk_limits():
    assert quality_gate(_metrics(), 100)
    assert not quality_gate(_metrics(trades=99), 100)
    assert not quality_gate(_metrics(compound=0.0), 100)
    assert not quality_gate(_metrics(mean=0.0), 100)
    assert not quality_gate(_metrics(profit_factor=1.19), 100)
    assert not quality_gate(_metrics(drawdown=-15.01), 100)


def test_candidate_must_pass_cost_gates_and_not_degrade_any_comparison_window():
    costs = ("10_bps", "15_bps", "20_bps")
    baseline = {
        window: {cost: _metrics(compound=2.0, drawdown=-8.0) for cost in costs}
        for window in ("validation", "final_supplemental")
    }
    candidate = {
        window: {cost: _metrics(compound=2.5, drawdown=-7.5) for cost in costs}
        for window in ("validation", "final_supplemental")
    }
    result = evaluate_candidate(baseline, candidate)
    assert result["all_quality_gates_passed"]
    assert result["all_baseline_comparisons_passed"]
    assert result["strict_improvement_over_baseline"]
    assert result["decision"] == "保留研究候选"


def test_candidate_is_rejected_if_any_cost_case_or_period_fails():
    costs = ("10_bps", "15_bps", "20_bps")
    baseline = {
        window: {cost: _metrics(compound=2.0, drawdown=-8.0) for cost in costs}
        for window in ("validation", "final_supplemental")
    }
    candidate = {
        window: {cost: _metrics(compound=2.5, drawdown=-7.5) for cost in costs}
        for window in ("validation", "final_supplemental")
    }
    candidate["final_supplemental"]["20_bps"] = _metrics(compound=-0.1)
    result = evaluate_candidate(baseline, candidate)
    assert not result["all_quality_gates_passed"]
    assert not result["strict_improvement_over_baseline"]
    assert result["decision"] == "拒绝本次策略改动"


def test_replay_cost_scenarios_include_base_and_stress_costs():
    from scripts.private_strategy_replay_backtest import BASE_COST_BPS, COST_MULTIPLIERS

    assert 1.0 in COST_MULTIPLIERS
    assert tuple(sorted(set(COST_MULTIPLIERS))) == COST_MULTIPLIERS
    cost_levels = {round(BASE_COST_BPS * multiplier, 6) for multiplier in COST_MULTIPLIERS}
    assert BASE_COST_BPS in cost_levels
    assert BASE_COST_BPS * 1.5 in cost_levels
    assert BASE_COST_BPS * 2.0 in cost_levels


def _contract_strategy(*, candidate: bool):
    precision = SimpleNamespace(
        COMPONENT_WEIGHTS={
            "trend": 0.32,
            "volume_confirmation": 0.18,
            "price_location": 0.16,
            "overnight_structure": 0.10,
            "risk_efficiency": 0.24,
        },
        MIN_RESEARCH_SCORE=58.0,
        MAX_CANDIDATES=2,
        MIN_MARKET_BREADTH_PCT=45.0,
        MIN_MARKET_MEDIAN_RETURN_PCT=-1.25,
        MIN_MARKET_ABOVE_MA20_PCT=25.0,
        MIN_MARKET_POSITIVE_5D_PCT=30.0,
    )
    return {
        "version": "3.0.0" if candidate else "2.6.0",
        "model": SimpleNamespace(STRATEGY_METHOD="risk_adjusted_cross_sectional_rank_v1") if candidate else SimpleNamespace(),
        "policy": SimpleNamespace(ADMISSION_POLICY="risk_adjusted_cross_sectional_rank_top_2") if candidate else SimpleNamespace(),
        "precision": precision,
        "minimum_price": 2.0,
        "minimum_turnover_amount": 30000000.0,
        "minimum_history_days": 20,
        "max_holding_sessions": 5,
        "maximum_exit_session_after_signal": 5,
        "signal_at": "T_close",
        "entry": "T+1_open",
        "earliest_exit": "T+2",
        "entry_limit_up_block": True,
        "minimum_research_score": 58.0,
        "maximum_candidates": 2,
        "minimum_market_breadth": 45.0,
        "minimum_market_median_return": -1.25,
        "minimum_market_above_ma20": 25.0,
        "minimum_market_positive_5d": 30.0,
        "target_return_pct": 2.5 if candidate else 1.0,
        "stop_loss_pct": 2.0 if candidate else 3.0,
    }


def test_reconstruction_contract_accepts_3_0_and_preserves_execution_timing():
    assert_reconstruction_contract(_contract_strategy(candidate=False), _contract_strategy(candidate=True))


def test_reconstruction_contract_rejects_2_9_and_invalid_stop_loss():
    baseline = _contract_strategy(candidate=False)
    candidate = _contract_strategy(candidate=True)
    with pytest.raises(ValueError, match="候选 3.0.0"):
        assert_replay_versions({"version": "2.6.0"}, {"version": "2.9.0"})
    candidate["stop_loss_pct"] = 4.0
    with pytest.raises(ValueError, match="止损"):
        assert_reconstruction_contract(baseline, candidate)
