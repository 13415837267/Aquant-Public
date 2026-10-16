from scripts.private_strategy_replay_backtest import evaluate_candidate, quality_gate


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
