from __future__ import annotations

import scripts.candidate_count_operating_point_research as research


def _row(top_k: int, threshold: float, target: float, stop: float, compound: float) -> dict:
    return {
        "status": "ready",
        "top_k": top_k,
        "probability_threshold": threshold,
        "net_target_pct": target,
        "stop_loss_pct": stop,
        "completed_trades": 120,
        "compound_return_pct": compound,
        "annualized_compound_return_pct": compound / 2,
        "max_drawdown_pct": -8.0,
        "mean_trade_net_return_pct": 0.2,
    }


def test_each_candidate_count_selects_its_own_validation_operating_point(monkeypatch):
    grid_by_top_k = {
        1: [_row(1, 0.65, 3.0, 2.5, 4.0), _row(1, 0.70, 5.0, 2.0, 6.0)],
        2: [_row(2, 0.60, 2.0, 3.0, 7.0), _row(2, 0.65, 8.0, 2.5, 5.0)],
    }
    calls = []

    def fake_sweep(candidates, diagnostics, dates, start, end, top_k):
        calls.append(top_k)
        return grid_by_top_k[top_k]

    monkeypatch.setattr(research.core, "_exit_parameter_sweep", fake_sweep)
    result = research.independently_select_parameters(
        daily_candidates={},
        diagnostics={},
        dates=["2026-01-02", "2026-01-05"],
        start_date="2026-01-02",
        end_date="2026-01-05",
    )

    assert calls == [1, 2]
    assert result[1]["selected"]["probability_threshold"] == 0.70
    assert result[1]["selected"]["net_target_pct"] == 5.0
    assert result[2]["selected"]["probability_threshold"] == 0.60
    assert result[2]["selected"]["net_target_pct"] == 2.0
    assert result[1]["selected"] != result[2]["selected"]


def test_eligibility_gate_rejects_low_sample_negative_return_and_high_drawdown():
    base = _row(1, 0.65, 3.0, 2.5, 4.0)
    assert research.eligible_operating_point(base, 100, -15.0)
    assert not research.eligible_operating_point({**base, "completed_trades": 99}, 100, -15.0)
    assert not research.eligible_operating_point({**base, "compound_return_pct": -0.1}, 100, -15.0)
    assert not research.eligible_operating_point({**base, "max_drawdown_pct": -16.0}, 100, -15.0)
    assert not research.eligible_operating_point({**base, "mean_trade_net_return_pct": 0.0}, 100, -15.0)


def test_adjacent_parameter_grid_changes_only_one_parameter():
    selected = {
        "probability_threshold": 0.65,
        "net_target_pct": 3.0,
        "stop_loss_pct": 2.5,
    }
    neighbors = research.adjacent_operating_points(
        selected,
        threshold_grid=(0.60, 0.65, 0.70),
        target_grid=(2.0, 3.0, 4.0),
        stop_grid=(2.0, 2.5, 3.0),
    )
    assert len(neighbors) == 6
    selected_tuple = (0.65, 3.0, 2.5)
    for row in neighbors:
        changed = (
            row["probability_threshold"] != selected_tuple[0],
            row["net_target_pct"] != selected_tuple[1],
            row["stop_loss_pct"] != selected_tuple[2],
        )
        assert sum(changed) == 1
        assert row["changed_parameter"] in {
            "probability_threshold",
            "net_target_pct",
            "stop_loss_pct",
        }


def test_adjacent_parameter_grid_rejects_parameters_outside_declared_grid():
    try:
        research.adjacent_operating_points(
            {"probability_threshold": 0.66, "net_target_pct": 3.0, "stop_loss_pct": 2.5},
            threshold_grid=(0.60, 0.65, 0.70),
            target_grid=(2.0, 3.0, 4.0),
            stop_grid=(2.0, 2.5, 3.0),
        )
    except ValueError as exc:
        assert "不在预先声明的网格内" in str(exc)
    else:
        raise AssertionError("网格外参数必须拒绝")
