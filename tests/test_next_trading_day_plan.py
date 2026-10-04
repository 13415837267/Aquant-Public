from datetime import date

from scripts.next_trading_day_plan import validate_plan


def test_next_trading_day_plan_must_match_candidate_snapshot():
    candidates = {
        "as_of": "2026-09-30T18:00:00+08:00",
        "strategy_version": "2.5.0",
        "strategy_commit": "41118d2ea019df87f67b240f418beafb31d4897a",
        "candidates": [
            {
                "rank": 1,
                "symbol": "601579",
                "name": "会稽山",
                "price": 37.3,
                "score": 88.601,
                "precision_probability": 0.886014,
                "admission_tier": "coverage_fallback_088",
            }
        ],
    }
    production = {
        "status": "not_released",
        "release_gate": False,
        "system_audit": False,
    }
    plan = {
        "status": "research_plan",
        "signal_date": "2026-09-30",
        "data_cutoff": "2026-09-30",
        "next_trading_day": "2026-10-08",
        "production_release": False,
        "strategy_version": "2.5.0",
        "strategy_commit": "41118d2ea019df87f67b240f418beafb31d4897a",
        "candidate_count": 1,
        "source_snapshot": {
            "path": "data/candidates.json",
            "as_of": "2026-09-30T18:00:00+08:00",
            "strategy_version": "2.5.0",
            "strategy_commit": "41118d2ea019df87f67b240f418beafb31d4897a",
        },
        "candidates": [
            {
                "rank": 1,
                "symbol": "601579",
                "name": "会稽山",
                "price": 37.3,
                "score": 88.601,
                "precision_probability": 0.886014,
                "admission_tier": "coverage_fallback_088",
                "signal_date": "2026-09-30",
                "execution_date": "2026-10-08",
                "earliest_exit_date": "2026-10-09",
                "net_win_threshold_pct": 1.0,
                "stop_loss_pct": 3.0,
                "flags": [],
            }
        ],
    }
    assert date.fromisoformat(plan["next_trading_day"]) > date.fromisoformat(plan["signal_date"])
    assert validate_plan(plan, candidates, production)["status"] == "pass"
