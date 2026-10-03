import pytest

from scripts.paper_replay_report import ReportError, summarize


def report():
    return {
        "mode": "paper_replay",
        "equity_curve": [
            {"execution_date": "2026-10-01", "equity": 100000},
            {"execution_date": "2026-10-02", "equity": 101000},
            {"execution_date": "2026-10-05", "equity": 99000},
            {"execution_date": "2026-10-06", "equity": 102000},
        ],
    }


def test_summary_is_descriptive_only():
    result = summarize(report())
    assert result["cycle_count"] == 4
    assert result["start_equity"] == 100000
    assert result["end_equity"] == 102000
    assert result["total_return"] == 0.02
    assert result["max_drawdown"] == 0.01980198
    assert result["positive_cycles"] == 2
    assert result["negative_cycles"] == 1
    assert result["audit"]["broker_api_used"] is False
    assert result["audit"]["descriptive_only"] is True


def test_summary_rejects_wrong_mode():
    data = report()
    data["mode"] = "live"
    with pytest.raises(ReportError, match="paper_replay"):
        summarize(data)


def test_summary_rejects_invalid_equity():
    data = report()
    data["equity_curve"][2]["equity"] = 0
    with pytest.raises(ReportError, match="invalid"):
        summarize(data)
