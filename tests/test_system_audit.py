from __future__ import annotations

import pytest

from scripts.system_audit import REQUIRED_HISTORY_FIELDS, validate_history_completion_marker


def _marker() -> dict:
    return {
        "completed_at": "2026-10-13T20:55:04+08:00",
        "target_start": "2026-10-13",
        "end": "2026-10-13",
        "trading_days": 1,
        "daily_fields": sorted(REQUIRED_HISTORY_FIELDS),
    }


def test_incremental_completion_marker_does_not_require_status_field():
    marker = _marker()
    assert "status" not in marker
    assert validate_history_completion_marker(marker) is True


@pytest.mark.parametrize(
    "mutator",
    [
        lambda marker: marker.pop("completed_at"),
        lambda marker: marker.update({"end": "not-a-date"}),
        lambda marker: marker.update({"daily_fields": ["symbol", "date", "close"]}),
    ],
)
def test_invalid_history_completion_marker_is_rejected(mutator):
    marker = _marker()
    mutator(marker)
    with pytest.raises(RuntimeError):
        validate_history_completion_marker(marker)
