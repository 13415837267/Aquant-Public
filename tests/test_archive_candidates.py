from pathlib import Path
import json

import pytest

from scripts.archive_candidates import archive_candidates


def _payload():
    return {
        "as_of": "2026-09-30T18:00:00+08:00",
        "status": "ready",
        "candidates": [{"rank": 1, "symbol": "600000"}],
    }


def test_archive_candidates_is_idempotent(tmp_path):
    first = archive_candidates(_payload(), tmp_path)
    assert first == Path(tmp_path) / "2026" / "2026-09-30.json"

    first_content = first.read_text(encoding="utf-8")
    second = archive_candidates(_payload(), tmp_path)
    assert second == first
    assert second.read_text(encoding="utf-8") == first_content


def test_archive_candidates_rejects_mutation_of_existing_day(tmp_path):
    archive_candidates(_payload(), tmp_path)
    changed = _payload()
    changed["candidates"][0]["symbol"] = "600001"
    with pytest.raises(RuntimeError, match="immutable"):
        archive_candidates(changed, tmp_path)


def test_archive_candidates_rejects_non_ready(tmp_path):
    payload = _payload()
    payload["status"] = "pending"
    with pytest.raises(RuntimeError):
        archive_candidates(payload, tmp_path)

def test_archive_candidates_accepts_existing_equivalent_json_with_different_format(tmp_path):
    payload = _payload()
    destination = Path(tmp_path) / "2026" / "2026-09-30.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    assert archive_candidates(payload, tmp_path) == destination
