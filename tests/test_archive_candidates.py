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


def test_archive_candidates_versions_same_day_when_payload_changes(tmp_path):
    first = _payload()
    archive_candidates(first, tmp_path)
    changed = _payload()
    changed["candidates"][0]["symbol"] = "600001"
    changed["strategy_version"] = "新训练基准"
    changed["strategy_commit"] = "1234567890abcdef"
    path = archive_candidates(changed, tmp_path)
    assert path.name == "2026-09-30__新训练基准__1234567890.json"
    assert path.exists()
    assert path != tmp_path / "2026" / "2026-09-30.json"


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
def test_archive_candidates_uses_content_hash_when_same_version_changes(tmp_path):
    first = _payload()
    first["strategy_version"] = "固定基准"
    first["strategy_commit"] = "3df9ef34ee23a86d6f844ceb9e96bf4b4f387592"
    archive_candidates(first, tmp_path)
    changed = _payload()
    changed["strategy_version"] = "固定基准"
    changed["strategy_commit"] = "3df9ef34ee23a86d6f844ceb9e96bf4b4f387592"
    changed["candidates"][0]["symbol"] = "600001"
    path = archive_candidates(changed, tmp_path)
    assert path.name != "2026-09-30__固定基准__3df9ef34ee.json"
    assert path.name.startswith("2026-09-30__固定基准__3df9ef34ee__")
    assert path.name.endswith(".json")
    assert path.exists()
