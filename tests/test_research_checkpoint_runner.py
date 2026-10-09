from __future__ import annotations

import json

from scripts.research_checkpoint_runner import (
    canonical_hash,
    checkpoint_reuse_reason,
    file_sha256,
)
from scripts.run_all_conditions_research import summarize_stage_status


def test_canonical_fingerprint_is_order_independent_and_input_sensitive():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    assert canonical_hash({"data": "v1"}) != canonical_hash({"data": "v2"})


def test_checkpoint_reuse_requires_matching_code_and_parameters(tmp_path):
    output = tmp_path / "result.json"
    output.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    fingerprint = canonical_hash({"script": "abc", "period": "2015-2026"})
    checkpoint = {
        "status": "success",
        "fingerprint": fingerprint,
        "output_sha256": file_sha256(output),
    }

    assert checkpoint_reuse_reason(checkpoint, fingerprint, output) is None
    assert "指纹" in checkpoint_reuse_reason(
        checkpoint, canonical_hash({"script": "changed", "period": "2015-2026"}), output
    )


def test_checkpoint_reuse_rejects_missing_or_modified_result(tmp_path):
    output = tmp_path / "result.json"
    output.write_text('{"value": 1}', encoding="utf-8")
    checkpoint = {
        "status": "success",
        "fingerprint": "same",
        "output_sha256": file_sha256(output),
    }
    output.write_text('{"value": 2}', encoding="utf-8")
    assert "摘要" in checkpoint_reuse_reason(checkpoint, "same", output)
    assert "不存在" in checkpoint_reuse_reason(
        {"status": "running", "fingerprint": "same"}, "same", output
    )


def test_individual_stage_runs_keep_overall_manifest_in_running_state_until_all_pass():
    keys = ["stage_a", "stage_b", "stage_c"]
    status, successful, failed = summarize_stage_status(
        {"stage_a": {"status": "success"}}, keys
    )
    assert status == "running"
    assert successful == ["stage_a"]
    assert failed == []

    status, successful, failed = summarize_stage_status(
        {"stage_a": {"status": "success"}, "stage_b": {"status": "failed"}}, keys
    )
    assert status == "failed"
    assert successful == ["stage_a"]
    assert failed == ["stage_b"]

    status, successful, failed = summarize_stage_status(
        {key: {"status": "success"} for key in keys}, keys
    )
    assert status == "completed"
    assert successful == keys
    assert failed == []
