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


import subprocess
import sys


def _git(cwd, *args):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def _git_commit(cwd, message):
    subprocess.run(["git", "add", "-A"], cwd=cwd, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=cwd, check=True, capture_output=True)
    return _git(cwd, "rev-parse", "HEAD")


def _init_research_repo(tmp_path):
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, check=True, capture_output=True)
    _git(tmp_path, "config", "user.name", "测试")
    _git(tmp_path, "config", "user.email", "test@example.com")
    (tmp_path / "data/history/2026").mkdir(parents=True)
    (tmp_path / "data/backtest").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "requirements.txt").write_text("pandas==2.3.3\n", encoding="utf-8")
    (tmp_path / "scripts/model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "data/history/2026/2026-09-30.csv.gz").write_bytes(b"截止日前行情")
    (tmp_path / "data/history/2026/2026-10-08.csv.gz").write_bytes(b"旧的截止日后行情")
    (tmp_path / "data/history/_BACKFILL_STATE.json").write_text("{}", encoding="utf-8")
    (tmp_path / "data/universe.json").write_text("[]", encoding="utf-8")
    (tmp_path / "data/backtest/result.json").write_text('{"status":"research_only"}', encoding="utf-8")
    return _git_commit(tmp_path, "建立研究指纹测试基线")


def test_history_fingerprint_ignores_later_incremental_market_data(tmp_path):
    baseline = _init_research_repo(tmp_path)
    stable_before = history_data_fingerprint(tmp_path, "2026-09-30")
    full_before = history_data_fingerprint_at_commit(baseline, tmp_path)

    (tmp_path / "data/history/2026/2026-10-08.csv.gz").write_bytes(b"修复后的截止日后行情")
    (tmp_path / "data/history/2026/2026-10-09.csv.gz").write_bytes(b"新增的截止日后行情")
    (tmp_path / "data/history/_BACKFILL_STATE.json").write_text('{"status":"complete"}', encoding="utf-8")
    (tmp_path / "data/universe.json").write_text('[{"symbol":"600000.SH"}]', encoding="utf-8")
    _git_commit(tmp_path, "追加研究截止日后的增量行情")
    assert history_data_fingerprint(tmp_path, "2026-09-30") == stable_before
    assert history_data_fingerprint_at_commit(baseline, tmp_path) == full_before
    assert history_changes_only_after_cutoff(baseline, "2026-09-30", tmp_path)

    (tmp_path / "data/history/2026/2026-09-30.csv.gz").write_bytes(b"被修改的截止日前历史行情")
    _git_commit(tmp_path, "修改研究截止日前的历史行情")
    assert not history_changes_only_after_cutoff(baseline, "2026-09-30", tmp_path)
    assert history_data_fingerprint(tmp_path, "2026-09-30") != stable_before


def test_legacy_checkpoint_reuse_requires_code_params_inputs_and_output_to_match(tmp_path):
    baseline = _init_research_repo(tmp_path)
    output = tmp_path / "data/backtest/result.json"
    command = ["python", "scripts/model.py"]
    dependencies = {"scripts/model.py": file_sha256(tmp_path / "scripts/model.py")}
    inputs = {}
    requirements_sha = file_sha256(tmp_path / "requirements.txt")
    old_history = history_data_fingerprint_at_commit(baseline, tmp_path)
    legacy_fingerprint = canonical_hash({
        "schema_version": 1,
        "key": "sample_stage",
        "output": "data/backtest/result.json",
        "command": command,
        "dependencies": dependencies,
        "inputs": inputs,
        "requirements_sha256": requirements_sha,
        "history_data_fingerprint": old_history,
        "python_major_minor": list(sys.version_info[:2]),
    })
    checkpoint = {
        "schema_version": 1,
        "status": "success",
        "key": "sample_stage",
        "fingerprint": legacy_fingerprint,
        "output": "data/backtest/result.json",
        "output_sha256": file_sha256(output),
        "success_at": "2026-10-10T10:00:00+08:00",
        "public_commit": baseline,
        "command": command,
        "dependencies": dependencies,
        "inputs": inputs,
    }

    (tmp_path / "data/history/2026/2026-10-09.csv.gz").write_bytes(b"新交易日行情")
    _git_commit(tmp_path, "只追加截止日后的行情")
    assert compatible_legacy_checkpoint_reason(
        checkpoint, "sample_stage", command, dependencies, inputs, output,
        cutoff="2026-09-30", root=tmp_path,
    ) is None

    (tmp_path / "scripts/model.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git_commit(tmp_path, "修改阶段代码依赖")
    updated_dependencies = {"scripts/model.py": file_sha256(tmp_path / "scripts/model.py")}
    assert compatible_legacy_checkpoint_reason(
        checkpoint, "sample_stage", command, updated_dependencies, inputs, output,
        cutoff="2026-09-30", root=tmp_path,
    ) is not None
