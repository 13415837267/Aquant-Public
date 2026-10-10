from scripts.short_term_release_validation import (
    checkpoint_mismatches,
    research_implementation_fingerprint,
)


def test_checkpoint_mismatch_detects_strategy_commit_drift():
    expected = {
        "schema_version": 2,
        "strategy_version": "2.5.0",
        "strategy_commit": "current-commit",
        "cost_bps": 3.0,
    }
    checkpoint = {
        **expected,
        "strategy_commit": "older-commit",
    }

    assert checkpoint_mismatches(checkpoint, expected) == ["strategy_commit"]


def test_checkpoint_match_allows_resume():
    expected = {
        "schema_version": 2,
        "strategy_version": "2.5.0",
        "strategy_commit": "same-commit",
        "cost_bps": 3.0,
    }

    assert checkpoint_mismatches(expected, expected) == []



def test_checkpoint_rejects_different_research_implementation():
    expected = {
        "schema_version": 2,
        "strategy_commit": "same-private-commit",
        "implementation_fingerprint": "new-code-fingerprint",
    }
    checkpoint = {
        **expected,
        "implementation_fingerprint": "old-code-fingerprint",
    }

    assert checkpoint_mismatches(checkpoint, expected) == ["implementation_fingerprint"]


def test_research_implementation_fingerprint_changes_when_logic_changes(tmp_path):
    scripts = tmp_path / "scripts"
    config = tmp_path / "config"
    scripts.mkdir()
    config.mkdir()
    code_file = scripts / "model.py"
    config_file = config / "params.json"
    code_file.write_text("LABEL_VERSION = 1\\n", encoding="utf-8")
    config_file.write_text('{"cost_bps": 10}\\n', encoding="utf-8")

    before = research_implementation_fingerprint(
        tmp_path, ("scripts/model.py", "config/params.json")
    )
    code_file.write_text("LABEL_VERSION = 2\\n", encoding="utf-8")
    after = research_implementation_fingerprint(
        tmp_path, ("scripts/model.py", "config/params.json")
    )

    assert before != after
