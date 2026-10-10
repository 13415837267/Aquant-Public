from scripts.short_term_release_validation import checkpoint_mismatches


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
