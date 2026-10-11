import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.path_aware_gbdt_compound_backtest import (
    complete_signal_window_end,
    rank_probability_candidates,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_path_aware_compound_cli_imports_outside_repository_root(tmp_path):
    script_path = REPOSITORY_ROOT / "scripts" / "path_aware_gbdt_compound_backtest.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout


def test_fixed_threshold_selects_only_executable_passing_candidates():
    probabilities = np.asarray([0.90, 0.99, 0.78, 0.79, np.nan, 0.70])
    executable = np.asarray([True, False, True, True, True, True])

    selected, accepted = rank_probability_candidates(
        probabilities, executable, threshold=0.78, top_k=2
    )

    assert selected.tolist() == [0, 3]
    assert accepted.tolist() == [True, False, True, True, False, False]


def test_probability_ranking_uses_stable_order_for_equal_scores():
    probabilities = np.asarray([0.80, 0.90, 0.90, 0.85])
    executable = np.asarray([True, True, True, True])

    selected, _ = rank_probability_candidates(
        probabilities, executable, threshold=0.80, top_k=2
    )

    assert selected.tolist() == [1, 2]


def test_probability_ranking_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="长度一致"):
        rank_probability_candidates(
            np.asarray([0.8, 0.9]),
            np.asarray([True]),
            threshold=0.78,
            top_k=2,
        )


def test_probability_ranking_rejects_invalid_threshold_and_top_k():
    probabilities = np.asarray([0.80])
    executable = np.asarray([True])

    with pytest.raises(ValueError, match="0 至 1"):
        rank_probability_candidates(probabilities, executable, threshold=1.1, top_k=1)
    with pytest.raises(ValueError, match="正整数"):
        rank_probability_candidates(probabilities, executable, threshold=0.78, top_k=0)



def test_final_signal_end_is_clamped_to_last_full_exit_window():
    dates = [f"2026-01-{day:02d}" for day in range(1, 11)]

    assert complete_signal_window_end(dates, dates[-1], max_forward_sessions=3) == dates[6]
    assert complete_signal_window_end(dates, dates[4], max_forward_sessions=3) == dates[4]


def test_final_signal_end_rejects_missing_nominal_date():
    dates = [f"2026-01-{day:02d}" for day in range(1, 11)]

    with pytest.raises(ValueError, match="不在交易日索引中"):
        complete_signal_window_end(dates, "2026-02-01", max_forward_sessions=3)
