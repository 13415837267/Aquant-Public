import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.high_precision_profit_mining import PairwiseRankingModel


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_pairwise_cli_imports_when_started_outside_repository_root(tmp_path):
    script_path = REPOSITORY_ROOT / "scripts" / "pairwise_rank_compound_backtest.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--model-result" in result.stdout


def test_pairwise_update_improves_positive_negative_ordering():
    x = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    y = np.asarray([1.0, 0.0], dtype=np.float64)
    model = PairwiseRankingModel(
        n_features=2,
        learning_rate=0.1,
        l2=0.0,
        max_positives_per_day=1,
        max_negatives_per_day=1,
        seed=7,
    )

    before = model.predict(x)[0] - model.predict(x)[1]
    pairs = model.update(x, y)
    after = model.predict(x)[0] - model.predict(x)[1]

    assert pairs == 1
    assert model.pair_updates == 1
    assert model.pairs_seen == 1
    assert after > before


def test_pairwise_update_skips_days_without_both_classes():
    model = PairwiseRankingModel(n_features=2, seed=7)
    x = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    assert model.update(x, np.asarray([1.0, 1.0])) == 0
    assert model.pair_updates == 0
    assert model.pairs_seen == 0
