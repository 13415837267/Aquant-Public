import gzip
import json

import numpy as np
import pandas as pd

from scripts.backtest import metrics, normalize_weights, turnover


def test_metrics_simple_path():
    daily = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05", "2026-01-06"],
            "net_return": [0.01, -0.005, 0.02],
            "turnover": [1.0, 0.2, 0.1],
        }
    )
    result = metrics(daily)

    expected = (1.01 * 0.995 * 1.02) - 1
    assert np.isclose(result["total_return_pct"], expected * 100)
    assert result["trading_days"] == 3
    assert result["average_turnover_pct"] > 0


def test_weights_are_equal_and_normalized():
    weights = normalize_weights(["000002", "000001", "000001"])
    assert weights == {"000001": 0.5, "000002": 0.5}
    assert np.isclose(sum(weights.values()), 1.0)


def test_turnover_is_l1_target_weight_change():
    prev = {"000001": 0.5, "000002": 0.5}
    target = {"000002": 0.5, "000003": 0.5}
    assert np.isclose(turnover(prev, target), 1.0)
