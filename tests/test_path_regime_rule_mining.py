from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.path_regime_rule_mining import (
    int_dot,
    int_vector_dot,
    path_targets as regime_path_targets,
    wilson_lower_bound,
)
from scripts.path_rule_mining import path_targets as rule_path_targets


def test_integer_matrix_products_use_transposed_left_operand_without_wrap():
    left = np.ones((1000, 2), dtype=np.uint8)
    right = np.ones((1000, 1), dtype=np.uint8)
    result = int_dot(left, right)
    assert result.shape == (2, 1)
    assert result[:, 0].tolist() == [1000, 1000]


def test_integer_vector_products_do_not_wrap_uint8():
    left = np.ones(1000, dtype=np.uint8)
    right = np.ones(1000, dtype=np.uint8)
    assert int_vector_dot(left, right) == 1000


def test_wilson_lower_bound_accepts_large_valid_counts():
    lower = wilson_lower_bound(700_000, 1_000_000)
    assert 0.0 < lower < 1.0



def _future_days_with_target_and_stop_conflict():
    symbols = ["000001", "000002"]
    future_days = []
    for day_index in range(5):
        rows = []
        for symbol in symbols:
            row = {
                "symbol": symbol,
                "open": 100.0,
                "high": 100.5,
                "low": 99.0,
                "close": 100.0,
            }
            if day_index == 0:
                # T+1 是买入日，低点触及止损也不得当日卖出。
                row["low"] = 95.0
            elif day_index == 1 and symbol == "000001":
                # T+2 先满足止盈，且未触及止损。
                row["high"] = 102.0
                row["low"] = 98.0
            elif day_index == 1 and symbol == "000002":
                # T+2 同时满足止盈与止损，必须保守地止损优先。
                row["high"] = 103.0
                row["low"] = 96.0
            rows.append(row)
        future_days.append(pd.DataFrame(rows))
    return symbols, future_days


@pytest.mark.parametrize(
    "target_builder",
    [rule_path_targets, regime_path_targets],
    ids=["单因子条件挖掘", "市场状态条件挖掘"],
)
def test_path_targets_use_scalar_prices_and_respect_t_plus_one(target_builder):
    symbols, future_days = _future_days_with_target_and_stop_conflict()

    win, _, _, complete, _, _, target_day, stop_day = target_builder(
        symbols, future_days
    )

    assert complete.tolist() == [True, True]
    assert win.tolist() == [True, False]
    assert target_day[0] == 2
    assert stop_day[1] == 2
