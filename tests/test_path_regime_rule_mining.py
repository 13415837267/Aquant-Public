from __future__ import annotations

import numpy as np

from scripts.path_regime_rule_mining import (
    int_dot,
    int_vector_dot,
    wilson_lower_bound,
)


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
