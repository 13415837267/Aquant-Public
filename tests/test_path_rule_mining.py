from __future__ import annotations

import unittest

from scripts.path_rule_mining import max_signal_index


class RuleMiningSplitBoundaryTests(unittest.TestCase):
    def test_training_and_validation_paths_cannot_cross_split_end(self):
        end_index = 100
        horizon = 5
        last_signal = max_signal_index(
            end_index, total_sessions=250, horizon_sessions=horizon
        )
        self.assertEqual(last_signal, end_index - horizon)
        self.assertLessEqual(last_signal + horizon, end_index)

    def test_final_window_may_use_future_bars_after_nominal_signal_end(self):
        end_index = 100
        last_signal = max_signal_index(
            end_index, total_sessions=250, horizon_sessions=5,
            allow_forward_after_end=True,
        )
        self.assertEqual(last_signal, end_index)

    def test_data_end_always_limits_signals_to_complete_exit_window(self):
        last_signal = max_signal_index(
            end_index=245, total_sessions=250, horizon_sessions=5,
            allow_forward_after_end=True,
        )
        self.assertEqual(last_signal, 244)
        self.assertLess(last_signal + 5, 250)

    def test_invalid_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            max_signal_index(10, total_sessions=50, horizon_sessions=0)


if __name__ == "__main__":
    unittest.main()
