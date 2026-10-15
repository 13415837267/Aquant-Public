from __future__ import annotations

import unittest

from scripts.path_aware_gbdt_compound_backtest import select_validation_exit_operating_point


def _row(
    threshold: float,
    target: float,
    stop: float,
    trades: int,
    compound: float,
    drawdown: float,
    mean_trade: float = 0.1,
) -> dict:
    return {
        "status": "ready",
        "probability_threshold": threshold,
        "net_target_pct": target,
        "stop_loss_pct": stop,
        "completed_trades": trades,
        "compound_return_pct": compound,
        "annualized_compound_return_pct": compound / 2.0,
        "max_drawdown_pct": drawdown,
        "mean_trade_net_return_pct": mean_trade,
    }


class ValidationExitParameterSelectionTests(unittest.TestCase):
    def test_selects_highest_return_only_from_eligible_validation_rows(self):
        rows = [
            _row(0.65, 1.5, 2.0, 120, 4.0, -12.0),
            _row(0.70, 3.0, 2.0, 140, 9.0, -16.0),
            _row(0.75, 2.0, 3.0, 99, 10.0, -5.0),
            _row(0.80, 4.0, 2.0, 130, 5.5, -10.0),
            _row(0.90, 2.0, 1.0, 150, 8.0, -7.0, mean_trade=-0.01),
        ]

        selected = select_validation_exit_operating_point(
            rows, minimum_trades=100, max_drawdown_floor_pct=-15.0
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected["probability_threshold"], 0.80)
        self.assertEqual(selected["net_target_pct"], 4.0)
        self.assertEqual(selected["stop_loss_pct"], 2.0)
        self.assertEqual(
            selected["selection_status"],
            "仅由验证期选择；联合网格存在多重比较偏差",
        )

    def test_returns_none_when_no_validation_row_meets_all_gates(self):
        rows = [
            _row(0.65, 1.0, 3.0, 150, -1.0, -5.0),
            _row(0.70, 2.0, 2.0, 80, 4.0, -5.0),
            _row(0.75, 3.0, 1.0, 120, 4.0, -18.0),
        ]
        self.assertIsNone(
            select_validation_exit_operating_point(
                rows, minimum_trades=100, max_drawdown_floor_pct=-15.0
            )
        )

    def test_rejects_nonfinite_metrics_and_invalid_parameters(self):
        rows = [
            _row(0.65, 1.0, 3.0, 150, float("nan"), -5.0),
            _row(1.2, 2.0, 2.0, 150, 4.0, -5.0),
            _row(0.70, -1.0, 2.0, 150, 4.0, -5.0),
            _row(0.75, 2.0, 2.0, 150, 4.0, -5.0, mean_trade=float("nan")),
        ]
        self.assertIsNone(
            select_validation_exit_operating_point(
                rows, minimum_trades=100, max_drawdown_floor_pct=-15.0
            )
        )


if __name__ == "__main__":
    unittest.main()
