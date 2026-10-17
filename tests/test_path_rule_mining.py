from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.path_rule_mining import (
    ALL_FEATURES,
    MARKET_FEATURES,
    RESEARCH_ABLATION_FAMILIES,
    RESEARCH_CONFIG,
    RULE_DEFS,
    STOCK_FEATURES,
    build_factor_ablation_candidate_pairs,
    max_signal_index,
    path_targets,
)
from scripts.selection_factor_catalog import market_thresholds


class RuleMiningSplitBoundaryTests(unittest.TestCase):
    def test_all_stock_and_market_factors_are_scanned_at_all_configured_thresholds(self):
        expected = len(STOCK_FEATURES) * 2 * len(RESEARCH_CONFIG["分位阈值"])
        for feature in MARKET_FEATURES:
            expected += len(market_thresholds(feature, RESEARCH_CONFIG)) * 2
        self.assertEqual(len(ALL_FEATURES), len(STOCK_FEATURES) + len(MARKET_FEATURES))
        self.assertEqual(len(RULE_DEFS), expected)
        self.assertGreaterEqual(len(STOCK_FEATURES), 50)
        self.assertEqual(len(MARKET_FEATURES), 6)

    def test_path_labels_use_executable_entry_and_realized_net_return(self):
        symbols = ["000001"]
        future = [
            pd.DataFrame([{"symbol":"000001","open":100.0,"high":130.0,"low":99.0,"close":110.0,
                           "high_limit":110.0,"is_paused":0}]),
            pd.DataFrame([{"symbol":"000001","open":100.0,"high":102.0,"low":99.0,"close":101.0,
                           "high_limit":111.0,"is_paused":0}]),
            pd.DataFrame([{"symbol":"000001","open":101.0,"high":103.0,"low":100.0,"close":102.0,
                           "high_limit":112.0,"is_paused":0}]),
            pd.DataFrame([{"symbol":"000001","open":102.0,"high":104.0,"low":101.0,"close":103.0,
                           "high_limit":113.0,"is_paused":0}]),
            pd.DataFrame([{"symbol":"000001","open":103.0,"high":105.0,"low":102.0,"close":104.0,
                           "high_limit":114.0,"is_paused":0}]),
        ]
        y, best, close, complete, mae, mfe, target_day, stop_day, path_net = path_targets(symbols, future)
        self.assertTrue(complete[0])
        self.assertTrue(y[0])
        self.assertEqual(target_day[0], 2)
        self.assertTrue(np.isclose(path_net[0], 1.0, atol=1e-8))
        self.assertTrue(np.isfinite(best[0]))
        self.assertTrue(np.isfinite(mfe[0]))

    def test_path_labels_fail_closed_when_entry_limit_is_missing_or_paused(self):
        symbols = ["000001"]
        future = [
            pd.DataFrame([{"symbol":"000001","open":100.0,"high":101.0,"low":99.0,"close":100.0,
                           "high_limit":np.nan,"is_paused":0}]),
            *[
                pd.DataFrame([{"symbol":"000001","open":100.0,"high":102.0,"low":99.0,"close":101.0,
                               "high_limit":110.0,"is_paused":0}])
                for _ in range(4)
            ],
        ]
        result = path_targets(symbols, future)
        self.assertFalse(result[3][0])
        future[0]["high_limit"] = 110.0
        future[0]["is_paused"] = 1
        result = path_targets(symbols, future)
        self.assertFalse(result[3][0])

    def test_family_balanced_ablation_candidates_cover_every_configured_family(self):
        atomic_rows = []
        index = 0
        for family, features in RESEARCH_ABLATION_FAMILIES.items():
            feature = sorted(features)[0]
            scale = "pct" if feature == "market_breadth_pct" else "rank"
            atomic_rows.append({
                "rule_indices": [index],
                "rules": [{
                    "feature": feature,
                    "op": "<=",
                    "threshold": 0.3 if scale == "rank" else 50.0,
                    "scale": scale,
                }],
                "samples": 20_000,
                "wilson_lower_pct": 60.0 + index,
                "path_win_1pct_rate_pct": 65.0 + index,
            })
            index += 1

        atomic_rows.append({
            "rule_indices": [index],
            "rules": [{
                "feature": "close_strength",
                "op": "<=",
                "threshold": 0.3,
                "scale": "rank",
            }],
            "samples": 30_000,
            "wilson_lower_pct": 75.0,
            "path_win_1pct_rate_pct": 78.0,
        })

        pairs, summary = build_factor_ablation_candidate_pairs(
            atomic_rows, target_limit=1, control_limit=1
        )
        self.assertEqual(
            {row["target_family"] for row in pairs},
            set(RESEARCH_ABLATION_FAMILIES),
        )
        self.assertEqual(set(summary), set(RESEARCH_ABLATION_FAMILIES))
        self.assertTrue(all(len(row["rule_indices"]) == 2 for row in pairs))
        self.assertTrue(all(row["rule_indices"][0] != row["rule_indices"][1] for row in pairs))
        self.assertTrue(all(summary[family]["pair_candidates"] >= 1 for family in RESEARCH_ABLATION_FAMILIES))

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
