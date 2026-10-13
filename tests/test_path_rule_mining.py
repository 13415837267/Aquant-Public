from __future__ import annotations

import unittest

from scripts.path_rule_mining import (
    RESEARCH_ABLATION_FAMILIES,
    build_factor_ablation_candidate_pairs,
    max_signal_index,
)


class RuleMiningSplitBoundaryTests(unittest.TestCase):
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
