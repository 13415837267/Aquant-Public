from __future__ import annotations

import unittest

import numpy as np

from scripts.after_close_rule_compound_backtest import (
    evaluate_rule_conditions,
    select_training_rule,
)
from scripts.path_rule_mining import FEATURE_INDEX


def candidate(
    label: str,
    rules: list[dict],
    *,
    samples: int = 20000,
    rate: float = 75.0,
    wilson: float = 74.0,
    mean_net: float = 0.15,
    profit_factor: float = 1.15,
) -> dict:
    return {
        "rule_type": "pair" if len(rules) > 1 else "atomic",
        "rule_label": label,
        "rules": rules,
        "samples": samples,
        "path_win_1pct_rate_pct": rate,
        "wilson_lower_pct": wilson,
        "mean_path_net_return_pct": mean_net,
        "path_profit_factor": profit_factor,
        "resolved_net_return_samples": samples,
    }


class AfterCloseRulePortfolioTests(unittest.TestCase):
    def test_rule_selection_prioritizes_path_net_return_over_hit_rate(self):
        high_hit_rule = candidate(
            "high-hit-low-return",
            [{"feature":"atr_14_pct","op":"<=","threshold":0.3,"scale":"rank"}],
            rate=82.0, wilson=81.0, mean_net=-0.2, profit_factor=0.7,
        )
        positive_return_rule = candidate(
            "positive-net-return",
            [{"feature":"return_20d_pct","op":">=","threshold":0.7,"scale":"rank"}],
            rate=63.0, wilson=62.0, mean_net=0.3, profit_factor=1.4,
        )
        result = select_training_rule({
            "status":"research_only",
            "train_top_atomic":[high_hit_rule, positive_return_rule],
            "train_top_pairs":[],
        })
        self.assertEqual(result["rule_label"], "positive-net-return")
        self.assertTrue(result["training_edge_gate_passed"])
        result = select_training_rule({
            "status":"research_only",
            "train_top_atomic":[high_hit_rule],
            "train_top_pairs":[],
        })
        self.assertFalse(result["training_edge_gate_passed"])

    def test_rule_selection_requires_enough_resolved_path_samples(self):
        incomplete = candidate(
            "too-few-resolved",
            [{"feature":"return_20d_pct","op":">=","threshold":0.7,"scale":"rank"}],
            samples=20000, mean_net=0.4, profit_factor=1.5,
        )
        incomplete["resolved_net_return_samples"] = 9999
        with self.assertRaises(ValueError):
            select_training_rule({
                "status":"research_only",
                "train_top_atomic":[incomplete],
                "train_top_pairs":[],
            })

    def test_rule_selection_uses_training_statistics_and_requires_stock_factor(self):
        market_only = candidate(
            "market-only",
            [{"feature": "market_median_return_pct", "op": ">=", "threshold": 0.5, "scale": "pct"}],
            wilson=95.0,
            rate=96.0,
        )
        train_candidate = candidate(
            "market-and-atr",
            [
                {"feature": "market_median_return_pct", "op": ">=", "threshold": 0.5, "scale": "pct"},
                {"feature": "atr_14_pct", "op": "<=", "threshold": 0.3, "scale": "rank"},
            ],
            wilson=70.0,
            rate=72.0,
        )
        result = select_training_rule({
            "status": "research_only",
            "train_top_atomic": [],
            "train_top_pairs": [market_only, train_candidate],
            "selected_rules": [
                {"rule_label": "validation-best-should-not-be-used"},
            ],
        })
        self.assertEqual(result["rule_label"], "market-and-atr")

    def test_rule_mask_and_rank_score_use_stock_factor_margin(self):
        x = np.zeros((3, len(FEATURE_INDEX)), dtype=float)
        x[:, FEATURE_INDEX["market_median_return_pct"]] = 1.2
        x[:, FEATURE_INDEX["atr_14_pct"]] = [0.10, 0.20, 0.25]
        conditions = [
            {"feature": "market_median_return_pct", "op": ">=", "threshold": 1.0, "scale": "pct"},
            {"feature": "atr_14_pct", "op": "<=", "threshold": 0.30, "scale": "rank"},
        ]
        matched, score = evaluate_rule_conditions(x, conditions)
        self.assertTrue(matched.all())
        self.assertGreater(score[0], score[1])
        self.assertGreater(score[1], score[2])

    def test_unmatched_stocks_do_not_pass_rule(self):
        x = np.zeros((2, len(FEATURE_INDEX)), dtype=float)
        x[:, FEATURE_INDEX["market_median_return_pct"]] = [1.2, 0.8]
        x[:, FEATURE_INDEX["atr_14_pct"]] = [0.1, 0.1]
        conditions = [
            {"feature": "market_median_return_pct", "op": ">=", "threshold": 1.0, "scale": "pct"},
            {"feature": "atr_14_pct", "op": "<=", "threshold": 0.3, "scale": "rank"},
        ]
        matched, _ = evaluate_rule_conditions(x, conditions)
        self.assertEqual(matched.tolist(), [True, False])

    def test_empty_or_market_only_training_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            select_training_rule({
                "train_top_atomic": [],
                "train_top_pairs": [],
            })
        with self.assertRaises(ValueError):
            select_training_rule({
                "train_top_atomic": [],
                "train_top_pairs": [
                    candidate("market-only", [
                        {"feature": "market_median_return_pct", "op": ">=", "threshold": 0.5, "scale": "pct"}
                    ]),
                ],
            })


if __name__ == "__main__":
    unittest.main()
