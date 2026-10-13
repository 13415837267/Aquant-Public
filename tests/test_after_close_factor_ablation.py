from __future__ import annotations

import unittest

from scripts.after_close_factor_ablation import select_ablation_candidates


def pair(label: str, rules: list[dict], *, samples: int, wilson: float, rate: float) -> dict:
    return {
        "rule_type": "pair",
        "rule_label": label,
        "rules": rules,
        "samples": samples,
        "wilson_lower_pct": wilson,
        "path_win_1pct_rate_pct": rate,
    }


def condition(feature: str, scale: str = "rank", threshold: float = 0.3) -> dict:
    return {"feature": feature, "op": "<=", "threshold": threshold, "scale": scale}


class FactorAblationSelectionTests(unittest.TestCase):
    def test_selects_pair_by_training_statistics_and_builds_matched_control(self):
        atr_lower = pair(
            "close-strength-plus-atr",
            [condition("close_strength", threshold=0.4), condition("atr_14_pct")],
            samples=20_000, wilson=60.0, rate=63.0,
        )
        range_higher = pair(
            "close-strength-plus-range",
            [condition("close_strength", threshold=0.4), condition("range_pct")],
            samples=25_000, wilson=62.0, rate=64.0,
        )
        overnight = pair(
            "close-strength-plus-overnight",
            [condition("close_strength", threshold=0.4), condition("overnight_1d_pct")],
            samples=18_000, wilson=59.0, rate=62.0,
        )
        breadth = pair(
            "close-strength-plus-market-breadth",
            [
                condition("close_strength", threshold=0.4),
                condition("market_breadth_pct", scale="pct", threshold=50.0),
            ],
            samples=21_000, wilson=61.0, rate=65.0,
        )
        data = {
            "status": "research_only",
            "train_top_pairs": [atr_lower, range_higher, overnight, breadth],
            "validation_top_rules": [{"rule_label": "validation-only-winner"}],
            "selected_rules": [{"rule_label": "final-only-winner"}],
        }

        selected = select_ablation_candidates(data)
        self.assertEqual(
            selected["ATR与日内振幅"]["training_pair_label"],
            "close-strength-plus-range",
        )
        self.assertEqual(
            selected["隔夜收益"]["training_pair_label"],
            "close-strength-plus-overnight",
        )
        self.assertEqual(
            selected["市场广度"]["training_pair_label"],
            "close-strength-plus-market-breadth",
        )
        self.assertEqual(
            selected["ATR与日内振幅"]["ablation"],
            [condition("close_strength", threshold=0.4)],
        )
        self.assertEqual(len(selected["ATR与日内振幅"]["with_factor"]), 2)
        self.assertNotIn("validation-only-winner", str(selected))

    def test_rejects_market_only_control_and_low_sample_pairs(self):
        data = {
            "status": "research_only",
            "train_top_pairs": [
                pair(
                    "market-only-control",
                    [
                        condition("market_median_return_pct", scale="pct", threshold=0.5),
                        condition("market_breadth_pct", scale="pct", threshold=50.0),
                    ],
                    samples=20_000, wilson=90.0, rate=92.0,
                ),
                pair(
                    "too-few-samples",
                    [condition("close_strength", threshold=0.4), condition("atr_14_pct")],
                    samples=9_999, wilson=95.0, rate=96.0,
                ),
            ],
        }
        self.assertEqual(select_ablation_candidates(data), {})

    def test_selects_reversal_breakout_and_volume_families_from_training_pairs(self):
        reversal = pair(
            "return1-plus-close-strength",
            [condition("return_1d_pct", threshold=0.2), condition("close_strength", threshold=0.3)],
            samples=22_000, wilson=65.0, rate=67.0,
        )
        breakout = pair(
            "breakout20-plus-close-strength",
            [condition("breakout_20d_pct", threshold=0.7), condition("close_strength", threshold=0.6)],
            samples=23_000, wilson=64.0, rate=66.0,
        )
        volume = pair(
            "volume-ratio-plus-close-strength",
            [condition("volume_ratio_5d", threshold=0.7), condition("close_strength", threshold=0.6)],
            samples=24_000, wilson=63.0, rate=65.0,
        )
        selected = select_ablation_candidates({
            "status": "research_only",
            "train_top_pairs": [],
            "train_factor_ablation_pairs": [
                {**reversal, "target_family": "短期反转"},
                {**breakout, "target_family": "趋势突破"},
                {**volume, "target_family": "量能过滤"},
            ],
        })
        self.assertEqual(selected["短期反转"]["training_pair_label"], "return1-plus-close-strength")
        self.assertEqual(selected["趋势突破"]["training_pair_label"], "breakout20-plus-close-strength")
        self.assertEqual(selected["量能过滤"]["training_pair_label"], "volume-ratio-plus-close-strength")
        self.assertEqual(selected["短期反转"]["ablation"], [condition("close_strength", threshold=0.3)])

    def test_requires_research_only_input_and_training_pairs(self):
        with self.assertRaises(ValueError):
            select_ablation_candidates({"status": "ready", "train_top_pairs": []})
        with self.assertRaises(ValueError):
            select_ablation_candidates({"status": "research_only"})


if __name__ == "__main__":
    unittest.main()
