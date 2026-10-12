from __future__ import annotations

import unittest

from scripts.after_close_strategy_report import build_report, summarize_model, summarize_rules


def rule_report(
    validation_rate: float = 81.0,
    validation_samples: int = 6000,
    validation_close: float = 0.4,
    final_rate: float = 72.0,
) -> dict:
    row = {
        "rule_type": "pair",
        "rule_label": "market_median_return_pct>=0.5 AND volatility_10d_pct<=0.7",
        "train": {
            "samples": 10000,
            "path_win_1pct_rate_pct": 75.0,
            "wilson_lower_pct": 74.0,
            "mean_best_return_pct": 4.0,
            "mean_5d_close_return_pct": 0.3,
            "mean_mae_pct": -3.0,
        },
        "validation": {
            "samples": validation_samples,
            "path_win_1pct_rate_pct": validation_rate,
            "wilson_lower_pct": validation_rate - 1.0,
            "mean_best_return_pct": 4.5,
            "mean_5d_close_return_pct": validation_close,
            "mean_mae_pct": -2.8,
        },
        "final": {
            "samples": 7000,
            "path_win_1pct_rate_pct": final_rate,
            "wilson_lower_pct": final_rate - 1.0,
            "mean_best_return_pct": 3.5,
            "mean_5d_close_return_pct": 0.1,
            "mean_mae_pct": -3.1,
        },
    }
    return {
        "schema_version": 1,
        "status": "research_only",
        "method": "path_aware_conditional_rule_mining",
        "splits": {
            "train": ["2015-01-05", "2022-12-30"],
            "validation": ["2023-01-03", "2024-12-31"],
            "final": ["2025-01-02", "2026-09-30"],
        },
        "base": {
            "train": {"samples": 100000, "path_win_1pct_rate_pct": 65.0},
            "validation": {"samples": 20000, "path_win_1pct_rate_pct": 69.0},
            "final": {"samples": 21000, "path_win_1pct_rate_pct": 68.0},
        },
        "selected_rules": [row],
    }


def model_report(
    validation_trades: int = 150,
    validation_return: float = 8.0,
    validation_drawdown: float = -18.0,
) -> dict:
    return {
        "schema_version": 2,
        "status": "research_only",
        "method": "path_aware_gbdt_validation_compound_threshold_sweep",
        "model": {"type": "HistGradientBoostingClassifier", "training_samples": 100000},
        "selection_policy": {
            "minimum_completed_trades": 100,
            "maximum_drawdown_floor_pct": -30.0,
        },
        "splits": {
            "validation": ["2023-01-03", "2024-12-20"],
            "final_holdout": ["2025-01-02", "2026-09-23"],
        },
        "validation": {
            "selected_threshold": {
                "probability_threshold": 0.8,
                "completed_trades": validation_trades,
                "compound_return_pct": validation_return,
                "annualized_compound_return_pct": validation_return / 2.0,
                "max_drawdown_pct": validation_drawdown,
                "selection_status": "selected_on_validation",
            },
            "threshold_sweep": [
                {
                    "probability_threshold": 0.8,
                    "completed_trades": validation_trades,
                    "compound_return_pct": validation_return,
                    "annualized_compound_return_pct": validation_return / 2.0,
                    "max_drawdown_pct": validation_drawdown,
                    "status": "ok",
                }
            ],
        },
        "final_holdout": {
            "portfolio": {
                "status": "supplemental",
                "completed_trades": 120,
                "compound_return_pct": 2.0,
                "annualized_compound_return_pct": 1.0,
                "max_drawdown_pct": -22.0,
                "final_equity": 102000.0,
            },
            "fresh_unseen_holdout": False,
        },
        "audit": {"formal_production_changed": False},
    }


class AfterCloseStrategyReportTests(unittest.TestCase):
    def test_rule_qualification_uses_validation_only(self):
        result = summarize_rules(rule_report(validation_rate=81.0, final_rate=60.0))
        self.assertTrue(result["validation_qualified_for_next_stage"])
        self.assertEqual(
            result["best_validation_rule"]["rule"],
            "market_median_return_pct>=0.5 AND volatility_10d_pct<=0.7",
        )
        self.assertEqual(result["supplemental_final_best_rule_vs_base_hit_rate_delta_pp"], -8.0)
        self.assertIn("不是每日TopN组合收益", result["important_limit"])

    def test_rule_does_not_qualify_with_small_sample_or_negative_close_return(self):
        low_sample = summarize_rules(rule_report(validation_rate=90.0, validation_samples=4999))
        negative_return = summarize_rules(rule_report(validation_rate=90.0, validation_close=-0.2))
        self.assertFalse(low_sample["validation_qualified_for_next_stage"])
        self.assertFalse(negative_return["validation_qualified_for_next_stage"])

    def test_model_gate_requires_enough_trades_positive_returns_and_drawdown(self):
        self.assertTrue(summarize_model(model_report())["validation_release_gate_passed"])
        self.assertFalse(
            summarize_model(model_report(validation_trades=99))["validation_release_gate_passed"]
        )
        self.assertFalse(
            summarize_model(model_report(validation_return=-1.0))["validation_release_gate_passed"]
        )
        self.assertFalse(
            summarize_model(model_report(validation_drawdown=-35.0))["validation_release_gate_passed"]
        )

    def test_combined_report_never_promotes_or_changes_production(self):
        report = build_report(rule_report(), model_report())
        self.assertEqual(report["status"], "research_only")
        self.assertFalse(report["decision"]["formal_production_changed"])
        self.assertFalse(report["decision"]["promotion_allowed"])
        self.assertFalse(report["model_strategy"]["final_supplemental_is_fresh_unseen_holdout"])
        self.assertIn("口径不同", report["decision"]["reason"])

    def test_incompatible_methods_are_rejected(self):
        rules = rule_report()
        rules["method"] = "unrecognized"
        with self.assertRaises(ValueError):
            summarize_rules(rules)


if __name__ == "__main__":
    unittest.main()
