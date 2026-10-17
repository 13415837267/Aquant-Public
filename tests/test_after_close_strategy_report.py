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
            "top_k_per_signal_day": 2,
            "initial_equity": 100000.0,
            "net_target_pct": 1.0,
            "stop_loss_pct": 3.0,
            "round_trip_cost_bps": 10.0,
            "cash_constrained_no_leverage": True,
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
                "target_hit_rate_pct": 72.5,
                "positive_trade_rate_pct": 74.0,
                "mean_trade_net_return_pct": 0.29,
                "sharpe_ratio": 0.68,
                "selection_policy": {
                    "signal": "T日收盘",
                    "entry": "T+1开盘",
                    "earliest_exit": "T+2",
                    "maximum_holding_sessions": 5,
                },
            },
            "fresh_unseen_holdout": False,
        },
        "audit": {"formal_production_changed": False},
    }


def rule_portfolio_report(
    validation_trades: int = 120,
    validation_return: float = 4.0,
    validation_drawdown: float = -12.0,
    final_return: float = 2.5,
) -> dict:
    validation = {
        "status": "ready",
        "completed_trades": validation_trades,
        "compound_return_pct": validation_return,
        "annualized_compound_return_pct": validation_return / 2.0,
        "max_drawdown_pct": validation_drawdown,
        "target_hit_rate_pct": 73.0,
        "final_equity": 104000.0,
        "measurement_window": ["2023-01-03", "2024-12-24"],
        "signal_window": ["2023-01-03", "2024-12-17"],
    }
    final = {
        "status": "ready",
        "completed_trades": 100,
        "compound_return_pct": final_return,
        "annualized_compound_return_pct": final_return / 1.8,
        "max_drawdown_pct": -8.0,
        "target_hit_rate_pct": 72.0,
        "final_equity": 102500.0,
        "measurement_window": ["2025-01-02", "2026-10-09"],
        "signal_window": ["2025-01-02", "2026-09-24"],
    }
    return {
        "schema_version": 1,
        "status": "research_only",
        "method": "training_frozen_rule_cash_constrained_portfolio",
        "training_selected_rule": {
            "rule_type": "pair",
            "rule_label": "market_median_return_pct>=0.5 AND atr_14_pct<=0.3",
            "rules": [
                {"feature": "market_median_return_pct", "op": ">=", "threshold": 0.5, "scale": "pct"},
                {"feature": "atr_14_pct", "op": "<=", "threshold": 0.3, "scale": "rank"},
            ],
            "training_samples": 20000,
            "training_path_win_1pct_rate_pct": 73.0,
            "training_wilson_lower_pct": 72.0,
            "training_mean_path_net_return_pct": 0.2,
            "training_path_profit_factor": 1.2,
            "training_edge_gate_passed": True,
            "selection_method": "training only",
        },
        "selection_policy": {
            "top_k_per_signal_day": 2,
            "initial_equity": 100000.0,
            "net_target_pct": 1.0,
            "stop_loss_pct": 3.0,
            "round_trip_cost_bps": 10.0,
            "maximum_holding_sessions": 5,
            "cash_constrained_no_leverage": True,
        },
        "validation_portfolio": validation,
        "final_supplemental_portfolio": final,
        "audit": {
            "rule_selected_from_training_only": True,
            "validation_used_only_to_evaluate_frozen_rule": True,
            "final_window_is_not_fresh_unseen_holdout": True,
            "formal_production_changed": False,
        },
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
        report = build_report(rule_report(), model_report(), rule_portfolio_report())
        self.assertEqual(report["status"], "research_only")
        self.assertFalse(report["decision"]["formal_production_changed"])
        self.assertFalse(report["decision"]["promotion_allowed"])
        self.assertFalse(report["model_strategy"]["final_supplemental_is_fresh_unseen_holdout"])
        self.assertIn("选择偏差", report["decision"]["reason"])
        self.assertTrue(report["same_execution_mechanics"])
        self.assertTrue(report["rule_strategy"]["validation_release_gate_passed"])
        self.assertFalse(report["portfolio_comparison"]["final_supplemental"]["fresh_unseen_holdout"])
        self.assertEqual(
            report["portfolio_comparison"]["final_supplemental"]["model"]["target_hit_rate_pct"],
            72.5,
        )

    def test_frozen_rule_portfolio_gate_rejects_missing_training_edge(self):
        from scripts.after_close_strategy_report import summarize_rule_portfolio
        data = rule_portfolio_report()
        data["training_selected_rule"]["training_edge_gate_passed"] = False
        result = summarize_rule_portfolio(data)
        self.assertFalse(result["training_edge_gate_passed"])
        self.assertFalse(result["validation_release_gate_passed"])

    def test_frozen_rule_portfolio_gate_rejects_negative_or_small_results(self):
        from scripts.after_close_strategy_report import summarize_rule_portfolio
        self.assertTrue(summarize_rule_portfolio(rule_portfolio_report())["validation_release_gate_passed"])
        self.assertFalse(summarize_rule_portfolio(rule_portfolio_report(validation_trades=99))["validation_release_gate_passed"])
        self.assertFalse(summarize_rule_portfolio(rule_portfolio_report(validation_return=-1.0))["validation_release_gate_passed"])
        self.assertFalse(summarize_rule_portfolio(rule_portfolio_report(validation_drawdown=-35.0))["validation_release_gate_passed"])

    def test_incompatible_methods_are_rejected(self):
        rules = rule_report()
        rules["method"] = "unrecognized"
        with self.assertRaises(ValueError):
            summarize_rules(rules)


if __name__ == "__main__":
    unittest.main()
