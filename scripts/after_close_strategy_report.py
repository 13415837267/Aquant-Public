"""汇总盘后规则挖掘与可执行模型组合回测；仅供研究，不改变正式策略。"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/backtest/after_close_strategy_comparison_latest.json"
MIN_RULE_SAMPLES = 5_000
MIN_PORTFOLIO_TRADES = 100
MAX_DRAWDOWN_FLOOR_PCT = -30.0
RULE_DISPLAY_LIMIT = 5


def _number(value: Any) -> float | None:
    """Convert JSON numeric values safely, including NaN emitted by older reports."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    result = _number(value)
    return int(result) if result is not None else None


def _clean_json(value: Any) -> Any:
    """Replace non-finite floating-point values before strict JSON serialization."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    return value


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label}结果无法读取：{path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("status") != "research_only":
        raise ValueError(f"{label}结果必须为 research_only 格式：{path}")
    return data


def _rule_summary(rule: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rule": rule.get("rule_label"),
        "rule_type": rule.get("rule_type"),
        "train": {},
        "validation": {},
        "final_supplemental": {},
    }
    aliases = {
        "samples": "samples",
        "path_win_1pct_rate_pct": "strict_net_1pct_win_rate_pct",
        "wilson_lower_pct": "wilson_lower_pct",
        "mean_best_return_pct": "mean_best_return_pct",
        "mean_5d_close_return_pct": "mean_5d_close_return_pct",
        "mean_mae_pct": "mean_mae_pct",
    }
    for source_key, dest_key in aliases.items():
        for split_key, out_key in (
            ("train", "train"),
            ("validation", "validation"),
            ("final", "final_supplemental"),
        ):
            split = rule.get(split_key) or {}
            result[out_key][dest_key] = (
                _integer(split.get(source_key))
                if source_key == "samples"
                else _number(split.get(source_key))
            )
    return result


def summarize_rules(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("method") != "path_aware_conditional_rule_mining":
        raise ValueError("规则结果的方法标识不匹配。")
    base = data.get("base") or {}
    selected = data.get("selected_rules") or []
    if not isinstance(selected, list):
        raise ValueError("selected_rules 必须为数组。")

    eligible = []
    for rule in selected:
        if not isinstance(rule, dict):
            continue
        validation = rule.get("validation") or {}
        samples = _integer(validation.get("samples")) or 0
        rate = _number(validation.get("path_win_1pct_rate_pct"))
        wilson = _number(validation.get("wilson_lower_pct"))
        if samples >= MIN_RULE_SAMPLES and rate is not None:
            eligible.append((rate, wilson if wilson is not None else -1.0, samples, rule))
    eligible.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)

    displayed = [_rule_summary(item[3]) for item in eligible[:RULE_DISPLAY_LIMIT]]
    best = eligible[0][3] if eligible else None
    base_validation = base.get("validation") or {}
    base_final = base.get("final") or {}
    best_validation = (best or {}).get("validation") or {}
    best_final = (best or {}).get("final") or {}
    validation_rate = _number(best_validation.get("path_win_1pct_rate_pct"))
    final_rate = _number(best_final.get("path_win_1pct_rate_pct"))
    validation_base_rate = _number(base_validation.get("path_win_1pct_rate_pct"))
    final_base_rate = _number(base_final.get("path_win_1pct_rate_pct"))

    # The historical final period has already been examined in earlier experiments.
    # Treat it as supplemental evidence, never as a fresh untouched holdout.
    validation_qualified = (
        best is not None
        and (_integer(best_validation.get("samples")) or 0) >= MIN_RULE_SAMPLES
        and validation_rate is not None
        and validation_rate >= 80.0
        and _number(best_validation.get("mean_5d_close_return_pct")) is not None
        and _number(best_validation.get("mean_5d_close_return_pct")) > 0
    )
    return {
        "status": "research_only",
        "method": data.get("method"),
        "splits": data.get("splits"),
        "base": {
            "train": base.get("train"),
            "validation": base.get("validation"),
            "final_supplemental": base.get("final"),
        },
        "best_validation_rule": _rule_summary(best) if best else None,
        "top_validation_rules": displayed,
        "selected_rule_count": len(selected),
        "validation_best_rule_vs_base_hit_rate_delta_pp": (
            round(validation_rate - validation_base_rate, 4)
            if validation_rate is not None and validation_base_rate is not None else None
        ),
        "supplemental_final_best_rule_vs_base_hit_rate_delta_pp": (
            round(final_rate - final_base_rate, 4)
            if final_rate is not None and final_base_rate is not None else None
        ),
        "validation_qualified_for_next_stage": bool(validation_qualified),
        "important_limit": (
            "规则命中率是股票级别条件统计，不是每日TopN组合收益；不得与模型组合复利直接横向排名。"
        ),
    }


def summarize_model(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("method") != "path_aware_gbdt_validation_compound_threshold_sweep":
        raise ValueError("模型组合结果的方法标识不匹配。")
    validation = data.get("validation") or {}
    final = data.get("final_holdout") or {}
    selected = validation.get("selected_threshold")
    selected = selected if isinstance(selected, dict) else None
    final_portfolio = final.get("portfolio")
    final_portfolio = final_portfolio if isinstance(final_portfolio, dict) else {}
    policy = data.get("selection_policy") or {}
    min_trades = _integer(policy.get("minimum_completed_trades")) or MIN_PORTFOLIO_TRADES
    drawdown_floor = _number(policy.get("maximum_drawdown_floor_pct"))
    if drawdown_floor is None:
        drawdown_floor = MAX_DRAWDOWN_FLOOR_PCT

    validation_return = _number((selected or {}).get("compound_return_pct"))
    validation_drawdown = _number((selected or {}).get("max_drawdown_pct"))
    validation_trades = _integer((selected or {}).get("completed_trades")) or 0
    validation_gate = bool(
        selected is not None
        and validation_trades >= min_trades
        and validation_return is not None
        and validation_return > 0
        and _number((selected or {}).get("annualized_compound_return_pct")) is not None
        and float((selected or {}).get("annualized_compound_return_pct")) > 0
        and validation_drawdown is not None
        and validation_drawdown >= drawdown_floor
    )

    final_summary = {
        key: final_portfolio.get(key)
        for key in (
            "status",
            "completed_trades",
            "compound_return_pct",
            "annualized_compound_return_pct",
            "max_drawdown_pct",
            "win_rate_pct",
            "profit_factor",
            "initial_equity",
            "final_equity",
            "selection_status",
        )
        if key in final_portfolio
    }
    selected_summary = None
    if selected is not None:
        selected_summary = {
            key: selected.get(key)
            for key in (
                "probability_threshold",
                "completed_trades",
                "compound_return_pct",
                "annualized_compound_return_pct",
                "max_drawdown_pct",
                "selection_status",
            )
            if key in selected
        }
    return {
        "status": "research_only",
        "method": data.get("method"),
        "model": data.get("model"),
        "splits": data.get("splits"),
        "validation_selected_operating_point": selected_summary,
        "validation_release_gate_passed": validation_gate,
        "final_supplemental_portfolio": final_summary,
        "final_supplemental_is_fresh_unseen_holdout": False,
        "validation_threshold_sweep": [
            {
                key: row.get(key)
                for key in (
                    "probability_threshold",
                    "completed_trades",
                    "compound_return_pct",
                    "annualized_compound_return_pct",
                    "max_drawdown_pct",
                    "status",
                )
                if key in row
            }
            for row in validation.get("threshold_sweep", [])
            if isinstance(row, dict)
        ],
        "formal_production_changed": False,
    }


def summarize_rule_portfolio(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("method") != "training_frozen_rule_cash_constrained_portfolio":
        raise ValueError("规则组合结果的方法标识不匹配")
    rule = data.get("training_selected_rule") or {}
    validation = data.get("validation_portfolio") or {}
    final = data.get("final_supplemental_portfolio") or {}
    policy = data.get("selection_policy") or {}
    minimum_trades = _integer(policy.get("minimum_completed_trades")) or MIN_PORTFOLIO_TRADES

    validation_return = _number(validation.get("compound_return_pct"))
    validation_annualized = _number(validation.get("annualized_compound_return_pct"))
    validation_drawdown = _number(validation.get("max_drawdown_pct"))
    validation_trades = _integer(validation.get("completed_trades")) or 0
    validation_gate = bool(
        validation.get("status") == "ready"
        and validation_trades >= minimum_trades
        and validation_return is not None
        and validation_return > 0
        and validation_annualized is not None
        and validation_annualized > 0
        and validation_drawdown is not None
        and validation_drawdown >= MAX_DRAWDOWN_FLOOR_PCT
    )

    keys = (
        "status", "requested_entries", "entered_trades", "completed_trades",
        "compound_return_pct", "annualized_compound_return_pct", "max_drawdown_pct",
        "target_hit_rate_pct", "positive_trade_rate_pct", "mean_trade_net_return_pct",
        "final_equity", "sharpe_ratio", "open_positions_at_end", "skipped_for_capacity_or_cash",
        "measurement_window", "signal_window", "selection_status",
    )
    validation_summary = {key: validation.get(key) for key in keys if key in validation}
    final_summary = {key: final.get(key) for key in keys if key in final}
    return {
        "status": "research_only",
        "method": data.get("method"),
        "training_selected_rule": {
            key: rule.get(key)
            for key in (
                "rule_type", "rule_label", "rules", "training_samples",
                "training_path_win_1pct_rate_pct", "training_wilson_lower_pct", "selection_method",
            )
            if key in rule
        },
        "rule_frozen_before_validation": bool((data.get("audit") or {}).get("rule_selected_from_training_only")),
        "validation_portfolio": validation_summary,
        "validation_release_gate_passed": validation_gate,
        "final_supplemental_portfolio": final_summary,
        "final_supplemental_is_fresh_unseen_holdout": False,
        "audit": data.get("audit") or {},
    }


def _comparison_metrics(portfolio: dict[str, Any] | None) -> dict[str, Any]:
    portfolio = portfolio if isinstance(portfolio, dict) else {}
    keys = (
        "status", "completed_trades", "compound_return_pct", "annualized_compound_return_pct",
        "max_drawdown_pct", "target_hit_rate_pct", "positive_trade_rate_pct",
        "mean_trade_net_return_pct", "final_equity", "sharpe_ratio",
    )
    return {key: portfolio.get(key) for key in keys if key in portfolio}


def build_report(
    rules: dict[str, Any],
    model: dict[str, Any],
    rule_portfolio: dict[str, Any],
) -> dict[str, Any]:
    rule_mining = summarize_rules(rules)
    frozen_rule = summarize_rule_portfolio(rule_portfolio)
    model_summary = summarize_model(model)
    model_validation = model_summary.get("validation_selected_operating_point") or {}
    model_final = model_summary.get("final_supplemental_portfolio") or {}
    frozen_validation = frozen_rule.get("validation_portfolio") or {}
    frozen_final = frozen_rule.get("final_supplemental_portfolio") or {}
    policy = rule_portfolio.get("selection_policy") or {}
    model_policy = model.get("selection_policy") or {}

    same_mechanics = bool(
        int(policy.get("top_k_per_signal_day", -1)) == int(model_policy.get("top_k_per_signal_day", -2))
        and _number(policy.get("net_target_pct")) == _number(model_policy.get("net_target_pct"))
        and _number(policy.get("stop_loss_pct")) == _number(model_policy.get("stop_loss_pct"))
        and _number(policy.get("round_trip_cost_bps")) == _number(model_policy.get("round_trip_cost_bps"))
        and _integer(policy.get("maximum_holding_sessions")) == _integer(model_policy.get("maximum_holding_sessions"))
        and policy.get("cash_constrained_no_leverage") is True
        and model_policy.get("cash_constrained_no_leverage") is True
    )
    return {
        "schema_version": 2,
        "status": "research_only",
        "title": "盘后选股策略：同口径规则与模型组合对照",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": {
            "signal": "T日收盘后",
            "entry": "T+1开盘",
            "earliest_exit": "T+2",
            "maximum_holding_sessions": 5,
            "target_net_profit_pct": 1.0,
            "stop_loss_pct": 3.0,
            "round_trip_cost_bps": 10.0,
            "top_k_per_signal_day": 2,
            "cash_constrained_no_leverage": True,
            "limit_up_or_paused_entry_blocked": True,
        },
        "rule_mining": rule_mining,
        "rule_strategy": frozen_rule,
        "model_strategy": model_summary,
        "same_execution_mechanics": same_mechanics,
        "portfolio_comparison": {
            "validation": {
                "rule": _comparison_metrics(frozen_validation),
                "model": _comparison_metrics(model_validation),
                "model_validation_is_used_for_threshold_selection": True,
                "rule_validation_is_used_only_for_evaluation": True,
                "note": "模型验证收益所对应的概率阈值由同一验证期选择，具有参数选择乐观偏差；规则条件在训练期冻结。",
            },
            "final_supplemental": {
                "rule": _comparison_metrics(frozen_final),
                "model": _comparison_metrics(model_final),
                "fresh_unseen_holdout": False,
                "note": "双方使用相同信号区间、Top2、目标收益、止损、成本和现金约束；最终区间已被此前研究查看，只能作为补充对照。",
            },
        },
        "decision": {
            "formal_production_changed": False,
            "promotion_allowed": False,
            "reason": (
                "当前仍未达到正式策略80%命中率门槛；最终历史区间并非全新留出。"
                "本次将训练期冻结规则与模型放在同一Top2现金约束组合口径下比较，"
                "但模型阈值依据验证期选出，其验证收益存在选择偏差。"
                "研究结果不得自动替换正式模型或启用自动下单。"
            ),
            "next_step": (
                "优先比较冻结规则与模型在最终补充期的交易样本、复利、最大回撤和年度稳定性；"
                "确认胜负后冻结唯一候选版本，启动不改参数的前瞻模拟记录，"
                "积累真正未参与选参的新交易数据后再审核正式晋级。"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总盘后规则策略和模型组合的统一研究审计报告")
    parser.add_argument("--rules", required=True, help="path_rule_mining.py 的结果 JSON")
    parser.add_argument("--model", required=True, help="path_aware_gbdt_compound_backtest.py 的结果 JSON")
    parser.add_argument("--rule-portfolio", required=True, help="训练期冻结规则的可执行组合回测结果 JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    rules = _load(Path(args.rules), "规则")
    model = _load(Path(args.model), "模型组合")
    rule_portfolio = _load(Path(args.rule_portfolio), "规则组合")
    report = build_report(rules, model, rule_portfolio)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_clean_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "formal_production_changed": report["decision"]["formal_production_changed"],
        "promotion_allowed": report["decision"]["promotion_allowed"],
        "best_validation_rule": (
            (report["rule_strategy"].get("best_validation_rule") or {}).get("rule")
        ),
        "model_validation_gate_passed": report["model_strategy"]["validation_release_gate_passed"],
        "rule_portfolio_validation_gate_passed": report["rule_strategy"]["validation_release_gate_passed"],
        "same_execution_mechanics": report["same_execution_mechanics"],
        "training_selected_rule": report["rule_strategy"]["training_selected_rule"].get("rule_label"),
        "final_supplemental_is_fresh_unseen_holdout": False,
        "output": str(output),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
