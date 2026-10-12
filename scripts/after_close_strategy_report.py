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


def build_report(rules: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    rule_summary = summarize_rules(rules)
    model_summary = summarize_model(model)
    return {
        "schema_version": 1,
        "status": "research_only",
        "title": "盘后选股策略：规则挖掘与模型组合对照",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": {
            "signal": "T日收盘后",
            "entry": "T+1开盘",
            "earliest_exit": "T+2",
            "maximum_holding_sessions": 5,
            "target_net_profit_pct": 1.0,
            "stop_loss_pct": 3.0,
            "round_trip_cost_bps": 10.0,
            "limit_up_or_paused_entry_blocked": True,
        },
        "rule_strategy": rule_summary,
        "model_strategy": model_summary,
        "decision": {
            "formal_production_changed": False,
            "promotion_allowed": False,
            "reason": (
                "本次历史最终评估区间已被之前研究使用；规则结果为股票级别统计，"
                "模型结果为组合回测，口径不同，不能据此宣布胜出。"
                "只有统一组合口径、验证集收益与回撤合格，并完成真正前瞻验证后，才可考虑晋级。"
            ),
            "next_step": (
                "以验证期冻结的规则生成每日可执行TopN候选，并采用与模型相同的资金、"
                "成本、止盈止损与持仓约束进行现金约束组合回测。"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总盘后规则策略和模型组合的统一研究审计报告")
    parser.add_argument("--rules", required=True, help="path_rule_mining.py 的结果 JSON")
    parser.add_argument("--model", required=True, help="path_aware_gbdt_compound_backtest.py 的结果 JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    rules = _load(Path(args.rules), "规则")
    model = _load(Path(args.model), "模型组合")
    report = build_report(rules, model)
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
        "final_supplemental_is_fresh_unseen_holdout": False,
        "output": str(output),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
