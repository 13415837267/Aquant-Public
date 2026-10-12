"""验证直接净收益回归：先用验证期选择策略，再对最终留出集只评估一次。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compound_portfolio_backtest import (
    解析模型,
    提取每日候选,
    运行组合,
)
from scripts.selection_factor_catalog import load_research_config
from scripts.short_term_research import history_files
from scripts.train_executable_return_ranker import ExecutableReturnMLPRegressor

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_research_config()
NET_TARGET_PCT = float(CONFIG["短线目标净收益百分比"])
STOP_LOSS_PCT = float(CONFIG["止损幅度百分比"])
ROUND_TRIP_COST_BPS = float(CONFIG["往返交易成本基点"])
MAX_FORWARD_SESSIONS = int(CONFIG["最大前瞻交易日数"])
MINIMUM_TRADES = int(CONFIG["参数选择最低交易数"])
MAX_DRAWDOWN_FLOOR_PCT = float(CONFIG["参数选择最大允许回撤百分比"])
TARGET_HIT_GATE_PCT = float(CONFIG["目标命中率百分比"])


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"找不到研究产物：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"研究产物格式错误：{path}")
    return payload


def load_return_model(payload: dict) -> ExecutableReturnMLPRegressor:
    if payload.get("status") != "research_only":
        raise ValueError("只允许载入研究净收益模型")
    if payload.get("audit", {}).get("formal_production_changed") is not False:
        raise ValueError("净收益研究模型不能改变正式生产模型")
    if payload.get("audit", {}).get("final_holdout_used_for_training_or_selection") is not False:
        raise ValueError("净收益模型最终留出集审计未通过")
    if payload.get("audit", {}).get("training_target_is_realized_executable_net_return_not_binary_hit") is not True:
        raise ValueError("训练目标不是实际可执行净收益")
    features = payload.get("features", [])
    model_payload = payload.get("model", {})
    model = ExecutableReturnMLPRegressor(
        n_features=len(features),
        hidden_units=int(payload["parameters"]["hidden_units"]),
        learning_rate=float(payload["parameters"]["learning_rate"]),
        l2=float(payload["parameters"]["l2"]),
        seed=int(payload["parameters"]["random_seed"]),
    )
    model.w1 = np.asarray(model_payload["input_weights"], dtype=np.float64)
    model.b1 = np.asarray(model_payload["hidden_bias"], dtype=np.float64)
    model.w2 = np.asarray(model_payload["output_weights"], dtype=np.float64)
    model.b2 = float(model_payload["output_bias"])
    if model.w1.shape != (len(features), model.hidden_units):
        raise ValueError("净收益模型输入权重形状与特征数量不一致")
    if model.b1.shape != (model.hidden_units,) or model.w2.shape != (model.hidden_units,):
        raise ValueError("净收益模型隐藏层权重形状不一致")
    if not (
        np.isfinite(model.w1).all()
        and np.isfinite(model.b1).all()
        and np.isfinite(model.w2).all()
        and np.isfinite(model.b2)
    ):
        raise ValueError("净收益模型包含非有限权重")
    return model


def evaluate_policy(
    candidates: dict,
    dates: list[str],
    policy: str,
    start: str,
    end: str,
) -> dict:
    if policy == "每日强制Top1":
        return 运行组合(
            candidates,
            dates,
            "每日排名",
            NET_TARGET_PCT,
            STOP_LOSS_PCT,
            ROUND_TRIP_COST_BPS,
            top_k=1,
            窗口起始日期=start,
            窗口结束日期=end,
        )
    if policy == "预测净收益为正才入场":
        # 零点是经济盈亏平衡条件，不通过验证/留出集搜索阈值网格。
        return 运行组合(
            candidates,
            dates,
            "概率阈值",
            NET_TARGET_PCT,
            STOP_LOSS_PCT,
            ROUND_TRIP_COST_BPS,
            top_k=1,
            概率阈值=0.0,
            窗口起始日期=start,
            窗口结束日期=end,
        )
    raise ValueError(f"未知的研究策略：{policy}")


def gate_result(metrics: dict) -> dict:
    hit = metrics.get("target_hit_rate_pct")
    compound = metrics.get("compound_return_pct")
    drawdown = metrics.get("max_drawdown_pct")
    trades = int(metrics.get("completed_trades") or 0)
    checks = {
        "strict_target_hit_rate_at_least_80pct": hit is not None and float(hit) >= TARGET_HIT_GATE_PCT,
        "compound_return_positive": compound is not None and float(compound) > 0.0,
        "max_drawdown_within_configured_floor": drawdown is not None and float(drawdown) >= MAX_DRAWDOWN_FLOOR_PCT,
        "minimum_completed_trades": trades >= MINIMUM_TRADES,
    }
    return {"passed": all(checks.values()), "checks": checks}


def select_validation_policy(validation_results: dict[str, dict]) -> dict:
    """仅在验证期选择；优先选择全部达标者，否则选择可交易样本充足的最优诊断候选。"""
    candidates = []
    for policy in ("每日强制Top1", "预测净收益为正才入场"):
        metrics = validation_results[policy]
        gate = gate_result(metrics)
        candidates.append({
            "policy": policy,
            "metrics": metrics,
            "gate": gate,
            "completed_trades": int(metrics.get("completed_trades") or 0),
        })
    passed = [row for row in candidates if row["gate"]["passed"]]
    if passed:
        chosen = max(
            passed,
            key=lambda row: (
                float(row["metrics"]["compound_return_pct"]),
                float(row["metrics"]["max_drawdown_pct"]),
                float(row["metrics"]["target_hit_rate_pct"]),
            ),
        )
        status = "验证期全部质量门槛通过"
    else:
        enough_samples = [row for row in candidates if row["completed_trades"] >= MINIMUM_TRADES]
        pool = enough_samples or candidates
        chosen = max(
            pool,
            key=lambda row: (
                float(row["metrics"]["compound_return_pct"])
                if row["metrics"].get("compound_return_pct") is not None else -1e9,
                float(row["metrics"]["max_drawdown_pct"])
                if row["metrics"].get("max_drawdown_pct") is not None else -100.0,
                float(row["metrics"]["target_hit_rate_pct"])
                if row["metrics"].get("target_hit_rate_pct") is not None else -1.0,
            ),
        )
        status = "仅选择验证期最优诊断方案；未满足全部门槛，不允许晋级生产"
    return {
        "policy": chosen["policy"],
        "selection_status": status,
        "validation_gate": chosen["gate"],
        "selected_by_validation_only": True,
        "validation_selection_metrics": chosen["metrics"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="评估直接净收益回归的真实可执行组合表现")
    parser.add_argument("--model-result", default="data/backtest/high_precision_profit_mining_research_latest.json")
    parser.add_argument("--ranker-result", default="data/backtest/executable_return_ranker_research_latest.json")
    parser.add_argument("--output", default="data/backtest/executable_return_backtest_latest.json")
    args = parser.parse_args()

    source_path = Path(args.model_result).resolve()
    ranker_path = Path(args.ranker_result).resolve()
    output_path = Path(args.output).resolve()
    source = load_json(source_path)
    ranker_source = load_json(ranker_path)
    if source.get("status") != "research_only" or source.get("audit", {}).get("formal_production_changed") is not False:
        raise ValueError("分类基准产物不符合研究隔离约束")
    if source.get("audit", {}).get("final_holdout_used_once_after_selection") is not True:
        raise ValueError("分类基准没有最终留出集单次评估审计声明")
    baseline_features = source.get("model", {}).get("features", [])
    if baseline_features != ranker_source.get("features"):
        raise ValueError("净收益回归与分类基准特征清单不一致")
    baseline_model = 解析模型(source)
    return_model = load_return_model(ranker_source)

    files = history_files()
    dates = [path.name[:10] for path in files]
    splits = source["splits"]
    validation_start, validation_nominal_end = splits["validation"]
    final_start, final_end = splits["final"]
    if final_start not in dates or validation_start not in dates:
        raise ValueError("验证或最终留出边界不在历史行情内")
    final_start_index = dates.index(final_start)
    validation_safe_end_index = final_start_index - MAX_FORWARD_SESSIONS - 1
    if validation_safe_end_index < dates.index(validation_start):
        raise RuntimeError("清除最终留出边界后，验证窗口没有剩余样本")
    validation_end = min(validation_nominal_end, dates[validation_safe_end_index])

    print(f"[执行净收益对照] 验证窗口 {validation_start} 至 {validation_end}", flush=True)
    baseline_validation_candidates, baseline_validation_diagnostics = 提取每日候选(
        files, dates, validation_start, validation_end, baseline_model, NET_TARGET_PCT, ROUND_TRIP_COST_BPS
    )
    return_validation_candidates, return_validation_diagnostics = 提取每日候选(
        files, dates, validation_start, validation_end, return_model, NET_TARGET_PCT, ROUND_TRIP_COST_BPS
    )

    baseline_validation = evaluate_policy(
        baseline_validation_candidates, dates, "每日强制Top1", validation_start, validation_end
    )
    regression_top1_validation = evaluate_policy(
        return_validation_candidates, dates, "每日强制Top1", validation_start, validation_end
    )
    regression_positive_validation = evaluate_policy(
        return_validation_candidates, dates, "预测净收益为正才入场", validation_start, validation_end
    )
    validation_results = {
        "分类基准每日Top1": baseline_validation,
        "净收益回归每日强制Top1": regression_top1_validation,
        "净收益回归预测净收益为正才入场": regression_positive_validation,
    }
    chosen = select_validation_policy({
        "每日强制Top1": regression_top1_validation,
        "预测净收益为正才入场": regression_positive_validation,
    })

    # 最终留出期只评估预先固定的基准控制组与验证期选中的一项挑战策略。
    print(f"[执行净收益对照] 验证期选中策略：{chosen['policy']}；最终窗口 {final_start} 至 {final_end}", flush=True)
    baseline_final_candidates, baseline_final_diagnostics = 提取每日候选(
        files, dates, final_start, final_end, baseline_model, NET_TARGET_PCT, ROUND_TRIP_COST_BPS
    )
    return_final_candidates, return_final_diagnostics = 提取每日候选(
        files, dates, final_start, final_end, return_model, NET_TARGET_PCT, ROUND_TRIP_COST_BPS
    )
    baseline_final = evaluate_policy(
        baseline_final_candidates, dates, "每日强制Top1", final_start, final_end
    )
    selected_final = evaluate_policy(
        return_final_candidates, dates, chosen["policy"], final_start, final_end
    )
    selected_final_gate = gate_result(selected_final)
    baseline_final_return = baseline_final.get("compound_return_pct")
    selected_final_return = selected_final.get("compound_return_pct")
    compound_delta = (
        float(selected_final_return) - float(baseline_final_return)
        if selected_final_return is not None and baseline_final_return is not None
        else None
    )
    production_allowed = bool(chosen["validation_gate"]["passed"] and selected_final_gate["passed"])

    result = {
        "schema_version": 1,
        "status": "research_only",
        "method": "direct_realized_executable_net_return_regression_with_economic_abstention",
        "source_model_result": source_path.relative_to(ROOT).as_posix(),
        "source_ranker_result": ranker_path.relative_to(ROOT).as_posix(),
        "source_public_commit": ranker_source.get("source_public_commit"),
        "selection_policy": {
            "ranking": "predicted executable net return, descending",
            "candidates_per_signal_day": 10,
            "forced_entry_control": "one highest-scoring executable candidate per signal day",
            "abstention_policy": "enter only when predicted net return after costs is above 0 percent",
            "zero_threshold_is_economic_break_even_not_a_searched_parameter": True,
            "signal": "T_close",
            "entry": "T+1_open",
            "earliest_exit": "T+2",
            "maximum_exit_sessions": MAX_FORWARD_SESSIONS,
            "net_target_pct": NET_TARGET_PCT,
            "stop_loss_pct": STOP_LOSS_PCT,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "cash_constrained_no_leverage": True,
        },
        "splits": {
            "validation": [validation_start, validation_end],
            "validation_nominal_end": validation_nominal_end,
            "final_holdout": [final_start, final_end],
        },
        "validation": {
            "results": validation_results,
            "selected_policy": chosen,
            "diagnostics": {
                "classifier_baseline": baseline_validation_diagnostics,
                "executable_return_regression": return_validation_diagnostics,
            },
        },
        "final_holdout": {
            "classifier_baseline_daily_top1": baseline_final,
            "validation_selected_regression_policy": selected_final,
            "validation_selected_policy_name": chosen["policy"],
            "validation_selected_policy_gate": selected_final_gate,
            "compound_return_delta_vs_classifier_baseline_pct": compound_delta,
            "diagnostics": {
                "classifier_baseline": baseline_final_diagnostics,
                "executable_return_regression": return_final_diagnostics,
            },
        },
        "acceptance": {
            "strict_target_hit_rate_gate_pct": TARGET_HIT_GATE_PCT,
            "minimum_completed_trades": MINIMUM_TRADES,
            "maximum_drawdown_floor_pct": MAX_DRAWDOWN_FLOOR_PCT,
            "validation_gate_passed": bool(chosen["validation_gate"]["passed"]),
            "final_holdout_gate_passed": bool(selected_final_gate["passed"]),
            "promotion_allowed": production_allowed,
            "reason": (
                "验证期与最终留出期均通过全部质量门槛"
                if production_allowed
                else "至少一个评估窗口未通过命中率、正复利、回撤或最低交易数门槛；只保留研究结果"
            ),
        },
        "audit": {
            "same_feature_list_as_classifier_baseline": True,
            "training_target_is_realized_executable_net_return": True,
            "training_label_windows_end_before_validation": True,
            "validation_not_used_for_training": True,
            "final_holdout_not_used_for_training_or_selection": True,
            "final_holdout_evaluated_for_baseline_and_validation_selected_policy_only": True,
            "policy_selected_using_validation_only": True,
            "no_probability_threshold_grid_search": True,
            "T_plus_1_entry_and_T_plus_2_earliest_exit": True,
            "stop_precedes_target_on_same_day": True,
            "cash_constrained_no_leverage": True,
            "formal_production_changed": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "training_method": ranker_source.get("method"),
        "validation_selected_policy": chosen["policy"],
        "validation_gate_passed": chosen["validation_gate"]["passed"],
        "final_holdout": {
            "classifier_baseline": baseline_final,
            "selected_regression_policy": selected_final,
            "selected_policy_gate": selected_final_gate,
            "compound_return_delta_vs_baseline_pct": compound_delta,
        },
        "promotion_allowed": production_allowed,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
