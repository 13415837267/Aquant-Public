"""以固定的每日Top2规则对照分类模型与成对排序模型，不做概率阈值搜索。"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compound_portfolio_backtest import (
    初始资金,
    解析模型,
    提取每日候选,
    运行组合,
)
from scripts.selection_factor_catalog import load_research_config
from scripts.short_term_research import history_files

DEFAULT_INPUT = ROOT / "data/backtest/high_precision_profit_mining_research_latest.json"
DEFAULT_OUTPUT = ROOT / "data/backtest/pairwise_rank_compound_latest.json"
CONFIG = load_research_config()
NET_TARGET_PCT = float(CONFIG["短线目标净收益百分比"])
STOP_LOSS_PCT = float(CONFIG["止损幅度百分比"])
ROUND_TRIP_COST_BPS = float(CONFIG["往返交易成本基点"])
MAX_FORWARD_SESSIONS = int(CONFIG["最大前瞻交易日数"])
TOP_K_PER_DAY = 2


def _sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


class PairwiseLinearRanker:
    def __init__(self, payload: dict):
        self.features = list(payload["features"])
        coefficients = payload["coefficients"]
        self.weights = np.asarray([coefficients[name] for name in self.features], dtype=np.float64)
        if self.weights.shape != (len(self.features),):
            raise ValueError("成对排序模型的特征权重数量不匹配")

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[1] != len(self.weights):
            raise ValueError("成对排序模型的输入特征数量不匹配")
        return _sigmoid(matrix @ self.weights)


def _backtest_window(model, files, dates, start_date: str, end_date: str) -> dict:
    daily_candidates, diagnostics = 提取每日候选(
        files, dates, start_date, end_date, model, NET_TARGET_PCT, ROUND_TRIP_COST_BPS
    )
    portfolio = 运行组合(
        daily_candidates,
        dates,
        "每日排名",
        NET_TARGET_PCT,
        STOP_LOSS_PCT,
        ROUND_TRIP_COST_BPS,
        top_k=TOP_K_PER_DAY,
    )
    return {
        "signal_window": [start_date, end_date],
        "selection_diagnostics": diagnostics,
        "portfolio": portfolio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-result", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_path = Path(args.model_result)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到高精度模型产物：{input_path}")
    source = json.loads(input_path.read_text(encoding="utf-8"))
    if source.get("status") != "research_only":
        raise ValueError("只允许读取研究模型产物")
    if source.get("audit", {}).get("formal_production_changed") is not False:
        raise ValueError("研究结果不得改变正式生产模型")
    if source.get("audit", {}).get("final_holdout_used_once_after_selection") is not True:
        raise ValueError("研究产物缺少最终留出集审计声明")
    pairwise_payload = source.get("pairwise_ranker")
    if not pairwise_payload or pairwise_payload.get("audit", {}).get("threshold_tuning_used") is not False:
        raise ValueError("缺少无阈值调参的成对排序研究产物")

    splits = source["splits"]
    validation_start = splits["validation"][0]
    validation_nominal_end = splits["validation"][1]
    final_start, final_end = splits["final"]
    files = history_files()
    dates = [p.name[:10] for p in files]
    final_start_index = dates.index(final_start)
    validation_safe_end_index = final_start_index - MAX_FORWARD_SESSIONS - 1
    if validation_safe_end_index < dates.index(validation_start):
        raise RuntimeError("验证窗口在清除最终留出边界后没有剩余样本")
    validation_safe_end = min(validation_nominal_end, dates[validation_safe_end_index])

    baseline_model = 解析模型(source)
    pairwise_model = PairwiseLinearRanker(pairwise_payload)
    models = {
        "分类基准模型": baseline_model,
        "成对排序模型": pairwise_model,
    }
    result = {
        "schema_version": 1,
        "status": "research_only",
        "method": "fixed_daily_top2_cash_constrained_compound_comparison",
        "source_model_result": str(input_path.relative_to(ROOT)) if input_path.is_relative_to(ROOT) else input_path.name,
        "source_public_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "selection_policy": {
            "top_k_per_signal_day": TOP_K_PER_DAY,
            "probability_threshold_used": False,
            "ranking_only": True,
            "signal": "T_close",
            "entry": "T+1_open",
            "earliest_exit": "T+2",
            "maximum_exit_sessions": MAX_FORWARD_SESSIONS,
            "net_target_pct": NET_TARGET_PCT,
            "stop_loss_pct": STOP_LOSS_PCT,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "initial_equity": 初始资金,
            "cash_constrained_no_leverage": True,
        },
        "splits": {
            "validation": [validation_start, validation_safe_end],
            "validation_nominal_end": validation_nominal_end,
            "final_holdout": [final_start, final_end],
        },
        "models": {},
        "audit": {
            "same_features_and_executable_entry_rules": True,
            "same_position_sizing_exit_rules_and_costs": True,
            "validation_label_windows_end_before_final_holdout": True,
            "final_holdout_used_for_parameter_selection": False,
            "no_threshold_or_final_period_tuning": True,
            "formal_production_changed": False,
        },
    }

    for name, model in models.items():
        validation = _backtest_window(model, files, dates, validation_start, validation_safe_end)
        final = _backtest_window(model, files, dates, final_start, final_end)
        result["models"][name] = {
            "validation": validation,
            "final_holdout": final,
        }

    baseline_val = result["models"]["分类基准模型"]["validation"]["portfolio"]
    pairwise_val = result["models"]["成对排序模型"]["validation"]["portfolio"]
    baseline_final = result["models"]["分类基准模型"]["final_holdout"]["portfolio"]
    pairwise_final = result["models"]["成对排序模型"]["final_holdout"]["portfolio"]
    result["observed_comparison"] = {
        "validation_pairwise_compound_return_delta_pct": (
            float(pairwise_val["compound_return_pct"]) - float(baseline_val["compound_return_pct"])
            if pairwise_val.get("compound_return_pct") is not None and baseline_val.get("compound_return_pct") is not None
            else None
        ),
        "final_pairwise_compound_return_delta_pct": (
            float(pairwise_final["compound_return_pct"]) - float(baseline_final["compound_return_pct"])
            if pairwise_final.get("compound_return_pct") is not None and baseline_final.get("compound_return_pct") is not None
            else None
        ),
        "final_pairwise_target_hit_rate_delta_pct": (
            float(pairwise_final["target_hit_rate_pct"]) - float(baseline_final["target_hit_rate_pct"])
            if pairwise_final.get("target_hit_rate_pct") is not None and baseline_final.get("target_hit_rate_pct") is not None
            else None
        ),
        "final_holdout_is_reported_for_single_confirmatory_comparison_only": True,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "splits": result["splits"],
        "validation": {
            name: item["validation"]["portfolio"]
            for name, item in result["models"].items()
        },
        "final_holdout": {
            name: item["final_holdout"]["portfolio"]
            for name, item in result["models"].items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
