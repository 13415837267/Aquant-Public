"""在固定每日Top2规则下比较分类、线性排序和非线性排序模型。"""
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

from scripts.compound_portfolio_backtest import 解析模型
from scripts.pairwise_rank_compound_backtest import (
    PairwiseLinearRanker,
    _backtest_window,
)
from scripts.selection_factor_catalog import load_research_config
from scripts.short_term_research import history_files

DEFAULT_INPUT = ROOT / "data/backtest/high_precision_profit_mining_research_latest.json"
DEFAULT_RANKER_INPUT = ROOT / "data/backtest/pairwise_mlp_ranker_research_latest.json"
DEFAULT_OUTPUT = ROOT / "data/backtest/pairwise_rank_compound_latest.json"
CONFIG = load_research_config()
NET_TARGET_PCT = float(CONFIG["短线目标净收益百分比"])
STOP_LOSS_PCT = float(CONFIG["止损幅度百分比"])
ROUND_TRIP_COST_BPS = float(CONFIG["往返交易成本基点"])
MAX_FORWARD_SESSIONS = int(CONFIG["最大前瞻交易日数"])


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


class PairwiseMLPRanker:
    """从独立训练产物载入非线性成对排序模型，只输出用于排名的分数。"""

    def __init__(self, payload: dict):
        self.features = list(payload["features"])
        model = payload["model"]
        self.w1 = np.asarray(model["input_weights"], dtype=np.float64)
        self.b1 = np.asarray(model["hidden_bias"], dtype=np.float64)
        self.w2 = np.asarray(model["output_weights"], dtype=np.float64)
        self.b2 = float(model["output_bias"])
        if self.w1.shape != (len(self.features), len(self.b1)):
            raise ValueError("非线性成对排序模型的输入权重维度不匹配")
        if self.w2.shape != (len(self.b1),):
            raise ValueError("非线性成对排序模型的输出权重维度不匹配")
        if not (
            np.isfinite(self.w1).all()
            and np.isfinite(self.b1).all()
            and np.isfinite(self.w2).all()
            and np.isfinite(self.b2)
        ):
            raise ValueError("非线性成对排序模型包含非有限权重")

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.features):
            raise ValueError("非线性成对排序模型的输入特征数量不匹配")
        hidden = np.tanh(np.clip(matrix @ self.w1 + self.b1, -8.0, 8.0))
        return sigmoid(hidden @ self.w2 + self.b2)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"找不到研究产物：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"研究产物格式错误：{path}")
    return payload


def build_result(
    source: dict,
    ranker_source: dict,
    input_path: Path,
    ranker_path: Path,
    files: list[Path],
) -> dict:
    if source.get("status") != "research_only":
        raise ValueError("只允许读取研究模型产物")
    if source.get("audit", {}).get("formal_production_changed") is not False:
        raise ValueError("研究结果不得改变正式生产模型")
    if source.get("audit", {}).get("final_holdout_used_once_after_selection") is not True:
        raise ValueError("分类基准缺少最终留出集审计声明")
    if ranker_source.get("status") != "research_only":
        raise ValueError("只允许读取非线性排序研究产物")
    if ranker_source.get("audit", {}).get("formal_production_changed") is not False:
        raise ValueError("非线性排序研究结果不得改变正式生产模型")
    if ranker_source.get("audit", {}).get("final_holdout_used_for_training_or_selection") is not False:
        raise ValueError("非线性排序模型的训练/选择审计不通过")
    if ranker_source.get("audit", {}).get("probability_threshold_tuning_used") is not False:
        raise ValueError("非线性排序模型不得通过概率阈值调参")
    baseline_features = source.get("model", {}).get("features", [])
    if not baseline_features or baseline_features != source.get("pairwise_ranker", {}).get("features"):
        raise ValueError("分类模型与线性排序模型的特征清单不一致")
    if baseline_features != ranker_source.get("features"):
        raise ValueError("分类模型与非线性排序模型的特征清单不一致")

    splits = source["splits"]
    validation_start = splits["validation"][0]
    validation_nominal_end = splits["validation"][1]
    final_start, final_end = splits["final"]
    dates = [path.name[:10] for path in files]
    final_start_index = dates.index(final_start)
    validation_safe_end_index = final_start_index - MAX_FORWARD_SESSIONS - 1
    if validation_safe_end_index < dates.index(validation_start):
        raise RuntimeError("清除最终留出边界后，验证窗口没有剩余样本")
    validation_safe_end = min(validation_nominal_end, dates[validation_safe_end_index])

    models = {
        "分类基准模型": 解析模型(source),
        "线性成对排序模型": PairwiseLinearRanker(source["pairwise_ranker"]),
        "非线性成对排序模型": PairwiseMLPRanker(ranker_source),
    }
    results = {}
    for name, model in models.items():
        validation = _backtest_window(
            model, files, dates, validation_start, validation_safe_end
        )
        final_holdout = _backtest_window(
            model, files, dates, final_start, final_end
        )
        results[name] = {
            "validation": validation,
            "final_holdout": final_holdout,
        }

    baseline_validation = results["分类基准模型"]["validation"]["portfolio"]
    baseline_final = results["分类基准模型"]["final_holdout"]["portfolio"]
    comparisons = {}
    candidates = {
        "linear_pairwise": "线性成对排序模型",
        "nonlinear_pairwise": "非线性成对排序模型",
    }
    for prefix, name in candidates.items():
        validation_portfolio = results[name]["validation"]["portfolio"]
        final_portfolio = results[name]["final_holdout"]["portfolio"]
        comparisons[f"validation_{prefix}_compound_return_delta_pct"] = (
            float(validation_portfolio["compound_return_pct"])
            - float(baseline_validation["compound_return_pct"])
        )
        comparisons[f"final_{prefix}_compound_return_delta_pct"] = (
            float(final_portfolio["compound_return_pct"])
            - float(baseline_final["compound_return_pct"])
        )
        comparisons[f"final_{prefix}_target_hit_rate_delta_pct"] = (
            float(final_portfolio["target_hit_rate_pct"])
            - float(baseline_final["target_hit_rate_pct"])
        )

    public_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    return {
        "schema_version": 2,
        "status": "research_only",
        "method": "fixed_daily_top2_three_model_compound_comparison",
        "source_model_result": input_path.relative_to(ROOT).as_posix(),
        "source_ranker_result": ranker_path.relative_to(ROOT).as_posix(),
        "source_public_commit": public_commit,
        "selection_policy": {
            "top_k_per_signal_day": 2,
            "probability_threshold_used": False,
            "ranking_only": True,
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
            "validation": [validation_start, validation_safe_end],
            "validation_nominal_end": validation_nominal_end,
            "final_holdout": [final_start, final_end],
        },
        "models": results,
        "observed_comparison": comparisons,
        "audit": {
            "same_features_and_executable_entry_rules": True,
            "same_position_sizing_exit_rules_and_costs": True,
            "validation_label_windows_end_before_final_holdout": True,
            "final_holdout_used_for_parameter_selection": False,
            "no_threshold_or_final_period_tuning": True,
            "nonlinear_ranker_trained_without_validation_or_holdout": True,
            "formal_production_changed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="固定每日Top2规则的三模型可执行复利对照")
    parser.add_argument("--model-result", default=str(DEFAULT_INPUT))
    parser.add_argument("--ranker-result", default=str(DEFAULT_RANKER_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_path = Path(args.model_result).resolve()
    ranker_path = Path(args.ranker_result).resolve()
    output_path = Path(args.output).resolve()
    source = load_json(input_path)
    ranker_source = load_json(ranker_path)
    result = build_result(source, ranker_source, input_path, ranker_path, history_files())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "splits": result["splits"],
                "observed_comparison": result["observed_comparison"],
                "validation": {
                    name: item["validation"]["portfolio"]
                    for name, item in result["models"].items()
                },
                "final_holdout": {
                    name: item["final_holdout"]["portfolio"]
                    for name, item in result["models"].items()
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
