"""训练期冻结规则的单因子匹配消融；仅产出研究报告，不修改生产策略。"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.after_close_rule_compound_backtest import (
    ROUND_TRIP_COST_BPS,
    STOP_LOSS_PCT,
    TOP_K_PER_DAY,
    NET_TARGET_PCT,
    collect_daily_candidates,
    complete_signal_end,
    run_portfolio,
)
from scripts.compound_portfolio_backtest import 初始资金
from scripts.path_rule_mining import (
    FINAL_END,
    FINAL_START,
    MAX_FORWARD_SESSIONS,
    STOCK_FEATURES,
    RESEARCH_ABLATION_FAMILIES,
    VALIDATION_START,
    history_files,
)
from scripts.selection_factor_catalog import load_research_config

TRAIN_END = "2022-12-30"
MIN_TRAIN_SAMPLES = int(load_research_config()["最少训练条件样本数"])
FAMILIES = RESEARCH_ABLATION_FAMILIES;


def _number(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return -math.inf
    return value if math.isfinite(value) else -math.inf


def _is_stock_rank(condition: dict[str, Any]) -> bool:
    return (
        condition.get("scale") == "rank"
        and condition.get("feature") in STOCK_FEATURES
    )


def select_ablation_candidates(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """仅根据训练期双条件候选挑规则；验证和补充期结果不参与选规则。"""
    if data.get("status") != "research_only":
        raise ValueError("输入必须是 research_only 规则挖掘结果")
    uses_family_candidates = "train_factor_ablation_pairs" in data
    if uses_family_candidates:
        train_pairs = data.get("train_factor_ablation_pairs")
        if not isinstance(train_pairs, list):
            raise ValueError("规则挖掘结果的 train_factor_ablation_pairs 必须是数组")
    else:
        train_pairs = data.get("train_top_pairs")
        if not isinstance(train_pairs, list):
            raise ValueError("规则挖掘结果缺少训练期双条件候选数组")

    selected: dict[str, dict[str, Any]] = {}
    for family, features in FAMILIES.items():
        candidates = []
        for row in train_pairs:
            if not isinstance(row, dict):
                continue
            if uses_family_candidates and row.get("target_family") != family:
                continue
            rules = row.get("rules")
            samples = row.get("samples")
            if not isinstance(rules, list) or len(rules) != 2:
                continue
            if not isinstance(samples, int) or samples < MIN_TRAIN_SAMPLES:
                continue
            targeted = [
                condition for condition in rules
                if isinstance(condition, dict) and condition.get("feature") in features
            ]
            controls = [
                condition for condition in rules
                if isinstance(condition, dict) and condition.get("feature") not in features
            ]
            # 一次只移除一个目标因子；移除后必须仍保留可用于个股筛选的因子。
            if len(targeted) != 1 or len(controls) != 1:
                continue
            if not _is_stock_rank(controls[0]):
                continue
            candidates.append((row, targeted[0], controls[0]))

        if not candidates:
            continue
        candidates.sort(
            key=lambda item: (
                _number(item[0], "wilson_lower_pct"),
                _number(item[0], "path_win_1pct_rate_pct"),
                _number(item[0], "samples"),
            ),
            reverse=True,
        )
        row, target_condition, control_condition = candidates[0]
        selected[family] = {
            "training_pair_label": row.get("rule_label"),
            "training_samples": row.get("samples"),
            "training_path_win_1pct_rate_pct": row.get("path_win_1pct_rate_pct"),
            "training_wilson_lower_pct": row.get("wilson_lower_pct"),
            "with_factor": [dict(control_condition), dict(target_condition)],
            "ablation": [dict(control_condition)],
            "target_condition": dict(target_condition),
            "control_condition": dict(control_condition),
            "selection_rule": "仅按训练期 Wilson 下界、路径命中率和样本数选定；不使用验证期及补充期指标",
        }
    return selected


def _portfolio_summary(portfolio: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status", "completed_trades", "compound_return_pct",
        "annualized_compound_return_pct", "max_drawdown_pct",
        "target_hit_rate_pct", "positive_trade_rate_pct",
        "mean_trade_net_return_pct", "final_equity", "sharpe_ratio",
        "open_positions_at_end", "skipped_for_capacity_or_cash",
        "signal_window", "measurement_window",
    )
    return {key: portfolio.get(key) for key in keys if key in portfolio}


def _evaluate(
    files: list[Path],
    dates: list[str],
    conditions: list[dict[str, Any]],
    start_date: str,
    signal_end: str,
) -> dict[str, Any]:
    daily, diagnostics = collect_daily_candidates(
        files, dates, start_date, signal_end, conditions
    )
    portfolio = run_portfolio(daily, dates, start_date, signal_end)
    result = _portfolio_summary(portfolio)
    result["candidate_diagnostics"] = diagnostics
    return result


def _delta(with_factor: Any, ablation: Any) -> float | None:
    try:
        left, right = float(with_factor), float(ablation)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(left) or not math.isfinite(right):
        return None
    return round(left - right, 6)


def run_ablation(data: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    selected = select_ablation_candidates(data)
    files = history_files()
    dates = [path.name[:10] for path in files]
    missing = [
        date for date in (TRAIN_END, VALIDATION_START, FINAL_START, FINAL_END)
        if date not in set(dates)
    ]
    if missing:
        raise ValueError("历史行情缺少研究边界日期：" + "、".join(missing))

    final_start_index = dates.index(FINAL_START)
    validation_start_index = dates.index(VALIDATION_START)
    validation_end_index = final_start_index - MAX_FORWARD_SESSIONS - 1
    if validation_end_index < validation_start_index:
        raise ValueError("清除跨期标签后，验证窗口为空")
    validation_end = dates[validation_end_index]
    final_end = complete_signal_end(dates, FINAL_END)

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "research_only",
        "method": "training_frozen_matched_factor_ablation",
        "source_public_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "objective": {
            "signal": "T日收盘",
            "entry": "T+1开盘",
            "earliest_exit": "T+2",
            "maximum_holding_sessions": MAX_FORWARD_SESSIONS,
            "net_target_pct": NET_TARGET_PCT,
            "stop_loss_pct": STOP_LOSS_PCT,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "top_k_per_signal_day": TOP_K_PER_DAY,
            "initial_equity": 初始资金,
            "cash_constrained_no_leverage": True,
        },
        "splits": {
            "training": ["2015-01-05", TRAIN_END],
            "validation": [VALIDATION_START, validation_end],
            "final_supplemental": [FINAL_START, final_end],
        },
        "families": {},
        "skipped_families": sorted(set(FAMILIES) - set(selected)),
        "skipped_reason": (
            "按因子族分层生成的训练期目标条件×个股控制条件候选中，没有规则达到样本量要求或保留有效个股控制条件。"
            if len(selected) < len(FAMILIES) else None
        ),
        "final_supplemental_is_fresh_unseen_holdout": False,
        "formal_production_changed": False,
        "elapsed_seconds": None,
    }

    total_runs = 4 * len(selected)
    run_number = 0
    for family, rule in selected.items():
        print(f"[因子消融] 开始：{family}；训练期冻结规则={rule['training_pair_label']}", flush=True)
        family_result = {
            "training_pair_label": rule["training_pair_label"],
            "training_samples": rule["training_samples"],
            "training_path_win_1pct_rate_pct": rule["training_path_win_1pct_rate_pct"],
            "training_wilson_lower_pct": rule["training_wilson_lower_pct"],
            "target_condition": rule["target_condition"],
            "control_condition": rule["control_condition"],
            "selection_rule": rule["selection_rule"],
            "variants": {},
            "comparison_delta": {},
        }
        for variant_name, conditions in (
            ("ablation", rule["ablation"]),
            ("with_factor", rule["with_factor"]),
        ):
            period_results = {}
            for split_name, start_date, signal_end in (
                ("validation", VALIDATION_START, validation_end),
                ("final_supplemental", FINAL_START, final_end),
            ):
                run_number += 1
                print(
                    f"[因子消融] {run_number}/{total_runs}：{family} / {variant_name} / {split_name}",
                    flush=True,
                )
                period_results[split_name] = _evaluate(
                    files, dates, conditions, start_date, signal_end
                )
            family_result["variants"][variant_name] = period_results

        for split_name in ("validation", "final_supplemental"):
            without = family_result["variants"]["ablation"][split_name]
            with_factor = family_result["variants"]["with_factor"][split_name]
            keys = (
                "compound_return_pct", "max_drawdown_pct",
                "target_hit_rate_pct", "positive_trade_rate_pct",
                "mean_trade_net_return_pct", "completed_trades",
            )
            family_result["comparison_delta"][split_name] = {
                key: _delta(with_factor.get(key), without.get(key))
                for key in keys
            }
        result["families"][family] = family_result
        print(
            f"[因子消融] 完成：{family}；"
            f"验证期复利差={family_result['comparison_delta']['validation']['compound_return_pct']}；"
            f"补充期复利差={family_result['comparison_delta']['final_supplemental']['compound_return_pct']}",
            flush=True,
        )

    result["elapsed_seconds"] = round(time.time() - started, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="训练期冻结规则的单因子匹配消融")
    parser.add_argument("--rules", required=True, help="路径规则挖掘 JSON")
    parser.add_argument(
        "--output",
        default=str(ROOT / "data/backtest/after_close_factor_ablation.json"),
        help="单因子消融输出 JSON",
    )
    args = parser.parse_args()
    source = Path(args.rules)
    data = json.loads(source.read_text(encoding="utf-8"))
    result = run_ablation(data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "method": result["method"],
        "families_tested": list(result["families"]),
        "skipped_families": result["skipped_families"],
        "elapsed_seconds": result["elapsed_seconds"],
        "output": str(output),
        "formal_production_changed": False,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
