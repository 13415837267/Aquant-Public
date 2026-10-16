"""分别为每日Top1和Top2候选上限选择验证期参数，并进行冻结后的补充评估。"""
from __future__ import annotations

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

import scripts.path_aware_gbdt_compound_backtest as core
from scripts.short_term_research import history_files
from scripts.train_path_aware_gbdt import (
    FINAL_END,
    FINAL_START,
    MAX_FORWARD_SESSIONS,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    collect_training,
)

DEFAULT_OUTPUT = ROOT / "data/backtest/candidate_count_operating_point_latest.json"
TOP_K_VALUES = (1, 2)
MINIMUM_PROFIT_FACTOR = float(core.配置.get("复利研究最低盈利因子", 1.2))
COST_STRESS_MULTIPLIERS = tuple(
    float(value) for value in core.配置.get("候选数量成本压力倍率", (1.5, 2.0))
)


def _finite_number(row: dict[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def eligible_operating_point(
    row: dict[str, Any],
    minimum_trades: int,
    max_drawdown_floor_pct: float,
) -> bool:
    """与正式研究门槛保持一致；仅用于判断验证期参数是否有资格被选中。"""
    try:
        trades = int(row.get("completed_trades") or 0)
    except (TypeError, ValueError):
        return False
    threshold = _finite_number(row, "probability_threshold")
    target = _finite_number(row, "net_target_pct")
    stop_loss = _finite_number(row, "stop_loss_pct")
    compound = _finite_number(row, "compound_return_pct")
    annualized = _finite_number(row, "annualized_compound_return_pct")
    drawdown = _finite_number(row, "max_drawdown_pct")
    mean_trade = _finite_number(row, "mean_trade_net_return_pct")
    profit_factor = _finite_number(row, "profit_factor")
    values = (threshold, target, stop_loss, compound, annualized, drawdown, mean_trade, profit_factor)
    if any(value is None for value in values):
        return False
    if not 0.0 <= threshold <= 1.0 or target <= 0.0 or stop_loss <= 0.0:
        return False
    return (
        trades >= minimum_trades
        and compound > 0.0
        and annualized > 0.0
        and mean_trade > 0.0
        and profit_factor >= MINIMUM_PROFIT_FACTOR
        and drawdown >= max_drawdown_floor_pct
    )


def independently_select_parameters(
    daily_candidates: dict[str, list[dict]],
    diagnostics: dict,
    dates: list[str],
    start_date: str,
    end_date: str,
    top_ks: tuple[int, ...] = TOP_K_VALUES,
) -> dict[int, dict[str, Any]]:
    """每个候选数量独立运行同一验证网格，不将Top2参数直接套给Top1。"""
    if (
        not top_ks
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in top_ks)
        or len(set(top_ks)) != len(top_ks)
    ):
        raise ValueError("候选数量必须是无重复的正整数序列")

    result: dict[int, dict[str, Any]] = {}
    for top_k in top_ks:
        print(f"[候选数量独立选参] 开始每日最多{top_k}只的验证期参数网格", flush=True)
        sweep = core._exit_parameter_sweep(
            daily_candidates, diagnostics, dates, start_date, end_date, top_k
        )
        eligible_rows = [
            row for row in sweep
            if eligible_operating_point(
                row,
                minimum_trades=core.最低验证交易数,
                max_drawdown_floor_pct=core.退出参数最大回撤门槛,
            )
        ]
        eligible_count = len(eligible_rows)
        selected = max(
            eligible_rows,
            key=lambda row: (
                float(row["compound_return_pct"]),
                float(row["annualized_compound_return_pct"]),
                float(row["mean_trade_net_return_pct"]),
                float(row["max_drawdown_pct"]),
                int(row["completed_trades"]),
            ),
            default=None,
        )
        if selected is not None:
            selected = dict(selected)
            selected["selection_status"] = (
                f"仅由验证期选择；盈利因子不低于{MINIMUM_PROFIT_FACTOR:.2f}；"
                "联合参数网格仍存在多重比较偏差"
            )
        print(
            f"[候选数量独立选参] 每日最多{top_k}只："
            f"网格{len(sweep)}组，达门槛{eligible_count}组，"
            f"选中={'是' if selected is not None else '否'}",
            flush=True,
        )
        result[top_k] = {
            "sweep": sweep,
            "selected": selected,
            "eligible_parameter_count": eligible_count,
            "grid_size": len(sweep),
        }
    return result


def select_cost_robust_operating_point(
    sweeps_by_cost: dict[float, list[dict[str, Any]]],
    minimum_trades: int,
    max_drawdown_floor_pct: float,
) -> dict[str, Any]:
    """只用验证期结果，筛选在基准成本及所有压力成本下同时达标的操作点。"""
    required_multipliers = (1.0, *COST_STRESS_MULTIPLIERS)
    normalized_sweeps = {round(float(multiplier), 6): rows for multiplier, rows in sweeps_by_cost.items()}
    missing = [multiplier for multiplier in required_multipliers if round(float(multiplier), 6) not in normalized_sweeps]
    if missing:
        raise ValueError("成本稳健选参缺少成本场景：" + "、".join(str(value) for value in missing))

    indexed: dict[float, dict[tuple[float, float, float], dict[str, Any]]] = {}
    for multiplier in required_multipliers:
        index: dict[tuple[float, float, float], dict[str, Any]] = {}
        for row in normalized_sweeps[round(float(multiplier), 6)]:
            key = (
                round(float(row["probability_threshold"]), 3),
                round(float(row["net_target_pct"]), 3),
                round(float(row["stop_loss_pct"]), 3),
            )
            if key in index:
                raise ValueError(f"成本场景{multiplier}包含重复参数组合：{key}")
            index[key] = row
        indexed[float(multiplier)] = index

    common_keys = set.intersection(*(set(rows) for rows in indexed.values()))
    eligible_keys = [
        key for key in common_keys
        if all(
            eligible_operating_point(
                indexed[multiplier][key],
                minimum_trades=minimum_trades,
                max_drawdown_floor_pct=max_drawdown_floor_pct,
            )
            for multiplier in required_multipliers
        )
    ]

    highest_cost = max(float(value) for value in required_multipliers)
    priority_multipliers = sorted((float(value) for value in required_multipliers), reverse=True)

    def score(key: tuple[float, float, float]) -> tuple[float, ...]:
        rows = [indexed[multiplier][key] for multiplier in priority_multipliers]
        compounds = [float(row["compound_return_pct"]) for row in rows]
        stressed_drawdown = float(indexed[highest_cost][key]["max_drawdown_pct"])
        profit_factors = [float(indexed[multiplier][key]["profit_factor"]) for multiplier in priority_multipliers]
        return (*compounds, stressed_drawdown, min(profit_factors), int(indexed[1.0][key]["completed_trades"]))

    selected_key = max(eligible_keys, key=score, default=None)
    return {
        "selected_parameters": indexed[1.0][selected_key] if selected_key is not None else None,
        "eligible_parameter_count": len(eligible_keys),
        "validation_by_cost": {
            float(multiplier): indexed[float(multiplier)][selected_key]
            for multiplier in required_multipliers
        } if selected_key is not None else {},
        "cost_multipliers": [float(value) for value in required_multipliers],
        "selection_rule": (
            "基准成本及所有压力成本均须满足交易数、正复利、正年化复利、"
            "正平均单笔净收益、盈利因子和回撤门槛；达标后按最高成本复利优先排序"
        ),
    }


def adjacent_operating_points(
    selected: dict[str, Any],
    threshold_grid: tuple[float, ...],
    target_grid: tuple[float, ...],
    stop_grid: tuple[float, ...],
) -> list[dict[str, Any]]:
    """返回与选中参数仅有一个维度、且相差一个网格步的相邻组合。"""
    grid_specs = (
        ("probability_threshold", threshold_grid),
        ("net_target_pct", target_grid),
        ("stop_loss_pct", stop_grid),
    )
    points: list[dict[str, Any]] = []
    selected_key = {
        "probability_threshold": round(float(selected["probability_threshold"]), 3),
        "net_target_pct": round(float(selected["net_target_pct"]), 3),
        "stop_loss_pct": round(float(selected["stop_loss_pct"]), 3),
    }
    for field, values in grid_specs:
        grid = [round(float(value), 3) for value in values]
        try:
            index = grid.index(selected_key[field])
        except ValueError as exc:
            raise ValueError(f"已选参数不在预先声明的网格内：{field}") from exc
        for neighbor_index in (index - 1, index + 1):
            if not 0 <= neighbor_index < len(grid):
                continue
            item = dict(selected_key)
            item[field] = grid[neighbor_index]
            item["changed_parameter"] = field
            points.append(item)
    return points


def _metric_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    fields = (
        "probability_threshold",
        "net_target_pct",
        "stop_loss_pct",
        "round_trip_cost_bps",
        "completed_trades",
        "compound_return_pct",
        "annualized_compound_return_pct",
        "max_drawdown_pct",
        "mean_trade_net_return_pct",
        "target_hit_rate_pct",
        "positive_trade_rate_pct",
        "profit_factor",
        "exit_reason_counts",
    )
    return {key: row.get(key) for key in fields}


def _portfolio_summary(portfolio: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": portfolio.get("status"),
        "round_trip_cost_bps": (portfolio.get("selection_policy") or {}).get("round_trip_cost_bps"),
        "requested_entries": portfolio.get("requested_entries"),
        "entered_trades": portfolio.get("entered_trades"),
        "completed_trades": portfolio.get("completed_trades"),
        "compound_return_pct": portfolio.get("compound_return_pct"),
        "annualized_compound_return_pct": portfolio.get("annualized_compound_return_pct"),
        "max_drawdown_pct": portfolio.get("max_drawdown_pct"),
        "mean_trade_net_return_pct": portfolio.get("mean_trade_net_return_pct"),
        "profit_factor": portfolio.get("profit_factor"),
        "yearly_compound_return_pct": portfolio.get("yearly_compound_return_pct"),
        "target_hit_rate_pct": portfolio.get("target_hit_rate_pct"),
        "positive_trade_rate_pct": portfolio.get("positive_trade_rate_pct"),
        "skipped_for_capacity_or_cash": portfolio.get("skipped_for_capacity_or_cash"),
        "open_positions_at_end": portfolio.get("open_positions_at_end"),
    }


def _find_grid_row(sweep: list[dict[str, Any]], parameters: dict[str, Any]) -> dict[str, Any] | None:
    target = (
        round(float(parameters["probability_threshold"]), 3),
        round(float(parameters["net_target_pct"]), 3),
        round(float(parameters["stop_loss_pct"]), 3),
    )
    for row in sweep:
        key = (
            round(float(row["probability_threshold"]), 3),
            round(float(row["net_target_pct"]), 3),
            round(float(row["stop_loss_pct"]), 3),
        )
        if key == target:
            return row
    return None


def _evaluate_final(
    final_candidates: dict[str, list[dict]],
    dates: list[str],
    final_signal_end: str,
    top_k: int,
    parameters: dict[str, Any],
    round_trip_cost_bps: float | None = None,
) -> dict[str, Any]:
    return core._run_threshold_portfolio(
        final_candidates,
        dates,
        FINAL_START,
        final_signal_end,
        float(parameters["probability_threshold"]),
        top_k,
        target_net_pct=float(parameters["net_target_pct"]),
        stop_loss_pct=float(parameters["stop_loss_pct"]),
        round_trip_cost_bps=round_trip_cost_bps,
    )


def filter_candidates_by_market_breadth(
    daily_candidates: dict[str, list[dict]],
    diagnostics: dict[str, Any],
    minimum_breadth_pct: float | None,
) -> dict[str, list[dict]]:
    """按T日收盘已知的市场上涨广度过滤整日候选，不使用未来价格。"""
    if minimum_breadth_pct is not None:
        floor = float(minimum_breadth_pct)
        if not math.isfinite(floor) or not 0.0 <= floor <= 100.0:
            raise ValueError("市场广度门槛必须是0至100之间的有限百分比")
    else:
        floor = None

    contexts = diagnostics.get("market_breadth_pct_by_signal_date")
    if not isinstance(contexts, dict):
        raise ValueError("候选诊断缺少按信号日保存的市场广度")
    filtered: dict[str, list[dict]] = {}
    for signal_date, candidates in daily_candidates.items():
        if signal_date not in contexts:
            raise ValueError(f"候选信号日缺少市场广度：{signal_date}")
        breadth = float(contexts[signal_date])
        if not math.isfinite(breadth) or not 0.0 <= breadth <= 100.0:
            raise ValueError(f"信号日市场广度无效：{signal_date}={breadth}")
        filtered[signal_date] = (
            list(candidates) if floor is None or breadth >= floor else []
        )
    return filtered


def select_market_breadth_floor(
    validation_grid: list[dict[str, Any]],
    minimum_trades: int,
    max_drawdown_floor_pct: float,
) -> dict[str, Any]:
    """只按验证期选市场广度门槛；所有成本场景必须同时达标。"""
    required_multipliers = tuple(sorted((1.0, *COST_STRESS_MULTIPLIERS)))
    eligible = []
    for item in validation_grid:
        by_multiplier = {
            round(float(row["cost_multiplier"]), 6): row
            for row in item.get("validation_by_cost", [])
        }
        if any(round(multiplier, 6) not in by_multiplier for multiplier in required_multipliers):
            continue
        if all(
            eligible_operating_point(
                by_multiplier[round(multiplier, 6)],
                minimum_trades=minimum_trades,
                max_drawdown_floor_pct=max_drawdown_floor_pct,
            )
            for multiplier in required_multipliers
        ):
            eligible.append(item)

    def score(item: dict[str, Any]) -> tuple[float, ...]:
        by_multiplier = {
            round(float(row["cost_multiplier"]), 6): row
            for row in item["validation_by_cost"]
        }
        ordered = sorted(required_multipliers, reverse=True)
        compounds = tuple(float(by_multiplier[round(multiplier, 6)]["compound_return_pct"]) for multiplier in ordered)
        highest_cost = by_multiplier[round(max(required_multipliers), 6)]
        minimum_profit_factor = min(
            float(by_multiplier[round(multiplier, 6)]["profit_factor"])
            for multiplier in required_multipliers
        )
        base_trades = int(by_multiplier[1.0]["completed_trades"])
        return (
            *compounds,
            float(highest_cost["max_drawdown_pct"]),
            minimum_profit_factor,
            base_trades,
        )

    selected = max(eligible, key=score, default=None)
    return {
        "eligible_market_breadth_floor_count": len(eligible),
        "eligible_market_breadth_floors_pct": [
            item.get("market_breadth_floor_pct") for item in eligible
        ],
        "selected": selected,
        "selection_rule": (
            "基准、1.5倍和2倍成本均须满足最低交易数、正复利、正年化复利、"
            "正平均单笔净收益、盈利因子不低于配置门槛及回撤约束；"
            "合格后按最高成本复利、其余成本复利、最高成本回撤排序"
        ),
    }


def run_market_breadth_filter_study(
    validation_candidates: dict[str, list[dict]],
    validation_diagnostics: dict[str, Any],
    final_candidates: dict[str, list[dict]],
    final_diagnostics: dict[str, Any],
    dates: list[str],
    validation_end: str,
    final_signal_end: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """固定Top2与已冻结退出参数，只检验市场广度整日准入门槛。"""
    configured_floors = sorted(set(float(value) for value in core.配置["市场广度阈值"]))
    if not configured_floors or any(
        not math.isfinite(value) or not 0.0 <= value <= 100.0 for value in configured_floors
    ):
        raise RuntimeError("市场广度阈值配置必须是0至100之间的有限百分比")
    floors: list[float | None] = [None, *configured_floors]
    cost_multipliers = tuple(sorted((1.0, *COST_STRESS_MULTIPLIERS)))

    def evaluate(
        candidates: dict[str, list[dict]],
        diagnostics: dict[str, Any],
        start: str,
        end: str,
        floor: float | None,
        multiplier: float,
    ) -> dict[str, Any]:
        filtered = filter_candidates_by_market_breadth(candidates, diagnostics, floor)
        cost_bps = round(core.往返交易成本基点 * multiplier, 4)
        portfolio = core._run_threshold_portfolio(
            filtered,
            dates,
            start,
            end,
            float(parameters["probability_threshold"]),
            2,
            target_net_pct=float(parameters["net_target_pct"]),
            stop_loss_pct=float(parameters["stop_loss_pct"]),
            round_trip_cost_bps=cost_bps,
        )
        summary = _portfolio_summary(portfolio)
        summary["market_breadth_floor_pct"] = floor
        summary["cost_multiplier"] = multiplier
        summary["round_trip_cost_bps"] = cost_bps
        summary["passes_cost_gate"] = eligible_operating_point(
            {
                **summary,
                "probability_threshold": float(parameters["probability_threshold"]),
                "net_target_pct": float(parameters["net_target_pct"]),
                "stop_loss_pct": float(parameters["stop_loss_pct"]),
            },
            minimum_trades=core.最低验证交易数,
            max_drawdown_floor_pct=core.退出参数最大回撤门槛,
        )
        return summary

    grid = []
    for floor in floors:
        rows = [
            evaluate(
                validation_candidates,
                validation_diagnostics,
                VALIDATION_START,
                validation_end,
                floor,
                multiplier,
            )
            for multiplier in cost_multipliers
        ]
        grid.append({
            "market_breadth_floor_pct": floor,
            "validation_by_cost": rows,
            "all_costs_passed": all(row["passes_cost_gate"] for row in rows),
        })
        print(
            f"[市场广度过滤] 门槛={'不启用' if floor is None else f'{floor:.1f}%'}；"
            f"基准/压力成本合格={sum(row['passes_cost_gate'] for row in rows)}/{len(rows)}",
            flush=True,
        )

    selection = select_market_breadth_floor(
        grid,
        minimum_trades=core.最低验证交易数,
        max_drawdown_floor_pct=core.退出参数最大回撤门槛,
    )
    selected = selection["selected"]
    selected_floor = selected.get("market_breadth_floor_pct") if selected else None

    # 补充期只评估已由验证期冻结的门槛，并与无过滤基准对照；不参与选择。
    final_floors: list[float | None] = [None]
    if selected is not None and selected_floor is not None:
        final_floors.append(float(selected_floor))
    final_comparison = []
    for floor in final_floors:
        rows = [
            evaluate(
                final_candidates,
                final_diagnostics,
                FINAL_START,
                final_signal_end,
                floor,
                multiplier,
            )
            for multiplier in cost_multipliers
        ]
        final_comparison.append({
            "market_breadth_floor_pct": floor,
            "role": "无过滤基准" if floor is None else "验证期冻结的市场广度过滤",
            "final_supplemental_by_cost": rows,
        })

    return {
        "status": (
            "no_eligible_validation_setting" if selected is None
            else "validation_selected_no_filter" if selected_floor is None
            else "validation_selected_market_breadth_filter"
        ),
        "fixed_parameters": {
            "top_k_per_signal_day": 2,
            "probability_threshold": float(parameters["probability_threshold"]),
            "net_target_pct": float(parameters["net_target_pct"]),
            "stop_loss_pct": float(parameters["stop_loss_pct"]),
        },
        "validation_signal_window": [VALIDATION_START, validation_end],
        "final_supplemental_signal_window": [FINAL_START, final_signal_end],
        "configured_market_breadth_floors_pct": configured_floors,
        "cost_multipliers": list(cost_multipliers),
        "validation_gate_grid": grid,
        "selection": selection,
        "selected_market_breadth_floor_pct": selected_floor if selected is not None else None,
        "final_supplemental_comparison": final_comparison,
        "audit": {
            "market_breadth_uses_T_day_close_only": True,
            "candidate_rank_and_exit_parameters_frozen_before_market_gate_experiment": True,
            "market_gate_selected_on_validation_only": True,
            "final_supplemental_used_for_selection": False,
            "final_supplemental_is_fresh_unseen_holdout": False,
            "automatic_ordering": False,
            "formal_production_changed": False,
        },
    }


def run_research() -> dict[str, Any]:
    started = time.time()
    if not COST_STRESS_MULTIPLIERS or any(
        not math.isfinite(value) or value <= 1.0 for value in COST_STRESS_MULTIPLIERS
    ):
        raise RuntimeError("成本压力倍率必须为非空、有限且大于1的配置")
    if len(set(COST_STRESS_MULTIPLIERS)) != len(COST_STRESS_MULTIPLIERS):
        raise RuntimeError("成本压力倍率配置不得重复")
    files = history_files()
    dates = [path.name[:10] for path in files]
    date_set = set(dates)
    required_dates = (TRAIN_END, VALIDATION_START, VALIDATION_END, FINAL_START, FINAL_END)
    missing = [day for day in required_dates if day not in date_set]
    if missing:
        raise RuntimeError("历史行情缺少必要研究边界：" + "、".join(missing))

    final_signal_end = core.complete_signal_window_end(dates, FINAL_END, MAX_FORWARD_SESSIONS)
    if dates.index(final_signal_end) < dates.index(FINAL_START):
        raise RuntimeError("历史行情不足以覆盖最终补充期内至少一个完整退出窗口")
    validation_start_index = dates.index(VALIDATION_START)
    final_start_index = dates.index(FINAL_START)
    validation_safe_end_index = final_start_index - MAX_FORWARD_SESSIONS - 1
    if validation_safe_end_index < validation_start_index:
        raise RuntimeError("验证期在清除跨期标签后没有足够的完整退出窗口")
    validation_safe_end = dates[validation_safe_end_index]

    print("[步骤1/5] 使用2015年至2022年训练数据建立路径感知模型", flush=True)
    x, y, processed_days, raw_samples = collect_training(
        files, dates, TRAIN_END, validation_start_index
    )
    if len(x) == 0 or len(set(y.tolist())) < 2:
        raise RuntimeError("训练样本为空或只包含单一类别，终止参数研究")
    model = core._build_model()
    model.fit(x, y)
    print(
        f"[步骤1/5] 训练完成：处理{processed_days}个交易日，"
        f"原始样本{raw_samples}，保留样本{len(y)}",
        flush=True,
    )

    print("[步骤2/5] 生成验证期可执行候选排名缓存", flush=True)
    validation_candidates, validation_diagnostics = core._collect_ranked_candidates(
        files, dates, model, VALIDATION_START, validation_safe_end, max(TOP_K_VALUES)
    )

    print("[步骤3/5] 为Top1与Top2各自独立选择验证期参数", flush=True)
    studies = independently_select_parameters(
        validation_candidates,
        validation_diagnostics,
        dates,
        VALIDATION_START,
        validation_safe_end,
    )

    print("[步骤3.5/5] 在同一验证期检查基准成本与成本压力场景", flush=True)
    robust_studies: dict[int, dict[str, Any]] = {}
    for top_k, study in studies.items():
        sweeps_by_cost: dict[float, list[dict[str, Any]]] = {
            1.0: study["sweep"],
        }
        for multiplier in COST_STRESS_MULTIPLIERS:
            cost_bps = round(core.往返交易成本基点 * multiplier, 4)
            print(
                f"[成本稳健选参] Top{top_k}开始{multiplier:.2f}倍成本网格，"
                f"往返成本{cost_bps:.4f}基点",
                flush=True,
            )
            sweeps_by_cost[float(multiplier)] = core._exit_parameter_sweep(
                validation_candidates,
                validation_diagnostics,
                dates,
                VALIDATION_START,
                validation_safe_end,
                top_k,
                round_trip_cost_bps=cost_bps,
            )
        robust_studies[top_k] = select_cost_robust_operating_point(
            sweeps_by_cost,
            minimum_trades=core.最低验证交易数,
            max_drawdown_floor_pct=core.退出参数最大回撤门槛,
        )
        robust = robust_studies[top_k]
        print(
            f"[成本稳健选参] Top{top_k}：所有成本场景共同达标组合"
            f"{robust['eligible_parameter_count']}组；"
            f"选中={'是' if robust['selected_parameters'] is not None else '否'}",
            flush=True,
        )

    selected_any = any(study["selected"] is not None for study in studies.values())
    final_candidates = None
    final_diagnostics = None
    if selected_any:
        print("[步骤4/5] 生成历史补充期排名缓存；不使用补充期选择参数", flush=True)
        final_candidates, final_diagnostics = core._collect_ranked_candidates(
            files, dates, model, FINAL_START, final_signal_end, max(TOP_K_VALUES)
        )

    market_breadth_study: dict[str, Any] = {
        "status": "not_evaluated_no_cost_robust_top2_parameters",
        "audit": {"formal_production_changed": False, "automatic_ordering": False},
    }
    frozen_top2 = robust_studies.get(2, {}).get("selected_parameters")
    if frozen_top2 is not None:
        market_breadth_study = run_market_breadth_filter_study(
            validation_candidates,
            validation_diagnostics,
            final_candidates,
            final_diagnostics,
            dates,
            validation_safe_end,
            final_signal_end,
            frozen_top2,
        )

    outcomes: dict[str, Any] = {}
    for top_k, study in studies.items():
        selected = study["selected"]
        if selected is None:
            outcomes[str(top_k)] = {
                "status": "no_eligible_validation_operating_point",
                "grid_size": study["grid_size"],
                "eligible_parameter_count": study["eligible_parameter_count"],
                "validation_selected_point": None,
                "final_supplemental": {
                    "status": "not_evaluated_no_eligible_validation_operating_point"
                },
                "neighbor_sensitivity": [],
                "cost_stress_sensitivity": [],
                "cost_robust_selection": {
                    "status": "no_eligible_cost_robust_point",
                    "eligible_parameter_count": robust_studies[top_k]["eligible_parameter_count"],
                    "selected_parameters": None,
                    "validation_by_cost": {},
                    "final_supplemental": None,
                    "final_cost_stress_sensitivity": [],
                    "selection_rule": robust_studies[top_k]["selection_rule"],
                },
            }
            continue

        validation_row = _find_grid_row(study["sweep"], selected)
        final_portfolio = _evaluate_final(
            final_candidates, dates, final_signal_end, top_k, selected
        )
        cost_stress_rows = []
        if any(not math.isfinite(multiplier) or multiplier <= 1.0 for multiplier in COST_STRESS_MULTIPLIERS):
            raise RuntimeError("成本压力倍率必须是有限且大于1的数值")
        if len(set(COST_STRESS_MULTIPLIERS)) != len(COST_STRESS_MULTIPLIERS):
            raise RuntimeError("成本压力倍率不得重复")
        for multiplier in COST_STRESS_MULTIPLIERS:
            stressed_cost_bps = round(core.往返交易成本基点 * multiplier, 4)
            stressed_portfolio = _evaluate_final(
                final_candidates,
                dates,
                final_signal_end,
                top_k,
                selected,
                round_trip_cost_bps=stressed_cost_bps,
            )
            stressed_summary = _portfolio_summary(stressed_portfolio)
            stressed_summary["cost_multiplier"] = multiplier
            stressed_summary["round_trip_cost_bps"] = stressed_cost_bps
            stressed_summary["passes_stress_gate"] = (
                int(stressed_summary.get("completed_trades") or 0) > 0
                and (stressed_summary.get("compound_return_pct") or 0.0) > 0.0
                and (stressed_summary.get("mean_trade_net_return_pct") or 0.0) > 0.0
                and (stressed_summary.get("profit_factor") or 0.0) >= MINIMUM_PROFIT_FACTOR
                and (stressed_summary.get("max_drawdown_pct") or -100.0) >= core.退出参数最大回撤门槛
            )
            cost_stress_rows.append(stressed_summary)
        neighbor_rows = []
        neighbor_points = adjacent_operating_points(
            selected, core.阈值网格, core.止盈目标网格, core.止损幅度网格
        )
        for point in neighbor_points:
            validation_neighbor = _find_grid_row(study["sweep"], point)
            if validation_neighbor is None:
                continue
            final_neighbor = _evaluate_final(
                final_candidates, dates, final_signal_end, top_k, point
            )
            neighbor_rows.append({
                "parameters": point,
                "validation": _metric_summary(validation_neighbor),
                "final_supplemental": _portfolio_summary(final_neighbor),
            })

        robust_study = robust_studies[top_k]
        robust_point = robust_study["selected_parameters"]
        robust_validation_summary = {
            f"{float(multiplier):.1f}倍基准成本": _metric_summary(row)
            for multiplier, row in robust_study["validation_by_cost"].items()
        }
        robust_final_portfolio = None
        robust_final_stress_rows = []
        robust_parameter_summary = None
        if robust_point is not None:
            robust_parameter_summary = {
                "probability_threshold": float(robust_point["probability_threshold"]),
                "net_target_pct": float(robust_point["net_target_pct"]),
                "stop_loss_pct": float(robust_point["stop_loss_pct"]),
            }
            robust_final_portfolio = _portfolio_summary(
                _evaluate_final(final_candidates, dates, final_signal_end, top_k, robust_point)
            )
            for multiplier in COST_STRESS_MULTIPLIERS:
                stressed_cost_bps = round(core.往返交易成本基点 * multiplier, 4)
                stressed_portfolio = _evaluate_final(
                    final_candidates,
                    dates,
                    final_signal_end,
                    top_k,
                    robust_point,
                    round_trip_cost_bps=stressed_cost_bps,
                )
                stressed_summary = _portfolio_summary(stressed_portfolio)
                stressed_summary["cost_multiplier"] = multiplier
                stressed_summary["passes_stress_gate"] = (
                    int(stressed_summary.get("completed_trades") or 0) > 0
                    and (stressed_summary.get("compound_return_pct") or 0.0) > 0.0
                    and (stressed_summary.get("mean_trade_net_return_pct") or 0.0) > 0.0
                    and (stressed_summary.get("profit_factor") or 0.0) >= MINIMUM_PROFIT_FACTOR
                    and (stressed_summary.get("max_drawdown_pct") or -100.0) >= core.退出参数最大回撤门槛
                )
                robust_final_stress_rows.append(stressed_summary)
        cost_robust_summary = {
            "status": "selected_on_validation_all_costs_then_frozen"
            if robust_point is not None else "no_eligible_cost_robust_point",
            "eligible_parameter_count": robust_study["eligible_parameter_count"],
            "selection_rule": robust_study["selection_rule"],
            "selected_parameters": robust_parameter_summary,
            "validation_by_cost": robust_validation_summary,
            "final_supplemental": robust_final_portfolio,
            "final_cost_stress_sensitivity": robust_final_stress_rows,
        }

        outcomes[str(top_k)] = {
            "status": "selected_on_validation_then_frozen",
            "grid_size": study["grid_size"],
            "eligible_parameter_count": study["eligible_parameter_count"],
            "minimum_validation_trades": core.最低验证交易数,
            "maximum_validation_drawdown_floor_pct": core.退出参数最大回撤门槛,
            "selection_rule": "交易数、正复利、正年化复利、正平均单笔净收益和最大回撤均达标后，按验证期复利优先排序",
            "validation_selected_point": _metric_summary(validation_row),
            "final_supplemental": _portfolio_summary(final_portfolio),
            "cost_stress_sensitivity": cost_stress_rows,
            "strict_validation_gate": {
                "minimum_profit_factor": MINIMUM_PROFIT_FACTOR,
                "validation_profit_factor": validation_row.get("profit_factor") if validation_row else None,
                "validation_profit_factor_passed": bool(
                    validation_row
                    and _finite_number(validation_row, "profit_factor") is not None
                    and float(validation_row["profit_factor"]) >= MINIMUM_PROFIT_FACTOR
                ),
                "final_supplemental_profit_factor": final_portfolio.get("profit_factor"),
                "final_supplemental_profit_factor_passed": bool(
                    final_portfolio.get("profit_factor") is not None
                    and float(final_portfolio["profit_factor"]) >= MINIMUM_PROFIT_FACTOR
                ),
                "all_cost_stress_cases_passed": bool(cost_stress_rows) and all(
                    row["passes_stress_gate"] for row in cost_stress_rows
                ),
            },
            "neighbor_sensitivity": neighbor_rows,
            "cost_robust_selection": cost_robust_summary,
            "parameter_selection_status": selected.get("selection_status"),
        }
        print(
            f"[步骤5/5] Top{top_k}冻结参数：阈值{selected['probability_threshold']:.3f}、"
            f"净止盈{selected['net_target_pct']:.3f}%、止损{selected['stop_loss_pct']:.3f}%；"
            f"验证复利{selected['compound_return_pct']:.4f}%，"
            f"补充期复利{final_portfolio.get('compound_return_pct')}%",
            flush=True,
        )

    return {
        "schema_version": 1,
        "status": "research_only",
        "experiment": "candidate_count_specific_validation_parameter_selection",
        "source_public_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "model": {
            "type": "HistGradientBoostingClassifier",
            "seed": 42,
            "training_feature_end": TRAIN_END,
            "training_processed_days": processed_days,
            "training_raw_samples": raw_samples,
            "training_samples": int(len(y)),
        },
        "splits": {
            "train_feature_end": TRAIN_END,
            "validation_signal_window": [VALIDATION_START, validation_safe_end],
            "validation_nominal_end": VALIDATION_END,
            "final_supplemental_signal_window": [FINAL_START, final_signal_end],
            "final_nominal_end": FINAL_END,
        },
        "shared_execution_assumptions": {
            "top_k_values": list(TOP_K_VALUES),
            "round_trip_cost_bps": core.往返交易成本基点,
            "signal": "T日收盘",
            "entry": "T+1开盘",
            "earliest_exit": "T+2",
            "maximum_holding_sessions": MAX_FORWARD_SESSIONS,
            "stop_first_when_both_barriers_hit": True,
            "cash_constrained_no_leverage": True,
            "candidate_ranking_and_execution_filter_shared": True,
        },
        "candidate_count_studies": outcomes,
        "market_breadth_filter_study": market_breadth_study,
        "audit": {
            "top1_and_top2_parameters_selected_independently": True,
            "only_validation_period_used_for_parameter_selection": True,
            "final_supplemental_used_for_parameter_selection": False,
            "final_supplemental_is_fresh_unseen_holdout": False,
            "neighbor_sensitivity_reported": True,
            "validation_selection_profit_factor_floor": MINIMUM_PROFIT_FACTOR,
            "cost_stress_multipliers": list(COST_STRESS_MULTIPLIERS),
            "historical_cost_stress_is_not_forward_validation": True,
            "cost_robust_operating_point_selected_on_validation_only": True,
            "all_cost_robust_gate_scenarios_required": True,
            "formal_production_changed": False,
            "automatic_ordering": False,
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "note": "两个候选数量分别使用相同验证网格选参，仍有多重比较与历史窗口偏差；历史补充期此前已被查看，只作补充审查，不可替代未来影子模拟。",
    }


def main() -> None:
    output_path = DEFAULT_OUTPUT
    if len(sys.argv) == 3 and sys.argv[1] == "--output":
        output_path = Path(sys.argv[2])
    elif len(sys.argv) != 1:
        raise SystemExit("用法：python scripts/candidate_count_operating_point_research.py [--output 输出路径]")

    result = run_research()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    compact = {
        "status": result["status"],
        "source_public_commit": result["source_public_commit"],
        "elapsed_seconds": result["elapsed_seconds"],
        "candidate_count_studies": {
            key: {
                "status": value["status"],
                "eligible_parameter_count": value.get("eligible_parameter_count"),
                "validation_selected_point": value.get("validation_selected_point"),
                "final_supplemental": value.get("final_supplemental"),
                "cost_stress_sensitivity": value.get("cost_stress_sensitivity"),
                "strict_validation_gate": value.get("strict_validation_gate"),
                "cost_robust_selection": value.get("cost_robust_selection"),
                "neighbor_sensitivity": value.get("neighbor_sensitivity"),
            }
            for key, value in result["candidate_count_studies"].items()
        },
        "market_breadth_filter_study": result.get("market_breadth_filter_study"),
    }
    print(json.dumps(compact, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
