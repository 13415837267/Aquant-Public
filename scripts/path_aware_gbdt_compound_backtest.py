"""路径感知 GBDT 的固定阈值可执行复利复核；仅用于研究，不改变生产模型。"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compound_portfolio_backtest import 初始资金, candidate_bar_path, 运行组合, 选择概率阈值
from scripts.selection_factor_catalog import load_research_config
from scripts.short_term_research import (
    FEATURE_WARMUP_SESSIONS,
    FeatureState,
    executable_entry_mask,
    history_files,
    read_daily,
)
from scripts.train_path_aware_gbdt import (
    FINAL_END,
    FINAL_START,
    MAX_FORWARD_SESSIONS,
    SEED,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    collect_training,
)
from scripts.train_short_term_model import make_features

默认输出 = ROOT / "data/backtest/path_aware_gbdt_compound_latest.json"
固定概率阈值 = 0.78
每日最多候选数 = 2
配置 = load_research_config()
目标净收益百分比 = float(配置["短线目标净收益百分比"])
止损幅度百分比 = float(配置["止损幅度百分比"])
往返交易成本基点 = float(配置["往返交易成本基点"])
阈值网格 = tuple(sorted(set(round(float(x), 3) for x in 配置["复利入场概率阈值网格"]) | {固定概率阈值}))
最低验证交易数 = int(配置["参数选择最低交易数"])
最大允许回撤百分比 = float(配置["参数选择最大允许回撤百分比"])
止盈目标网格 = tuple(sorted(set(round(float(x), 3) for x in 配置["复利止盈目标网格百分比"])))
止损幅度网格 = tuple(sorted(set(round(float(x), 3) for x in 配置["复利止损幅度网格百分比"])))
退出参数最大回撤门槛 = float(配置["复利止盈止损参数最大允许回撤百分比"])


def rank_probability_candidates(
    probabilities: np.ndarray,
    executable_mask: np.ndarray,
    threshold: float,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """只按信号日概率与 T+1 可执行掩码排序；不读取未来收益或路径完整性。"""
    scores = np.asarray(probabilities, dtype=np.float64)
    executable = np.asarray(executable_mask, dtype=bool)
    if scores.ndim != 1 or executable.ndim != 1 or scores.shape != executable.shape:
        raise ValueError("模型概率与可执行掩码必须是长度一致的一维数组")
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("固定概率阈值必须在 0 至 1 之间")
    if int(top_k) < 1:
        raise ValueError("每日最多候选数必须为正整数")

    accepted = executable & np.isfinite(scores) & (scores >= float(threshold))
    indices = np.flatnonzero(accepted)
    ranked = indices[np.argsort(-scores[indices], kind="stable")]
    return ranked[: int(top_k)], accepted


def complete_signal_window_end(
    dates: list[str], nominal_end: str, max_forward_sessions: int
) -> str:
    """将信号截止日限制在数据仍覆盖完整退出窗口的最后一个交易日。"""
    if nominal_end not in dates:
        raise ValueError(f"名义截止日不在交易日索引中：{nominal_end}")
    if int(max_forward_sessions) < 1:
        raise ValueError("最大前瞻交易日数必须为正整数")
    last_complete_index = len(dates) - int(max_forward_sessions) - 1
    if last_complete_index < 0:
        raise ValueError("历史行情不足以覆盖一个完整退出窗口")
    return min(nominal_end, dates[last_complete_index])


def _build_model() -> HistGradientBoostingClassifier:
    """与路径感知研究基准使用相同的模型结构和训练参数。"""
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        class_weight="balanced",
        random_state=SEED,
    )


def _symbol_keys(values) -> np.ndarray:
    return (
        pd.Series(values, dtype="string")
        .str.extract(r"(\d{6})", expand=False)
        .fillna("")
        .str.zfill(6)
        .to_numpy(dtype=str)
    )


def _cache_reader(files: list[Path]):
    cache: dict[int, pd.DataFrame] = {}

    def read(index: int) -> pd.DataFrame:
        if index not in cache:
            cache[index] = read_daily(files[index])
        while len(cache) > MAX_FORWARD_SESSIONS + 2:
            del cache[next(iter(cache))]
        return cache[index]

    return read



def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.3f}"


def select_validation_threshold(rows: list[dict]) -> dict | None:
    """仅按验证集收益、最低交易数与最大回撤门槛选择阈值。"""
    return 选择概率阈值(rows, 最低验证交易数, 最大允许回撤百分比)


def select_validation_exit_operating_point(
    rows: list[dict],
    minimum_trades: int,
    max_drawdown_floor_pct: float,
) -> dict | None:
    """只按验证期筛选入场概率、止盈与止损；最终期不得参与选参。"""
    eligible = []
    for row in rows:
        try:
            trades = int(row.get("completed_trades") or 0)
            threshold = float(row["probability_threshold"])
            target = float(row["net_target_pct"])
            stop_loss = float(row["stop_loss_pct"])
            compound = float(row["compound_return_pct"])
            annualized = float(row["annualized_compound_return_pct"])
            drawdown = float(row["max_drawdown_pct"])
            mean_trade = float(row["mean_trade_net_return_pct"])
        except (KeyError, TypeError, ValueError):
            continue
        values = (threshold, target, stop_loss, compound, annualized, drawdown, mean_trade)
        if not all(math.isfinite(value) for value in values):
            continue
        if not 0.0 <= threshold <= 1.0 or target <= 0.0 or stop_loss <= 0.0:
            continue
        if trades < minimum_trades:
            continue
        if compound <= 0.0 or annualized <= 0.0 or mean_trade <= 0.0:
            continue
        if drawdown < max_drawdown_floor_pct:
            continue
        eligible.append(dict(row))

    if not eligible:
        return None
    selected = max(
        eligible,
        key=lambda row: (
            float(row["compound_return_pct"]),
            float(row["annualized_compound_return_pct"]),
            float(row["mean_trade_net_return_pct"]),
            float(row["max_drawdown_pct"]),
            int(row["completed_trades"]),
        ),
    )
    selected["selection_status"] = "仅由验证期选择；联合网格存在多重比较偏差"
    return selected


def _collect_ranked_candidates(
    files: list[Path],
    dates: list[str],
    model: HistGradientBoostingClassifier,
    start_date: str,
    end_date: str,
    top_k: int,
) -> tuple[dict[str, list[dict]], dict]:
    """每个信号日只推理一次，保存当日最高的可执行候选供阈值网格共用。"""
    date_index = {date: index for index, date in enumerate(dates)}
    if start_date not in date_index or end_date not in date_index:
        raise ValueError(f"研究窗口不在历史交易日内：{start_date} 至 {end_date}")
    start_i, end_i = date_index[start_date], date_index[end_date]
    if start_i > end_i:
        raise ValueError(f"研究窗口起始日晚于结束日：{start_date} 至 {end_date}")

    read = _cache_reader(files)
    state = FeatureState()
    daily_candidates: dict[str, list[dict]] = {}
    threshold_diagnostics = {
        _threshold_key(threshold): {
            "probability_threshold": float(threshold),
            "threshold_and_executable_candidates": 0,
            "selected_candidates": 0,
        }
        for threshold in 阈值网格
    }
    diagnostics = {
        "signal_days": 0,
        "universe_samples": 0,
        "executable_entries": 0,
        "missing_or_invalid_entry_prices": 0,
        "thresholds": threshold_diagnostics,
    }

    # 仅使用当前及过去日线构造特征；候选准入不依赖未来路径是否完整。
    for i in range(max(0, start_i - FEATURE_WARMUP_SESSIONS), end_i + 1):
        frame = state.build(read(i))
        signal_date = dates[i]
        if signal_date < start_date or signal_date > end_date or frame.empty:
            continue
        if i + MAX_FORWARD_SESSIONS >= len(files):
            continue

        symbols = _symbol_keys(frame["symbol"])
        probabilities = model.predict_proba(make_features(frame))[:, 1]
        future_days = [read(i + offset) for offset in range(1, MAX_FORWARD_SESSIONS + 1)]
        future_dates = dates[i + 1 : i + MAX_FORWARD_SESSIONS + 1]
        entry_day = future_days[0]
        executable = np.asarray(executable_entry_mask(symbols, entry_day), dtype=bool)
        finite_executable = executable & np.isfinite(probabilities)

        diagnostics["signal_days"] += 1
        diagnostics["universe_samples"] += int(len(symbols))
        diagnostics["executable_entries"] += int(executable.sum())
        for threshold in 阈值网格:
            key = _threshold_key(threshold)
            threshold_diagnostics[key]["threshold_and_executable_candidates"] += int(
                (finite_executable & (probabilities >= threshold)).sum()
            )

        # 先缓存每日排名前 top_k 的可执行候选，再在同一批候选上应用各阈值。
        # 这与“先按阈值过滤、再取 top_k”完全等价，因为排名是按概率降序。
        selected_indices, _ = rank_probability_candidates(
            probabilities, executable, threshold=0.0, top_k=top_k
        )
        candidates: list[dict] = []
        if len(selected_indices):
            entry_frame = entry_day.copy()
            entry_frame["_symbol_key"] = _symbol_keys(entry_frame["symbol"])
            entry_frame = entry_frame.drop_duplicates("_symbol_key", keep="last").set_index("_symbol_key")
            open_prices = pd.to_numeric(entry_frame["open"], errors="coerce").reindex(symbols).to_numpy(dtype=float)

            for row_index in selected_indices:
                symbol = str(symbols[row_index])
                entry_price = float(open_prices[row_index])
                if not math.isfinite(entry_price) or entry_price <= 0:
                    diagnostics["missing_or_invalid_entry_prices"] += 1
                    continue
                bars = candidate_bar_path(symbol, future_days, future_dates, entry_price)
                if not bars or not bars[0].get("bar_available", False):
                    raise RuntimeError(f"T+1 入场掩码与候选路径不一致：{signal_date} {symbol}")
                candidates.append({
                    "signal_date": signal_date,
                    "symbol": symbol,
                    "probability": float(probabilities[row_index]),
                    "entry_date": bars[0]["date"],
                    "entry_price": entry_price,
                    "bars": bars,
                })

        daily_candidates[signal_date] = candidates
        for threshold in 阈值网格:
            key = _threshold_key(threshold)
            threshold_diagnostics[key]["selected_candidates"] += sum(
                candidate["probability"] >= threshold for candidate in candidates
            )

        if diagnostics["signal_days"] % 50 == 0:
            baseline = threshold_diagnostics[_threshold_key(固定概率阈值)]
            print(
                f"[路径感知可执行复核] {start_date} 至 {end_date}："
                f"已处理{diagnostics['signal_days']}个信号日，"
                f"基准阈值候选{baseline['threshold_and_executable_candidates']}，"
                f"已入选前二候选{baseline['selected_candidates']}",
                flush=True,
            )

    return daily_candidates, diagnostics


def _run_threshold_portfolio(
    daily_candidates: dict[str, list[dict]],
    dates: list[str],
    start_date: str,
    end_date: str,
    threshold: float,
    top_k: int,
    target_net_pct: float | None = None,
    stop_loss_pct: float | None = None,
) -> dict:
    """按冻结参数过滤缓存候选，并按完整研究窗口计算收益与风险。"""
    target_net = 目标净收益百分比 if target_net_pct is None else float(target_net_pct)
    stop_loss = 止损幅度百分比 if stop_loss_pct is None else float(stop_loss_pct)
    filtered_candidates = {
        signal_date: [
            candidate for candidate in candidates
            if candidate["probability"] >= float(threshold)
        ]
        for signal_date, candidates in daily_candidates.items()
    }
    date_index = {date: index for index, date in enumerate(dates)}
    end_index = date_index[end_date] + MAX_FORWARD_SESSIONS
    if end_index >= len(dates):
        raise ValueError(f"研究窗口末端缺少完整退出行情：{end_date}")
    portfolio = 运行组合(
        filtered_candidates,
        dates,
        "每日排名",
        target_net,
        stop_loss,
        往返交易成本基点,
        top_k=top_k,
        窗口起始日期=start_date,
        窗口结束日期=dates[end_index],
    )
    portfolio["probability_threshold"] = float(threshold)
    portfolio["measurement_window"] = [start_date, dates[end_index]]
    portfolio["signal_window"] = [start_date, end_date]
    portfolio["selection_policy"] = {
        "fixed_probability_threshold": float(threshold),
        "top_k_per_signal_day": int(top_k),
        "signal": "T日收盘",
        "entry": "T+1开盘",
        "earliest_exit": "T+2",
        "maximum_holding_sessions": MAX_FORWARD_SESSIONS,
        "net_target_pct": target_net,
        "stop_loss_pct": stop_loss,
        "round_trip_cost_bps": 往返交易成本基点,
        "annualization_includes_full_signal_window": True,
    }
    return portfolio


def compare_candidate_counts(
    daily_candidates: dict[str, list[dict]],
    dates: list[str],
    start_date: str,
    end_date: str,
    threshold: float,
    target_net_pct: float,
    stop_loss_pct: float,
    period_label: str,
    parameter_source: str,
    top_ks: tuple[int, ...] = (1, 2),
) -> list[dict]:
    """在同一候选排序与冻结参数下，只比较每日最多入选数。"""
    if (
        not top_ks
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in top_ks)
        or len(set(top_ks)) != len(top_ks)
    ):
        raise ValueError("候选数对照必须是无重复的正整数序列")

    rows = []
    for top_k in top_ks:
        portfolio = _run_threshold_portfolio(
            daily_candidates,
            dates,
            start_date,
            end_date,
            threshold,
            top_k,
            target_net_pct=target_net_pct,
            stop_loss_pct=stop_loss_pct,
        )
        rows.append(
            {
                "period": period_label,
                "top_k_per_signal_day": top_k,
                "probability_threshold": float(threshold),
                "net_target_pct": float(target_net_pct),
                "stop_loss_pct": float(stop_loss_pct),
                "parameter_source": parameter_source,
                "status": portfolio.get("status"),
                "requested_entries": portfolio.get("requested_entries"),
                "entered_trades": portfolio.get("entered_trades"),
                "completed_trades": int(portfolio.get("completed_trades") or 0),
                "compound_return_pct": portfolio.get("compound_return_pct"),
                "annualized_compound_return_pct": portfolio.get("annualized_compound_return_pct"),
                "max_drawdown_pct": portfolio.get("max_drawdown_pct"),
                "target_hit_rate_pct": portfolio.get("target_hit_rate_pct"),
                "positive_trade_rate_pct": portfolio.get("positive_trade_rate_pct"),
                "mean_trade_net_return_pct": portfolio.get("mean_trade_net_return_pct"),
                "skipped_for_capacity_or_cash": portfolio.get("skipped_for_capacity_or_cash"),
                "open_positions_at_end": portfolio.get("open_positions_at_end"),
                "formal_production_changed": False,
            }
        )
    return rows

def _threshold_sweep(
    daily_candidates: dict[str, list[dict]],
    diagnostics: dict,
    dates: list[str],
    start_date: str,
    end_date: str,
    top_k: int,
) -> list[dict]:
    rows = []
    for threshold in 阈值网格:
        portfolio = _run_threshold_portfolio(
            daily_candidates, dates, start_date, end_date, threshold, top_k
        )
        item = {
            "probability_threshold": float(threshold),
            **diagnostics["thresholds"][_threshold_key(threshold)],
            "status": portfolio.get("status"),
            "completed_trades": int(portfolio.get("completed_trades") or 0),
            "compound_return_pct": portfolio.get("compound_return_pct"),
            "annualized_compound_return_pct": portfolio.get("annualized_compound_return_pct"),
            "max_drawdown_pct": portfolio.get("max_drawdown_pct"),
            "portfolio": portfolio,
        }
        rows.append(item)
    return rows


def _exit_parameter_sweep(
    daily_candidates: dict[str, list[dict]],
    diagnostics: dict,
    dates: list[str],
    start_date: str,
    end_date: str,
    top_k: int,
) -> list[dict]:
    """在验证期枚举概率阈值×净止盈×止损；最终期不进入评分或排序。"""
    rows = []
    total = len(阈值网格) * len(止盈目标网格) * len(止损幅度网格)
    completed = 0
    for threshold in 阈值网格:
        for target in 止盈目标网格:
            for stop_loss in 止损幅度网格:
                portfolio = _run_threshold_portfolio(
                    daily_candidates,
                    dates,
                    start_date,
                    end_date,
                    threshold,
                    top_k,
                    target_net_pct=target,
                    stop_loss_pct=stop_loss,
                )
                rows.append({
                    "probability_threshold": float(threshold),
                    "net_target_pct": float(target),
                    "stop_loss_pct": float(stop_loss),
                    **diagnostics["thresholds"][_threshold_key(threshold)],
                    "status": portfolio.get("status"),
                    "completed_trades": int(portfolio.get("completed_trades") or 0),
                    "compound_return_pct": portfolio.get("compound_return_pct"),
                    "annualized_compound_return_pct": portfolio.get("annualized_compound_return_pct"),
                    "max_drawdown_pct": portfolio.get("max_drawdown_pct"),
                    "mean_trade_net_return_pct": portfolio.get("mean_trade_net_return_pct"),
                    "target_hit_rate_pct": portfolio.get("target_hit_rate_pct"),
                    "positive_trade_rate_pct": portfolio.get("positive_trade_rate_pct"),
                    "exit_reason_counts": portfolio.get("exit_reason_counts", {}),
                })
                completed += 1
                if completed % 100 == 0 or completed == total:
                    print(
                        f"[验证期出场参数实验] 已完成组合 {completed}/{total}",
                        flush=True,
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="验证集阈值网格与可执行复利研究")
    parser.add_argument("--output", default=str(默认输出))
    args = parser.parse_args()

    started = time.time()
    if round(固定概率阈值, 3) not in 阈值网格:
        raise SystemExit(f"基准阈值 {固定概率阈值} 不在本轮验证阈值网格中")

    files = history_files()
    dates = [path.name[:10] for path in files]
    required_dates = (TRAIN_END, VALIDATION_START, VALIDATION_END, FINAL_START, FINAL_END)
    missing = [date for date in required_dates if date not in set(dates)]
    if missing:
        raise SystemExit("历史行情缺少研究边界交易日：" + "、".join(missing))
    final_signal_end = complete_signal_window_end(dates, FINAL_END, MAX_FORWARD_SESSIONS)
    if dates.index(final_signal_end) < dates.index(FINAL_START):
        raise SystemExit("历史行情不足以覆盖最终留出区间内至少一个完整退出窗口")

    validation_start_index = dates.index(VALIDATION_START)
    final_start_index = dates.index(FINAL_START)
    validation_safe_end_index = final_start_index - MAX_FORWARD_SESSIONS - 1
    if validation_safe_end_index < validation_start_index:
        raise SystemExit("清除最终留出边界后，验证集没有足够的可执行研究日期")
    validation_safe_end = dates[validation_safe_end_index]

    x, y, train_days, train_samples = collect_training(
        files, dates, TRAIN_END, validation_start_index
    )
    if len(x) == 0 or len(np.unique(y)) < 2:
        raise SystemExit("训练样本不足或缺少正负两类标签，终止可执行回测")
    model = _build_model()
    model.fit(x, y)

    validation_candidates, validation_diagnostics = _collect_ranked_candidates(
        files, dates, model, VALIDATION_START, validation_safe_end, 每日最多候选数
    )
    validation_sweep = _threshold_sweep(
        validation_candidates,
        validation_diagnostics,
        dates,
        VALIDATION_START,
        validation_safe_end,
        每日最多候选数,
    )
    selected_threshold = select_validation_threshold(validation_sweep)

    exit_parameter_sweep = _exit_parameter_sweep(
        validation_candidates,
        validation_diagnostics,
        dates,
        VALIDATION_START,
        validation_safe_end,
        每日最多候选数,
    )
    selected_exit_parameters = select_validation_exit_operating_point(
        exit_parameter_sweep,
        minimum_trades=最低验证交易数,
        max_drawdown_floor_pct=退出参数最大回撤门槛,
    )

    final_diagnostics = None
    final_candidates = None
    final_portfolio = {
        "status": "not_evaluated_no_eligible_validation_threshold",
        "reason": "验证期没有满足既定样本数、正复利和回撤条件的入场阈值；不在最终补充期反向选参",
        "formal_production_changed": False,
    }
    final_exit_portfolio = {
        "status": "not_evaluated_no_eligible_validation_exit_parameters",
        "reason": "验证期没有满足最低交易数、正复利、正平均单笔收益及更严格回撤门槛的联合参数组合",
        "formal_production_changed": False,
    }
    if selected_threshold is not None or selected_exit_parameters is not None:
        final_candidates, final_diagnostics = _collect_ranked_candidates(
            files, dates, model, FINAL_START, final_signal_end, 每日最多候选数
        )

    if final_candidates is None:
        final_candidates, final_diagnostics = _collect_ranked_candidates(
            files, dates, model, FINAL_START, final_signal_end, 每日最多候选数
        )

    if selected_exit_parameters is not None:
        comparison_threshold = float(selected_exit_parameters["probability_threshold"])
        comparison_target = float(selected_exit_parameters["net_target_pct"])
        comparison_stop_loss = float(selected_exit_parameters["stop_loss_pct"])
        comparison_parameter_source = "验证期联合网格选择；参数曾按Top2组合排序，候选数对照仅作敏感性分析"
    elif selected_threshold is not None:
        comparison_threshold = float(selected_threshold["probability_threshold"])
        comparison_target = 目标净收益百分比
        comparison_stop_loss = 止损幅度百分比
        comparison_parameter_source = "入场阈值由验证期选择；止盈止损为固定研究基准"
    else:
        comparison_threshold = 固定概率阈值
        comparison_target = 目标净收益百分比
        comparison_stop_loss = 止损幅度百分比
        comparison_parameter_source = "预设固定基准参数；本轮未在最终补充期选参"

    candidate_count_experiment = {
        "status": "research_only",
        "method": "同一每日概率排名下，对照每个信号日最多入选1只与2只；现金约束、交易时序、成本、止盈止损与阈值保持一致",
        "top_k_values": [1, 2],
        "fixed_reference": {
            "parameters": {
                "probability_threshold": 固定概率阈值,
                "net_target_pct": 目标净收益百分比,
                "stop_loss_pct": 止损幅度百分比,
                "source": "脚本预设基准，不根据本轮Top1/Top2结果调整",
            },
            "validation": compare_candidate_counts(
                validation_candidates, dates, VALIDATION_START, validation_safe_end,
                固定概率阈值, 目标净收益百分比, 止损幅度百分比,
                "验证期", "预设固定基准参数",
            ),
            "final_supplemental": compare_candidate_counts(
                final_candidates, dates, FINAL_START, final_signal_end,
                固定概率阈值, 目标净收益百分比, 止损幅度百分比,
                "历史补充期", "预设固定基准参数",
            ),
        },
        "validation_selected_operating_point": {
            "parameters": {
                "probability_threshold": comparison_threshold,
                "net_target_pct": comparison_target,
                "stop_loss_pct": comparison_stop_loss,
                "source": comparison_parameter_source,
            },
            "validation": compare_candidate_counts(
                validation_candidates, dates, VALIDATION_START, validation_safe_end,
                comparison_threshold, comparison_target, comparison_stop_loss,
                "验证期", comparison_parameter_source,
            ),
            "final_supplemental": compare_candidate_counts(
                final_candidates, dates, FINAL_START, final_signal_end,
                comparison_threshold, comparison_target, comparison_stop_loss,
                "历史补充期", comparison_parameter_source,
            ),
        },
        "decision_boundary": "仅用于研究候选数敏感性；不修改Private策略、生产候选上限或正式发布状态。历史补充期此前已被查看，只能用于补充否决，不能视为独立新留出。",
        "audit": {
            "same_ranked_candidates_within_each_period": True,
            "same_operating_parameters_between_top1_and_top2": True,
            "entry_is_T_plus_1_open": True,
            "exit_starts_T_plus_2": True,
            "round_trip_cost_applied": True,
            "cash_constrained_no_leverage": True,
            "final_period_used_for_candidate_count_selection": False,
            "final_period_is_fresh_unseen_holdout": False,
            "formal_production_changed": False,
        },
    }
    if selected_threshold is not None:
        threshold = float(selected_threshold["probability_threshold"])
        final_portfolio = _run_threshold_portfolio(
            final_candidates,
            dates,
            FINAL_START,
            final_signal_end,
            threshold,
            每日最多候选数,
        )
        final_portfolio["selection_status"] = "入场阈值仅由验证期选择；最终补充期未用于选参"

    if selected_exit_parameters is not None:
        final_exit_portfolio = _run_threshold_portfolio(
            final_candidates,
            dates,
            FINAL_START,
            final_signal_end,
            float(selected_exit_parameters["probability_threshold"]),
            每日最多候选数,
            target_net_pct=float(selected_exit_parameters["net_target_pct"]),
            stop_loss_pct=float(selected_exit_parameters["stop_loss_pct"]),
        )
        final_exit_portfolio["selection_status"] = (
            "概率阈值、止盈与止损均由验证期一次冻结；最终期仅作历史补充检验"
        )

    result = {
        "schema_version": 2,
        "status": "research_only",
        "method": "path_aware_gbdt_validation_compound_threshold_sweep",
        "source_public_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "objective": "扣除往返成本后，以现金约束组合检验5个交易日内净收益达到1%的短线机会",
        "model": {
            "type": "HistGradientBoostingClassifier",
            "learning_rate": 0.05,
            "max_iter": 250,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 100,
            "l2_regularization": 1.0,
            "class_weight": "balanced",
            "random_state": SEED,
            "train_feature_end": TRAIN_END,
            "training_samples": int(len(y)),
            "training_positive_rate_pct": round(float(np.mean(y) * 100.0), 4),
            "training_processed_days": int(train_days),
            "training_raw_samples": int(train_samples),
        },
        "selection_policy": {
            "threshold_grid": list(阈值网格),
            "includes_previous_baseline_threshold": 固定概率阈值,
            "minimum_completed_trades": 最低验证交易数,
            "maximum_drawdown_floor_pct": 最大允许回撤百分比,
            "selection_rule": "仅按验证集筛选最低交易数、正累计/年化复利及回撤达标方案，再取累计复利最高者",
            "top_k_per_signal_day": 每日最多候选数,
            "initial_equity": 初始资金,
            "net_target_pct": 目标净收益百分比,
            "stop_loss_pct": 止损幅度百分比,
            "round_trip_cost_bps": 往返交易成本基点,
            "cash_constrained_no_leverage": True,
        },
        "splits": {
            "train_feature_end": TRAIN_END,
            "validation": [VALIDATION_START, validation_safe_end],
            "validation_nominal_end": VALIDATION_END,
            "final_holdout": [FINAL_START, final_signal_end],
            "final_nominal_end": FINAL_END,
            "final_signal_end_truncated_for_complete_exit_window": final_signal_end < FINAL_END,
            "validation_exit_paths_end_before_final_holdout": True,
        },
        "validation": {
            "signal_window": [VALIDATION_START, validation_safe_end],
            "candidate_diagnostics": validation_diagnostics,
            "threshold_sweep": validation_sweep,
            "selected_threshold": selected_threshold,
        },
        "final_holdout": {
            "signal_window": [FINAL_START, final_signal_end],
            "candidate_diagnostics": final_diagnostics,
            "selected_threshold": (
                float(selected_threshold["probability_threshold"])
                if selected_threshold is not None else None
            ),
            "portfolio": final_portfolio,
            "fresh_unseen_holdout": False,
            "note": "最终留出期此前已用于路径标签报告和固定阈值组合复核；本结果属于补充验证，不是从未查看过的新留出集",
        },
        "candidate_count_experiment": candidate_count_experiment,
        "exit_parameter_experiment": {
            "status": "research_only",
            "method": "验证期联合搜索入场概率、净止盈目标与止损幅度；最终期仅评估已冻结参数",
            "selection_policy": {
                "probability_threshold_grid": list(阈值网格),
                "net_target_pct_grid": list(止盈目标网格),
                "stop_loss_pct_grid": list(止损幅度网格),
                "combination_count": len(exit_parameter_sweep),
                "minimum_validation_trades": 最低验证交易数,
                "max_drawdown_floor_pct": 退出参数最大回撤门槛,
                "selection_rule": "验证期至少100笔完成交易、复利与平均单笔收益均为正、最大回撤不差于配置门槛，再按复利收益优先选择",
            },
            "validation": {
                "signal_window": [VALIDATION_START, validation_safe_end],
                "sweep": exit_parameter_sweep,
                "selected_operating_point": selected_exit_parameters,
            },
            "final_supplemental": {
                "signal_window": [FINAL_START, final_signal_end],
                "selected_validation_operating_point": (
                    {
                        "probability_threshold": selected_exit_parameters["probability_threshold"],
                        "net_target_pct": selected_exit_parameters["net_target_pct"],
                        "stop_loss_pct": selected_exit_parameters["stop_loss_pct"],
                    }
                    if selected_exit_parameters is not None else None
                ),
                "portfolio": final_exit_portfolio,
                "fresh_unseen_holdout": False,
            },
            "audit": {
                "final_period_used_for_parameter_selection": False,
                "parameter_combination_frozen_before_final_evaluation": True,
                "joint_grid_selection_has_multiple_comparison_bias": True,
                "final_period_is_supplemental_not_fresh_unseen_holdout": True,
                "formal_production_changed": False,
            },
        },
        "audit": {
            "no_future_features": True,
            "candidate_admission_does_not_use_future_path_completeness": True,
            "entry_is_T_plus_1_open": True,
            "exit_starts_T_plus_2": True,
            "upper_limit_and_paused_entries_blocked_by_shared_execution_mask": True,
            "stop_first_on_same_bar": True,
            "round_trip_cost_applied": True,
            "cash_constrained_no_leverage": True,
            "threshold_selected_on_validation_only": True,
            "full_validation_window_includes_cash_only_dates": True,
            "full_holdout_window_includes_cash_only_dates": True,
            "final_holdout_result_is_supplemental_not_a_fresh_unseen_holdout": True,
            "final_signal_window_requires_complete_forward_exit_bars": True,
            "formal_production_changed": False,
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "threshold_sweep": [
                    {
                        "probability_threshold": row["probability_threshold"],
                        "threshold_and_executable_candidates": row["threshold_and_executable_candidates"],
                        "selected_candidates": row["selected_candidates"],
                        "completed_trades": row["completed_trades"],
                        "compound_return_pct": row["compound_return_pct"],
                        "max_drawdown_pct": row["max_drawdown_pct"],
                    }
                    for row in validation_sweep
                ],
                "selected_validation_threshold": (
                    {
                        "probability_threshold": selected_threshold["probability_threshold"],
                        "completed_trades": selected_threshold["completed_trades"],
                        "compound_return_pct": selected_threshold["compound_return_pct"],
                        "max_drawdown_pct": selected_threshold["max_drawdown_pct"],
                        "selection_status": selected_threshold["selection_status"],
                    }
                    if selected_threshold is not None else None
                ),
                "final_holdout_portfolio": {
                    "status": final_portfolio.get("status"),
                    "completed_trades": final_portfolio.get("completed_trades"),
                    "compound_return_pct": final_portfolio.get("compound_return_pct"),
                    "max_drawdown_pct": final_portfolio.get("max_drawdown_pct"),
                },
                "exit_parameter_grid_size": len(exit_parameter_sweep),
                "selected_validation_exit_parameters": (
                    {
                        "probability_threshold": selected_exit_parameters["probability_threshold"],
                        "net_target_pct": selected_exit_parameters["net_target_pct"],
                        "stop_loss_pct": selected_exit_parameters["stop_loss_pct"],
                        "completed_trades": selected_exit_parameters["completed_trades"],
                        "compound_return_pct": selected_exit_parameters["compound_return_pct"],
                        "max_drawdown_pct": selected_exit_parameters["max_drawdown_pct"],
                    }
                    if selected_exit_parameters is not None else None
                ),
                "final_supplemental_exit_portfolio": {
                    "status": final_exit_portfolio.get("status"),
                    "completed_trades": final_exit_portfolio.get("completed_trades"),
                    "compound_return_pct": final_exit_portfolio.get("compound_return_pct"),
                    "max_drawdown_pct": final_exit_portfolio.get("max_drawdown_pct"),
                    "selection_policy": final_exit_portfolio.get("selection_policy"),
                },
                "candidate_count_experiment": {
                    "fixed_reference_validation": [
                        {
                            "top_k": row["top_k_per_signal_day"],
                            "completed_trades": row["completed_trades"],
                            "compound_return_pct": row["compound_return_pct"],
                            "max_drawdown_pct": row["max_drawdown_pct"],
                            "mean_trade_net_return_pct": row["mean_trade_net_return_pct"],
                        }
                        for row in result["candidate_count_experiment"]["fixed_reference"]["validation"]
                    ],
                    "fixed_reference_final_supplemental": [
                        {
                            "top_k": row["top_k_per_signal_day"],
                            "completed_trades": row["completed_trades"],
                            "compound_return_pct": row["compound_return_pct"],
                            "max_drawdown_pct": row["max_drawdown_pct"],
                            "mean_trade_net_return_pct": row["mean_trade_net_return_pct"],
                        }
                        for row in result["candidate_count_experiment"]["fixed_reference"]["final_supplemental"]
                    ],
                    "validation_selected_point_validation": [
                        {
                            "top_k": row["top_k_per_signal_day"],
                            "completed_trades": row["completed_trades"],
                            "compound_return_pct": row["compound_return_pct"],
                            "max_drawdown_pct": row["max_drawdown_pct"],
                            "mean_trade_net_return_pct": row["mean_trade_net_return_pct"],
                        }
                        for row in result["candidate_count_experiment"]["validation_selected_operating_point"]["validation"]
                    ],
                    "validation_selected_point_final_supplemental": [
                        {
                            "top_k": row["top_k_per_signal_day"],
                            "completed_trades": row["completed_trades"],
                            "compound_return_pct": row["compound_return_pct"],
                            "max_drawdown_pct": row["max_drawdown_pct"],
                            "mean_trade_net_return_pct": row["mean_trade_net_return_pct"],
                        }
                        for row in result["candidate_count_experiment"]["validation_selected_operating_point"]["final_supplemental"]
                    ],
                },
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
