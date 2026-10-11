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
) -> dict:
    """按冻结阈值过滤缓存候选，并按完整研究窗口计算收益与风险。"""
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
        目标净收益百分比,
        止损幅度百分比,
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
        "net_target_pct": 目标净收益百分比,
        "stop_loss_pct": 止损幅度百分比,
        "round_trip_cost_bps": 往返交易成本基点,
        "annualization_includes_full_signal_window": True,
    }
    return portfolio


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

    final_diagnostics = None
    final_portfolio = None
    if selected_threshold is not None:
        threshold = float(selected_threshold["probability_threshold"])
        final_candidates, final_diagnostics = _collect_ranked_candidates(
            files, dates, model, FINAL_START, final_signal_end, 每日最多候选数
        )
        final_portfolio = _run_threshold_portfolio(
            final_candidates,
            dates,
            FINAL_START,
            final_signal_end,
            threshold,
            每日最多候选数,
        )
        final_portfolio["selection_status"] = "阈值仅由验证集选择；最终留出未用于选参"
    else:
        final_portfolio = {
            "status": "not_evaluated_no_eligible_validation_threshold",
            "reason": "验证集没有同时满足最低交易数、正复利和最大回撤门槛的阈值；不在最终留出集反向挑参",
            "formal_production_changed": False,
        }

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
                "final_holdout_portfolio": final_portfolio,
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
