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

from scripts.compound_portfolio_backtest import 初始资金, candidate_bar_path, 运行组合
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
阈值网格 = tuple(round(float(x), 2) for x in 配置["路径模型概率阈值网格"])


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


def _portfolio_window(
    files: list[Path],
    dates: list[str],
    model: HistGradientBoostingClassifier,
    start_date: str,
    end_date: str,
    threshold: float,
    top_k: int,
) -> dict:
    """逐日推理并模拟可执行组合；候选准入不使用未来路径标签。"""
    date_index = {date: index for index, date in enumerate(dates)}
    if start_date not in date_index or end_date not in date_index:
        raise ValueError(f"研究窗口不在历史交易日内：{start_date} 至 {end_date}")
    start_i, end_i = date_index[start_date], date_index[end_date]
    if start_i > end_i:
        raise ValueError(f"研究窗口起始日晚于结束日：{start_date} 至 {end_date}")

    read = _cache_reader(files)
    state = FeatureState()
    daily_candidates: dict[str, list[dict]] = {}
    diagnostics = {
        "signal_days": 0,
        "universe_samples": 0,
        "executable_entries": 0,
        "threshold_and_executable_candidates": 0,
        "selected_candidates": 0,
        "missing_or_invalid_entry_prices": 0,
    }

    # 仅用历史日线预热特征；不调用 build_targets，不根据未来路径是否完整来筛选候选。
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
        executable = executable_entry_mask(symbols, entry_day)
        selected_indices, accepted = rank_probability_candidates(
            probabilities, executable, threshold, top_k
        )

        diagnostics["signal_days"] += 1
        diagnostics["universe_samples"] += int(len(symbols))
        diagnostics["executable_entries"] += int(np.asarray(executable, dtype=bool).sum())
        diagnostics["threshold_and_executable_candidates"] += int(accepted.sum())
        diagnostics["selected_candidates"] += int(len(selected_indices))

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
                candidates.append(
                    {
                        "signal_date": signal_date,
                        "symbol": symbol,
                        "probability": float(probabilities[row_index]),
                        "entry_date": bars[0]["date"],
                        "entry_price": entry_price,
                        "bars": bars,
                    }
                )
        daily_candidates[signal_date] = candidates

        if diagnostics["signal_days"] % 50 == 0:
            print(
                f"[路径感知可执行复核] {start_date} 至 {end_date}："
                f"已处理{diagnostics['signal_days']}个信号日，"
                f"阈值与可执行候选{diagnostics['threshold_and_executable_candidates']}",
                flush=True,
            )

    portfolio = 运行组合(
        daily_candidates,
        dates,
        "每日排名",
        目标净收益百分比,
        止损幅度百分比,
        往返交易成本基点,
        top_k=top_k,
    )
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
    }
    return {
        "signal_window": [start_date, end_date],
        "selection_diagnostics": diagnostics,
        "portfolio": portfolio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="固定路径模型阈值的可执行复利研究")
    parser.add_argument("--output", default=str(默认输出))
    args = parser.parse_args()

    started = time.time()
    if round(固定概率阈值, 2) not in 阈值网格:
        raise SystemExit(f"固定阈值 {固定概率阈值} 不在路径模型研究阈值网格中")

    files = history_files()
    dates = [path.name[:10] for path in files]
    required_dates = (TRAIN_END, VALIDATION_START, VALIDATION_END, FINAL_START, FINAL_END)
    missing = [date for date in required_dates if date not in set(dates)]
    if missing:
        raise SystemExit("历史行情缺少研究边界交易日：" + "、".join(missing))
    if dates.index(FINAL_END) + MAX_FORWARD_SESSIONS >= len(dates):
        raise SystemExit("最终留出区间之后的历史行情不足，无法完成最长持有窗口的退出模拟")

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

    validation = _portfolio_window(
        files, dates, model, VALIDATION_START, validation_safe_end,
        固定概率阈值, 每日最多候选数,
    )
    final_holdout = _portfolio_window(
        files, dates, model, FINAL_START, FINAL_END,
        固定概率阈值, 每日最多候选数,
    )

    result = {
        "schema_version": 1,
        "status": "research_only",
        "method": "path_aware_gbdt_fixed_threshold_executable_compound_backtest",
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
            "fixed_probability_threshold": 固定概率阈值,
            "top_k_per_signal_day": 每日最多候选数,
            "threshold_source": "上一轮路径感知模型验证集已选基准；本轮固定，不搜索或调整",
            "initial_equity": 初始资金,
            "net_target_pct": 目标净收益百分比,
            "stop_loss_pct": 止损幅度百分比,
            "round_trip_cost_bps": 往返交易成本基点,
            "cash_constrained_no_leverage": True,
        },
        "splits": {
            "validation": [VALIDATION_START, validation_safe_end],
            "validation_nominal_end": VALIDATION_END,
            "final_holdout": [FINAL_START, FINAL_END],
            "validation_exit_paths_end_before_final_holdout": True,
        },
        "validation": validation,
        "final_holdout": final_holdout,
        "audit": {
            "no_future_features": True,
            "candidate_admission_does_not_use_future_path_completeness": True,
            "entry_is_T_plus_1_open": True,
            "exit_starts_T_plus_2": True,
            "upper_limit_and_paused_entries_blocked_by_shared_execution_mask": True,
            "stop_first_on_same_bar": True,
            "round_trip_cost_applied": True,
            "cash_constrained_no_leverage": True,
            "threshold_not_tuned_on_final_holdout": True,
            "final_holdout_result_is_supplemental_not_a_fresh_unseen_holdout": True,
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
                "threshold": 固定概率阈值,
                "splits": result["splits"],
                "validation_portfolio": validation["portfolio"],
                "final_holdout_portfolio": final_holdout["portfolio"],
                "elapsed_seconds": result["elapsed_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
