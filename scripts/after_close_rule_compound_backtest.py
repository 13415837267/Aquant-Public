"""使用发现期冻结的可解释条件，按统一执行约束进行组合回测。"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compound_portfolio_backtest import 初始资金, candidate_bar_path, 运行组合
from scripts.path_rule_mining import (
    FEATURE_INDEX,
    STOCK_FEATURES,
    TRAIN_END,
    VALIDATION_END,
    VALIDATION_START,
    FINAL_START,
    FINAL_END,
    make_features,
)
from scripts.selection_factor_catalog import load_research_config
from scripts.short_term_research import (
    FEATURE_WARMUP_SESSIONS,
    FeatureState,
    executable_entry_mask,
    history_files,
    read_daily,
)

CONFIG = load_research_config()
MAX_FORWARD_SESSIONS = int(CONFIG["最大前瞻交易日数"])
NET_TARGET_PCT = float(CONFIG["短线目标净收益百分比"])
STOP_LOSS_PCT = float(CONFIG["止损幅度百分比"])
ROUND_TRIP_COST_BPS = float(CONFIG["往返交易成本基点"])
MIN_TRAIN_SAMPLES = int(CONFIG["最少训练条件样本数"])
TOP_K_PER_DAY = 2


def _metric(row: dict, name: str) -> float:
    value = row.get(name)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return number if math.isfinite(number) else -math.inf


def select_training_rule(data: dict) -> dict:
    """只依据训练期统计从已冻结候选中选规则，不查看验证或最终期表现。"""
    candidates = []
    for key in ("train_top_atomic", "train_top_pairs"):
        rows = data.get(key) or []
        if not isinstance(rows, list):
            raise ValueError(f"{key} 必须是数组")
        for row in rows:
            if not isinstance(row, dict):
                continue
            conditions = row.get("rules")
            if not isinstance(conditions, list) or not conditions:
                continue
            # 市场环境条件对所有股票取值相同。至少保留一个个股因子，才能形成个股筛选与排序。
            has_stock_condition = any(
                condition.get("scale") == "rank"
                and condition.get("feature") in STOCK_FEATURES
                for condition in conditions
                if isinstance(condition, dict)
            )
            if not has_stock_condition:
                continue
            samples = row.get("samples")
            resolved_samples = row.get("resolved_net_return_samples")
            if (
                not isinstance(samples, int) or samples < MIN_TRAIN_SAMPLES
                or not isinstance(resolved_samples, int) or resolved_samples < MIN_TRAIN_SAMPLES
            ):
                continue
            candidates.append(row)
    if not candidates:
        raise ValueError("训练期没有达到样本门槛且包含个股因子的可解释规则")
    candidates.sort(
        key=lambda row: (
            _metric(row, "mean_path_net_return_pct"),
            _metric(row, "path_profit_factor"),
            _metric(row, "wilson_lower_pct"),
            _metric(row, "samples"),
        ),
        reverse=True,
    )
    chosen = dict(candidates[0])
    mean_net = _metric(chosen, "mean_path_net_return_pct")
    profit_factor = _metric(chosen, "path_profit_factor")
    chosen["training_edge_gate_passed"] = bool(
        math.isfinite(mean_net) and mean_net > 0.0
        and math.isfinite(profit_factor) and profit_factor > 1.0
    )
    chosen["training_selection_objective"] = "训练期路径净收益均值优先，其次路径盈利因子、Wilson下界和样本数"
    return chosen


def evaluate_rule_conditions(
    feature_matrix: np.ndarray, conditions: list[dict]
) -> tuple[np.ndarray, np.ndarray]:
    """返回规则命中掩码及条件边界内的个股排序分，不使用未来数据。"""
    x = np.asarray(feature_matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("个股特征矩阵必须是二维数组")
    matched = np.ones(len(x), dtype=bool)
    stock_margins = []

    for condition in conditions:
        if not isinstance(condition, dict):
            raise ValueError("规则条件必须为对象")
        feature = condition.get("feature")
        if feature not in FEATURE_INDEX:
            raise ValueError(f"规则引用未启用的因子：{feature}")
        operator = condition.get("op")
        threshold = float(condition.get("threshold"))
        if not math.isfinite(threshold) or operator not in (">=", "<="):
            raise ValueError("规则阈值或运算符无效")
        values = x[:, FEATURE_INDEX[feature]]
        valid = np.isfinite(values)
        if operator == ">=":
            matched &= valid & (values >= threshold)
        else:
            matched &= valid & (values <= threshold)

        if condition.get("scale") == "rank" and feature in STOCK_FEATURES:
            if operator == ">=":
                denominator = max(1.0 - threshold, 1e-9)
                margin = (values - threshold) / denominator
            else:
                denominator = max(abs(threshold), 1e-9)
                margin = (threshold - values) / denominator
            stock_margins.append(np.clip(margin, 0.0, 1.0))

    if not stock_margins:
        raise ValueError("规则必须包含至少一个个股分位因子，才能产生可复现的排序")
    score = np.mean(np.column_stack(stock_margins), axis=1)
    return matched, score


def _normal_symbol_keys(values) -> np.ndarray:
    return (
        pd.Series(values, dtype="string")
        .str.extract(r"(\d{6})", expand=False)
        .fillna("")
        .str.zfill(6)
        .to_numpy(dtype=str)
    )


def complete_signal_end(dates: list[str], nominal_end: str) -> str:
    """只对仍覆盖完整未来退出窗口的信号日记分。"""
    if nominal_end not in dates:
        raise ValueError(f"名义结束日期不在行情索引中：{nominal_end}")
    last_complete = len(dates) - MAX_FORWARD_SESSIONS - 1
    if last_complete < 0:
        raise ValueError("历史行情不足以覆盖完整退出窗口")
    return dates[min(dates.index(nominal_end), last_complete)]


def collect_daily_candidates(
    files: list[Path],
    dates: list[str],
    start_date: str,
    end_date: str,
    conditions: list[dict],
) -> tuple[dict[str, list[dict]], dict]:
    """每个信号日只保留规则排序最高的两个可执行候选。"""
    index = {date: i for i, date in enumerate(dates)}
    if start_date not in index or end_date not in index:
        raise ValueError(f"研究窗口不在交易日索引中：{start_date} 至 {end_date}")
    start_i, end_i = index[start_date], index[end_date]
    if start_i > end_i or end_i + MAX_FORWARD_SESSIONS >= len(files):
        raise ValueError(f"研究窗口缺少完整的未来行情：{start_date} 至 {end_date}")

    cache: dict[int, pd.DataFrame] = {}

    def get(i: int) -> pd.DataFrame:
        if i not in cache:
            cache[i] = read_daily(files[i])
        while len(cache) > MAX_FORWARD_SESSIONS + 3:
            del cache[next(iter(cache))]
        return cache[i]

    state = FeatureState()
    daily: dict[str, list[dict]] = {}
    diagnostics = {
        "signal_days": 0,
        "matched_symbols": 0,
        "executable_matched_symbols": 0,
        "selected_candidates": 0,
    }

    for i in range(max(0, start_i - FEATURE_WARMUP_SESSIONS), end_i + 1):
        signal_date = dates[i]
        frame = state.build(get(i))
        if signal_date < start_date or signal_date > end_date or frame.empty:
            continue

        symbols = _normal_symbol_keys(frame["symbol"])
        x = make_features(frame)
        matched, scores = evaluate_rule_conditions(x, conditions)
        future_days = [get(i + offset) for offset in range(1, MAX_FORWARD_SESSIONS + 1)]
        future_dates = dates[i + 1 : i + MAX_FORWARD_SESSIONS + 1]
        entry_day = future_days[0]
        executable = executable_entry_mask(symbols, entry_day)
        eligible = np.flatnonzero(matched & executable & np.isfinite(scores))
        diagnostics["signal_days"] += 1
        diagnostics["matched_symbols"] += int(matched.sum())
        diagnostics["executable_matched_symbols"] += int(len(eligible))

        ordered = eligible[np.argsort(-scores[eligible], kind="stable")]
        chosen = ordered[:TOP_K_PER_DAY]
        entry_frame = entry_day.copy()
        entry_frame["_symbol_key"] = _normal_symbol_keys(entry_frame["symbol"])
        entry_frame = (
            entry_frame.drop_duplicates("_symbol_key", keep="last")
            .set_index("_symbol_key")
        )
        opens = pd.to_numeric(entry_frame["open"], errors="coerce").reindex(symbols).to_numpy(dtype=float)

        candidates = []
        for row_index in chosen:
            symbol = str(symbols[row_index])
            entry_price = float(opens[row_index])
            if not math.isfinite(entry_price) or entry_price <= 0:
                continue
            bars = candidate_bar_path(symbol, future_days, future_dates, entry_price)
            if not bars or not bars[0].get("bar_available", False):
                raise RuntimeError(f"T+1可执行掩码与候选路径不一致：{signal_date} {symbol}")
            ranking_score = float(scores[row_index])
            candidates.append({
                "signal_date": signal_date,
                "symbol": symbol,
                # 运行组合接口以 probability 字段统一排序；此处存的是规则边界排序分，不是模型概率。
                "probability": ranking_score,
                "ranking_score": ranking_score,
                "entry_date": bars[0]["date"],
                "entry_price": entry_price,
                "bars": bars,
            })
        daily[signal_date] = candidates
        diagnostics["selected_candidates"] += len(candidates)

        if diagnostics["signal_days"] % 50 == 0:
            print(
                f"[训练期冻结规则组合] {start_date}至{end_date}："
                f"已处理{diagnostics['signal_days']}个信号日，"
                f"规则命中股票{diagnostics['matched_symbols']}只次，"
                f"可执行命中{diagnostics['executable_matched_symbols']}只次",
                flush=True,
            )

    return daily, diagnostics


def run_portfolio(
    candidates: dict[str, list[dict]],
    dates: list[str],
    start_date: str,
    signal_end: str,
) -> dict:
    measurement_end_index = dates.index(signal_end) + MAX_FORWARD_SESSIONS
    if measurement_end_index >= len(dates):
        raise ValueError(f"完整退出窗口超出行情范围：{signal_end}")
    result = 运行组合(
        candidates,
        dates,
        "每日排名",
        NET_TARGET_PCT,
        STOP_LOSS_PCT,
        ROUND_TRIP_COST_BPS,
        top_k=TOP_K_PER_DAY,
        窗口起始日期=start_date,
        窗口结束日期=dates[measurement_end_index],
    )
    result["measurement_window"] = [start_date, dates[measurement_end_index]]
    result["signal_window"] = [start_date, signal_end]
    result["selection_policy"] = {
        "signal": "T日收盘",
        "entry": "T+1开盘",
        "earliest_exit": "T+2",
        "maximum_holding_sessions": MAX_FORWARD_SESSIONS,
        "top_k_per_signal_day": TOP_K_PER_DAY,
        "net_target_pct": NET_TARGET_PCT,
        "stop_loss_pct": STOP_LOSS_PCT,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "ranking": "规则条件边界内的个股分位余量；不是模型概率",
        "cash_constrained_no_leverage": True,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="训练期冻结规则的现金约束组合回测")
    parser.add_argument("--rules", required=True, help="规则挖掘结果JSON")
    parser.add_argument(
        "--output",
        default=str(ROOT / "data/backtest/after_close_rule_compound.json"),
        help="输出JSON",
    )
    args = parser.parse_args()

    started = time.time()
    data = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("status") != "research_only":
        raise ValueError("规则挖掘输入必须是research_only结果")
    rule = select_training_rule(data)

    files = history_files()
    dates = [path.name[:10] for path in files]
    required_dates = (TRAIN_END, VALIDATION_START, VALIDATION_END, FINAL_START, FINAL_END)
    missing = [date for date in required_dates if date not in set(dates)]
    if missing:
        raise ValueError("历史行情缺少研究边界交易日：" + "、".join(missing))

    validation_start_index = dates.index(VALIDATION_START)
    final_start_index = dates.index(FINAL_START)
    validation_end_index = final_start_index - MAX_FORWARD_SESSIONS - 1
    if validation_end_index < validation_start_index:
        raise ValueError("清除跨入最终期的前瞻标签后，验证窗口为空")
    validation_end = dates[validation_end_index]
    final_end = complete_signal_end(dates, FINAL_END)

    validation_candidates, validation_diagnostics = collect_daily_candidates(
        files, dates, VALIDATION_START, validation_end, rule["rules"]
    )
    validation_portfolio = run_portfolio(
        validation_candidates, dates, VALIDATION_START, validation_end
    )
    validation_portfolio["candidate_diagnostics"] = validation_diagnostics

    final_candidates, final_diagnostics = collect_daily_candidates(
        files, dates, FINAL_START, final_end, rule["rules"]
    )
    final_portfolio = run_portfolio(final_candidates, dates, FINAL_START, final_end)
    final_portfolio["candidate_diagnostics"] = final_diagnostics
    final_portfolio["selection_status"] = "条件仅由训练期统计选择；最终期仅作补充评估"

    result = {
        "schema_version": 1,
        "status": "research_only",
        "method": "training_frozen_rule_cash_constrained_portfolio",
        "source_public_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "training_selected_rule": {
            "rule_type": rule.get("rule_type"),
            "rule_label": rule.get("rule_label"),
            "rules": rule.get("rules"),
            "training_samples": rule.get("samples"),
            "training_path_win_1pct_rate_pct": rule.get("path_win_1pct_rate_pct"),
            "training_wilson_lower_pct": rule.get("wilson_lower_pct"),
            "training_mean_path_net_return_pct": rule.get("mean_path_net_return_pct"),
            "training_path_profit_factor": rule.get("path_profit_factor"),
            "training_edge_gate_passed": rule.get("training_edge_gate_passed"),
            "selection_method": "只按2015—2022发现期路径净收益均值、路径盈利因子、Wilson下界和样本量排序；验证与补充期不参与选式",
        },
        "selection_policy": {
            "top_k_per_signal_day": TOP_K_PER_DAY,
            "initial_equity": 初始资金,
            "net_target_pct": NET_TARGET_PCT,
            "stop_loss_pct": STOP_LOSS_PCT,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "maximum_holding_sessions": MAX_FORWARD_SESSIONS,
            "cash_constrained_no_leverage": True,
        },
        "splits": {
            "train": [files[0].name[:10], TRAIN_END],
            "validation": [VALIDATION_START, validation_end],
            "validation_nominal_end": VALIDATION_END,
            "final_supplemental": [FINAL_START, final_end],
            "final_nominal_end": FINAL_END,
        },
        "validation_portfolio": validation_portfolio,
        "final_supplemental_portfolio": final_portfolio,
        "audit": {
            "rule_selected_from_training_only": True,
            "validation_used_only_to_evaluate_frozen_rule": True,
            "final_window_is_not_fresh_unseen_holdout": True,
            "ranking_uses_only_T_close_features": True,
            "no_future_features": True,
            "entry_is_T_plus_1_open": True,
            "exit_starts_T_plus_2": True,
            "entry_at_upper_limit_blocked": True,
            "paused_entry_blocked": True,
            "stop_first_on_same_bar": True,
            "round_trip_cost_applied": True,
            "cash_constrained_no_leverage": True,
            "formal_production_changed": False,
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "rule": result["training_selected_rule"]["rule_label"],
        "rule_train_hit_rate_pct": result["training_selected_rule"]["training_path_win_1pct_rate_pct"],
        "rule_train_mean_path_net_return_pct": result["training_selected_rule"]["training_mean_path_net_return_pct"],
        "rule_train_profit_factor": result["training_selected_rule"]["training_path_profit_factor"],
        "rule_training_edge_gate_passed": result["training_selected_rule"]["training_edge_gate_passed"],
        "validation": {
            key: validation_portfolio.get(key)
            for key in ("completed_trades", "compound_return_pct", "annualized_compound_return_pct", "max_drawdown_pct", "target_hit_rate_pct")
        },
        "final_supplemental": {
            key: final_portfolio.get(key)
            for key in ("completed_trades", "compound_return_pct", "annualized_compound_return_pct", "max_drawdown_pct", "target_hit_rate_pct")
        },
        "formal_production_changed": False,
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
