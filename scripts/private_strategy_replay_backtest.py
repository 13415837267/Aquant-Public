"""统一回放冻结基准与3.0.0结构重构策略，比较真实执行约束下的成本后组合表现。"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compound_portfolio_backtest import 初始资金, candidate_bar_path, 运行组合
from scripts.generate_research_candidates import eligible_universe
from scripts.selection_factor_catalog import load_research_config
from scripts.short_term_research import (
    FEATURE_WARMUP_SESSIONS,
    FeatureState,
    executable_entry_mask,
    history_files,
    read_daily,
)

BASELINE_COMMIT = "886a01a3a314dd1ba133b9273054b280e3bc871c"
CANDIDATE_COMMIT = "d9d6a3b4cdbcc922942694372f93eeb1b906e746"
VALIDATION_START = "2023-01-03"
FINAL_START = "2025-01-02"
FINAL_NOMINAL_END = "2026-09-30"
MIN_VALIDATION_TRADES = 100
BASELINE_VERSION = "2.6.0"
CANDIDATE_VERSION = "3.0.0"
MIN_FINAL_TRADES = 40
MIN_PROFIT_FACTOR = 1.2
MAX_DRAWDOWN_FLOOR_PCT = -15.0
ROOT_CONFIG = load_research_config()
BASE_COST_BPS = float(ROOT_CONFIG["往返交易成本基点"])
COST_MULTIPLIERS = tuple(sorted({1.0, *[float(x) for x in ROOT_CONFIG["候选数量成本压力倍率"]]}))
MAX_HOLD = int(ROOT_CONFIG["最大前瞻交易日数"])


def yaml_number(text: str, key: str) -> float:
    match = re.search(
        r"^\s*" + re.escape(key) + r"\s*:\s*([-+]?\d+(?:\.\d+)?)\s*$",
        text,
        re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"私有策略配置缺少数值参数：{key}")
    return float(match.group(1))


def yaml_string(text: str, key: str) -> str:
    match = re.search(r'^\s*' + re.escape(key) + r'\s*:\s*"([^"]+)"\s*$', text, re.MULTILINE)
    if match is None:
        raise ValueError(f"私有策略配置缺少文本参数：{key}")
    return match.group(1)


def yaml_bool(text: str, key: str) -> bool:
    match = re.search(r'^\s*' + re.escape(key) + r'\s*:\s*(true|false)\s*$', text, re.MULTILINE)
    if match is None:
        raise ValueError(f"私有策略配置缺少布尔参数：{key}")
    return match.group(1) == "true"


def assert_replay_versions(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    """确保回放契约版本与当前冻结基准及研究候选一致。"""
    baseline_version = baseline.get("version")
    candidate_version = candidate.get("version")
    if baseline_version != BASELINE_VERSION or candidate_version != CANDIDATE_VERSION:
        raise ValueError(
            "基准与实验版本不符合本次回放契约："
            f"期望基准 {BASELINE_VERSION}、候选 {CANDIDATE_VERSION}；"
            f"实际基准 {baseline_version}、候选 {candidate_version}"
        )


def load_strategy(root: Path, alias: str) -> dict[str, Any]:
    """以独立包名加载不同提交的策略，避免基准与实验模块互相污染。"""
    root = root.resolve()
    package_root = root / "strategy"
    if not package_root.is_dir():
        raise FileNotFoundError(f"策略目录不存在：{package_root}")
    package = types.ModuleType(alias)
    package.__path__ = [str(package_root)]
    package.__package__ = alias
    sys.modules[alias] = package
    modules: dict[str, Any] = {}
    for name in ("precision_config", "candidate_policy", "version", "model"):
        module_name = f"{alias}.{name}"
        spec = importlib.util.spec_from_file_location(module_name, package_root / f"{name}.py")
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载策略模块：{module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules[name] = module

    config_text = (root / "config" / "strategy.yaml").read_text(encoding="utf-8")
    version = str(modules["version"].STRATEGY_VERSION)
    config_version = re.search(r'^\s*version\s*:\s*"([^"]+)"\s*$', config_text, re.MULTILINE)
    if config_version is None or config_version.group(1) != version:
        raise ValueError(f"{alias} 的代码版本与策略配置版本不一致")
    git_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "root": root,
        "version": version,
        "commit": git_commit,
        "model": modules["model"],
        "precision": modules["precision_config"],
        "policy": modules["candidate_policy"],
        "minimum_price": yaml_number(config_text, "min_price"),
        "minimum_turnover_amount": yaml_number(config_text, "min_turnover_amount_cny"),
        "minimum_history_days": int(yaml_number(config_text, "minimum_history_days")),
        "minimum_market_breadth": yaml_number(config_text, "minimum_breadth_pct"),
        "minimum_market_median_return": yaml_number(config_text, "minimum_median_return_pct"),
        "minimum_market_above_ma20": yaml_number(config_text, "minimum_above_ma20_pct") if re.search(r"^\s*minimum_above_ma20_pct\s*:", config_text, re.MULTILINE) else float("nan"),
        "minimum_market_positive_5d": yaml_number(config_text, "minimum_positive_5d_pct") if re.search(r"^\s*minimum_positive_5d_pct\s*:", config_text, re.MULTILINE) else float("nan"),
        "minimum_research_score": yaml_number(config_text, "minimum_research_score") if re.search(r"^\s*minimum_research_score\s*:", config_text, re.MULTILINE) else float("nan"),
        "maximum_candidates": int(yaml_number(config_text, "maximum_candidates")) if re.search(r"^\s*maximum_candidates\s*:", config_text, re.MULTILINE) else int(modules["model"].MAX_CANDIDATES),
        "target_return_pct": yaml_number(config_text, "target_return_pct"),
        "stop_loss_pct": yaml_number(config_text, "stop_loss_pct"),
        "max_holding_sessions": int(yaml_number(config_text, "max_holding_sessions")),
        "maximum_exit_session_after_signal": int(yaml_number(config_text, "maximum_exit_session_after_signal")),
        "signal_at": yaml_string(config_text, "signal_at"),
        "entry": yaml_string(config_text, "entry"),
        "earliest_exit": yaml_string(config_text, "earliest_exit"),
        "entry_limit_up_block": yaml_bool(config_text, "entry_limit_up_block"),
    }


def assert_reconstruction_contract(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    """核验3.0.0允许进行结构重构，同时保留A股执行时序和流动性范围。"""
    assert_replay_versions(baseline, candidate)

    shared_config = (
        "minimum_price",
        "minimum_turnover_amount",
        "minimum_history_days",
        "max_holding_sessions",
        "maximum_exit_session_after_signal",
        "signal_at",
        "entry",
        "earliest_exit",
        "entry_limit_up_block",
    )
    for key in shared_config:
        if baseline[key] != candidate[key]:
            raise ValueError(f"结构重构改变了禁止变动的执行契约：{key}")

    if candidate["model"].STRATEGY_METHOD != "risk_adjusted_cross_sectional_rank_v1":
        raise ValueError("候选策略未启用3.0.0风险调整横截面评分方法")
    if candidate["policy"].ADMISSION_POLICY != "risk_adjusted_cross_sectional_rank_top_2":
        raise ValueError("候选策略准入规则与3.0.0契约不符")

    if not math.isclose(
        sum(float(value) for value in candidate["precision"].COMPONENT_WEIGHTS.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("3.0.0评分因子组权重之和必须为1")
    if not math.isclose(
        candidate["minimum_research_score"],
        float(candidate["precision"].MIN_RESEARCH_SCORE),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("配置文件与策略代码的最低研究评分不一致")
    if candidate["maximum_candidates"] != int(candidate["precision"].MAX_CANDIDATES):
        raise ValueError("配置文件与策略代码的候选数量限制不一致")

    market_values = (
        ("minimum_market_breadth", "MIN_MARKET_BREADTH_PCT"),
        ("minimum_market_median_return", "MIN_MARKET_MEDIAN_RETURN_PCT"),
        ("minimum_market_above_ma20", "MIN_MARKET_ABOVE_MA20_PCT"),
        ("minimum_market_positive_5d", "MIN_MARKET_POSITIVE_5D_PCT"),
    )
    for config_key, module_key in market_values:
        if not math.isclose(
            float(candidate[config_key]),
            float(getattr(candidate["precision"], module_key)),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"配置文件与策略代码中的市场状态门槛不一致：{config_key}")

    if not (candidate["target_return_pct"] > baseline["target_return_pct"]):
        raise ValueError("3.0.0必须使用大于基准的研究止盈目标")
    if not (0.0 < candidate["stop_loss_pct"] < baseline["stop_loss_pct"]):
        raise ValueError("3.0.0必须使用正数且小于基准的研究止损幅度")
    if candidate["maximum_candidates"] != 2:
        raise ValueError("3.0.0当前契约要求每日最多两个候选")


def _symbol_keys(values: Any) -> np.ndarray:
    return (
        pd.Series(values, dtype="string")
        .str.extract(r"(\d{6})", expand=False)
        .fillna("")
        .str.zfill(6)
        .to_numpy(dtype=str)
    )


def _number(row: pd.Series, key: str) -> float:
    value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) and np.isfinite(float(value)) else 0.0


def collect_strategy_candidates(
    strategy: dict[str, Any],
    feature_frame: pd.DataFrame,
    entry_day: pd.DataFrame,
    future_days: list[pd.DataFrame],
    future_dates: list[str],
    signal_date: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """T日完成排序和准入；T+1行情只用于实际入场可执行性，不做候选回填。"""
    eligible = eligible_universe(
        feature_frame,
        strategy["minimum_price"],
        strategy["minimum_turnover_amount"],
    )
    if eligible.empty:
        return [], {"eligible": 0, "selected_at_close": 0, "entered": 0, "blocked_at_entry": 0}

    scored = strategy["model"].score_universe(eligible)
    selected = strategy["model"].admit_candidates(scored).copy()
    if selected.empty:
        return [], {"eligible": len(eligible), "selected_at_close": 0, "entered": 0, "blocked_at_entry": 0}

    symbols = _symbol_keys(selected["symbol"])
    executable = executable_entry_mask(symbols, entry_day)
    entry = entry_day.copy()
    entry["_symbol_key"] = _symbol_keys(entry["symbol"])
    entry = entry.drop_duplicates("_symbol_key", keep="last").set_index("_symbol_key")
    open_prices = pd.to_numeric(entry["open"], errors="coerce").reindex(symbols).to_numpy(dtype=float)

    candidates: list[dict[str, Any]] = []
    blocked = 0
    for index, (_, row) in enumerate(selected.iterrows()):
        if not bool(executable[index]):
            blocked += 1
            continue
        symbol = str(symbols[index])
        entry_price = float(open_prices[index])
        if not np.isfinite(entry_price) or entry_price <= 0:
            blocked += 1
            continue
        bars = candidate_bar_path(symbol, future_days, future_dates, entry_price)
        if not bars or not bars[0].get("bar_available", False):
            raise RuntimeError(f"T+1执行门禁与价格路径不一致：{signal_date} {symbol}")
        candidates.append({
            "signal_date": signal_date,
            "symbol": symbol,
            "probability": _number(row, "score"),
            "ranking_score": _number(row, "score"),
            "entry_date": bars[0]["date"],
            "entry_price": entry_price,
            "bars": bars,
        })
    return candidates, {
        "eligible": len(eligible),
        "selected_at_close": len(selected),
        "entered": len(candidates),
        "blocked_at_entry": blocked,
    }


def complete_signal_end(dates: list[str], nominal_end: str) -> str:
    if nominal_end not in dates:
        raise ValueError(f"研究边界日期不在行情索引中：{nominal_end}")
    last = min(dates.index(nominal_end), len(dates) - MAX_HOLD - 1)
    if last < 0:
        raise ValueError("历史数据长度不足以覆盖退出窗口")
    return dates[last]


def _metric_summary(result: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "status", "requested_entries", "entered_trades", "completed_trades",
        "compound_return_pct", "annualized_compound_return_pct", "max_drawdown_pct",
        "mean_trade_net_return_pct", "profit_factor", "positive_trade_rate_pct",
        "target_hit_rate_pct", "sharpe_ratio", "skipped_for_capacity_or_cash",
        "open_positions_at_end",
    )
    return {key: result.get(key) for key in fields}


def _valid_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def quality_gate(metrics: dict[str, Any], minimum_trades: int) -> bool:
    """使用预先固定的交易数、收益、盈利因子和回撤门槛。"""
    return (
        int(metrics.get("completed_trades") or 0) >= minimum_trades
        and _valid_number(metrics.get("compound_return_pct"))
        and float(metrics["compound_return_pct"]) > 0.0
        and _valid_number(metrics.get("mean_trade_net_return_pct"))
        and float(metrics["mean_trade_net_return_pct"]) > 0.0
        and _valid_number(metrics.get("profit_factor"))
        and float(metrics["profit_factor"]) >= MIN_PROFIT_FACTOR
        and _valid_number(metrics.get("max_drawdown_pct"))
        and float(metrics["max_drawdown_pct"]) >= MAX_DRAWDOWN_FLOOR_PCT
    )


def evaluate_candidate(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """只有成本场景均过门槛，且各窗口收益及回撤不差于基准，才保留策略改动。"""
    gates: dict[str, bool] = {}
    comparisons: dict[str, bool] = {}
    windows = ("validation", "final_supplemental")
    for window in windows:
        min_trades = MIN_VALIDATION_TRADES if window == "validation" else MIN_FINAL_TRADES
        for cost_key in baseline[window]:
            label = f"{window}_{cost_key}"
            cand = candidate[window][cost_key]
            base = baseline[window][cost_key]
            gates[label] = quality_gate(cand, min_trades)
            valid_returns = _valid_number(cand.get("compound_return_pct")) and _valid_number(base.get("compound_return_pct"))
            valid_drawdowns = _valid_number(cand.get("max_drawdown_pct")) and _valid_number(base.get("max_drawdown_pct"))
            comparisons[label] = bool(
                valid_returns
                and valid_drawdowns
                and float(cand["compound_return_pct"]) >= float(base["compound_return_pct"])
                and float(cand["max_drawdown_pct"]) >= float(base["max_drawdown_pct"]) - 0.10
            )
    passed = all(gates.values())
    beats_baseline = all(comparisons.values()) and any(
        float(candidate[w][c]["compound_return_pct"]) > float(baseline[w][c]["compound_return_pct"])
        for w in windows
        for c in baseline[w]
        if _valid_number(candidate[w][c].get("compound_return_pct"))
        and _valid_number(baseline[w][c].get("compound_return_pct"))
    )
    return {
        "quality_gates_by_window_and_cost": gates,
        "beats_baseline_by_window_and_cost": comparisons,
        "all_quality_gates_passed": passed,
        "all_baseline_comparisons_passed": all(comparisons.values()),
        "strict_improvement_over_baseline": bool(passed and beats_baseline),
        "decision": "保留研究候选" if passed and beats_baseline else "拒绝本次策略改动",
        "formal_production_changed": False,
        "automatic_ordering": False,
    }


def run_replay(baseline_root: Path, candidate_root: Path) -> dict[str, Any]:
    started = time.time()
    baseline = load_strategy(baseline_root, "aquant_baseline")
    candidate = load_strategy(candidate_root, "aquant_candidate")
    assert_reconstruction_contract(baseline, candidate)
    if baseline["commit"] != BASELINE_COMMIT:
        raise ValueError(f"基准策略提交号不符：{baseline['commit']}")
    if candidate["commit"] != CANDIDATE_COMMIT:
        raise ValueError(f"实验策略提交号不符：{candidate['commit']}")
    assert_replay_versions(baseline, candidate)

    files = history_files()
    dates = [path.name[:10] for path in files]
    required = (VALIDATION_START, FINAL_START, FINAL_NOMINAL_END)
    missing = [date for date in required if date not in set(dates)]
    if missing:
        raise ValueError("历史行情缺少必要研究边界：" + "、".join(missing))

    date_index = {date: i for i, date in enumerate(dates)}
    validation_end = dates[date_index[FINAL_START] - MAX_HOLD - 1]
    final_end = complete_signal_end(dates, FINAL_NOMINAL_END)
    validation_start_index = date_index[VALIDATION_START]
    final_end_index = date_index[final_end]
    windows = {
        "validation": (VALIDATION_START, validation_end),
        "final_supplemental": (FINAL_START, final_end),
    }
    daily = {
        "baseline": {name: {} for name in windows},
        "candidate": {name: {} for name in windows},
    }
    diagnostics = {
        "baseline": {name: {"signal_days": 0, "eligible": 0, "selected_at_close": 0, "entered": 0, "blocked_at_entry": 0} for name in windows},
        "candidate": {name: {"signal_days": 0, "eligible": 0, "selected_at_close": 0, "entered": 0, "blocked_at_entry": 0} for name in windows},
    }

    cache: dict[int, pd.DataFrame] = {}

    def get(index: int) -> pd.DataFrame:
        if index not in cache:
            cache[index] = read_daily(files[index])
        while len(cache) > MAX_HOLD + 3:
            cache.pop(next(iter(cache)))
        return cache[index]

    state = FeatureState()
    first_index = max(0, validation_start_index - FEATURE_WARMUP_SESSIONS)
    for i in range(first_index, final_end_index + 1):
        raw = get(i)
        feature_frame = state.build(raw)
        signal_date = dates[i]
        if signal_date < VALIDATION_START or signal_date > final_end:
            continue
        if feature_frame.empty or i + MAX_HOLD >= len(files):
            continue
        window_name = None
        for name, (start, end) in windows.items():
            if start <= signal_date <= end:
                window_name = name
                break
        if window_name is None:
            continue

        future_days = [get(i + offset) for offset in range(1, MAX_HOLD + 1)]
        future_dates = dates[i + 1 : i + MAX_HOLD + 1]
        entry_day = future_days[0]
        for variant, strategy in (("baseline", baseline), ("candidate", candidate)):
            records, counts = collect_strategy_candidates(
                strategy, feature_frame, entry_day, future_days, future_dates, signal_date
            )
            daily[variant][window_name][signal_date] = records
            diagnostics[variant][window_name]["signal_days"] += 1
            for key in ("eligible", "selected_at_close", "entered", "blocked_at_entry"):
                diagnostics[variant][window_name][key] += counts[key]

        processed = diagnostics["baseline"][window_name]["signal_days"]
        if processed % 50 == 0:
            print(
                f"[策略回放] {window_name} 已完成 {processed} 个信号日；"
                f"基准累计入场={diagnostics['baseline'][window_name]['entered']}，"
                f"实验累计入场={diagnostics['candidate'][window_name]['entered']}",
                flush=True,
            )

    results: dict[str, dict[str, dict[str, Any]]] = {
        "baseline": {},
        "candidate": {},
    }
    for variant in ("baseline", "candidate"):
        for window_name, (start, end) in windows.items():
            measurement_end = dates[min(date_index[end] + MAX_HOLD, len(dates) - 1)]
            by_cost: dict[str, Any] = {}
            for multiplier in COST_MULTIPLIERS:
                cost_bps = BASE_COST_BPS * multiplier
                raw_metrics = 运行组合(
                    daily[variant][window_name],
                    dates,
                    "每日排名",
                    (baseline if variant == "baseline" else candidate)["target_return_pct"],
                    (baseline if variant == "baseline" else candidate)["stop_loss_pct"],
                    cost_bps,
                    top_k=2,
                    窗口起始日期=start,
                    窗口结束日期=measurement_end,
                )
                cost_key = f"{cost_bps:g}_bps"
                by_cost[cost_key] = _metric_summary(raw_metrics)
                print(
                    f"[策略回放] {variant} / {window_name} / 往返成本{cost_bps:g}基点："
                    f"交易={raw_metrics.get('completed_trades', 0)}，"
                    f"复利={raw_metrics.get('compound_return_pct')}%，"
                    f"回撤={raw_metrics.get('max_drawdown_pct')}%，"
                    f"盈利因子={raw_metrics.get('profit_factor')}",
                    flush=True,
                )
            results[variant][window_name] = by_cost

    baseline_result = results["baseline"]
    candidate_result = results["candidate"]
    decision = evaluate_candidate(baseline_result, candidate_result)
    return {
        "schema_version": 2,
        "status": "research_only",
        "method": "risk_adjusted_strategy_reconstruction_replay",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_public_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "strategy_versions": {
            "baseline": {"version": baseline["version"], "commit": baseline["commit"]},
            "candidate": {"version": candidate["version"], "commit": candidate["commit"]},
        },
        "objective": {
            "signal": "T日收盘",
            "entry": "T+1开盘",
            "earliest_exit": "T+2",
            "max_holding_sessions": MAX_HOLD,
            "target_net_profit_pct": baseline["target_return_pct"],
            "stop_loss_pct": baseline["stop_loss_pct"],
            "target_and_stop_by_version": {
                "baseline": {"target_return_pct": baseline["target_return_pct"], "stop_loss_pct": baseline["stop_loss_pct"]},
                "candidate": {"target_return_pct": candidate["target_return_pct"], "stop_loss_pct": candidate["stop_loss_pct"]},
            },
            "base_round_trip_cost_bps": BASE_COST_BPS,
            "cost_multipliers": list(COST_MULTIPLIERS),
            "top_k_per_signal_day": 2,
            "cash_constrained_no_leverage": True,
            "entry_at_limit_up_blocked": True,
            "signal_selection_uses_future_path": False,
        },
        "splits": {
            "validation": list(windows["validation"]),
            "final_supplemental": list(windows["final_supplemental"]),
            "final_supplemental_is_fresh_unseen_holdout": False,
        },
        "strategy_change": {
            "description": "3.0.0结构重构：以趋势、量能确认、价格位置、隔夜结构和风险效率五组横截面评分替代旧固定系数/概率阈值；加入多指标市场状态过滤，并将研究止盈/止损从1.0%/3.0%调整为2.5%/2.0%。",
            "candidate_strategy_method": candidate["model"].STRATEGY_METHOD,
            "candidate_admission_policy": candidate["policy"].ADMISSION_POLICY,
            "multi_factor_scoring_changed": True,
            "market_regime_filter_changed": True,
            "admission_policy_changed": True,
            "target_return_changed": baseline["target_return_pct"] != candidate["target_return_pct"],
            "stop_loss_changed": baseline["stop_loss_pct"] != candidate["stop_loss_pct"],
            "execution_contract_unchanged": True,
            "historical_research_only": True,
        },
        "results": results,
        "candidate_diagnostics": diagnostics,
        "decision": decision,
        "elapsed_seconds": round(time.time() - started, 3),
        "release_status": "not_released",
        "formal_production_changed": False,
        "automatic_ordering": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="以真实执行约束回放冻结基准与3.0.0结构重构候选的成本后组合表现")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/backtest/private_strategy_replay_latest.json")
    args = parser.parse_args()
    result = run_replay(args.baseline_dir, args.candidate_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "baseline_version": result["strategy_versions"]["baseline"]["version"],
        "candidate_version": result["strategy_versions"]["candidate"]["version"],
        "strict_improvement_over_baseline": result["decision"]["strict_improvement_over_baseline"],
        "decision": result["decision"]["decision"],
        "elapsed_seconds": result["elapsed_seconds"],
        "formal_production_changed": False,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
