"""使用 Aquant-Private/main 当前版本生成研究候选；不读取历史独立神经网络权重。"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.short_term_research import FeatureState, history_files, read_daily  # noqa: E402

VERSION_MANIFEST = ROOT / "config" / "system_version.json"
DEFAULT_OUTPUT = ROOT / "data" / "candidates.json"
MIN_HISTORY_SESSIONS = 20
MAX_CANDIDATES = 2


def _yaml_number(text: str, key: str) -> float:
    pattern = r"^\s*" + re.escape(key) + r"\s*:\s*([-+]?\d+(?:\.\d+)?)\s*$"
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"私有策略配置缺少数值参数：{key}")
    return float(match.group(1))


def load_current_strategy(private_root: Path) -> dict[str, Any]:
    """加载唯一私有策略源，并校验版本号与提交号的一致性。"""
    if not private_root.is_dir():
        raise RuntimeError(f"找不到私有策略检出目录：{private_root}")
    manifest = json.loads(VERSION_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("lifecycle_status") != "research" or manifest.get("release_status") != "not_released":
        raise RuntimeError("当前系统版本清单不是研究阶段，拒绝按研究候选流程运行。")
    if manifest.get("production_release") is not False or manifest.get("automatic_ordering") is not False:
        raise RuntimeError("研究阶段必须关闭正式发布与自动下单。")

    version_file = private_root / "strategy" / "version.py"
    config_file = private_root / "config" / "strategy.yaml"
    if not version_file.is_file() or not config_file.is_file():
        raise RuntimeError("私有策略版本文件或策略配置缺失。")
    config_text = config_file.read_text(encoding="utf-8")
    if 'release_status: "research_only"' not in config_text:
        raise RuntimeError("当前私有策略未标记为仅研究版本。")

    commit = subprocess.check_output(
        ["git", "-C", str(private_root), "rev-parse", "HEAD"], text=True
    ).strip()
    sys.path.insert(0, str(private_root))
    importlib.invalidate_caches()
    version_module = importlib.import_module("strategy.version")
    model_module = importlib.import_module("strategy.model")
    precision_module = importlib.import_module("strategy.precision_config")
    version = str(version_module.STRATEGY_VERSION)

    if version != manifest.get("system_version") or version != manifest.get("strategy_version"):
        raise RuntimeError(
            f"系统/策略版本不一致：系统={manifest.get('system_version')}，策略={version}。"
        )
    if commit != manifest.get("strategy_commit"):
        raise RuntimeError(
            f"策略提交与系统版本清单不一致：清单={manifest.get('strategy_commit')}，实际={commit}。"
        )
    if str(config_text).find(f'version: "{version}"') < 0:
        raise RuntimeError("策略代码版本与策略配置文件版本不一致。")
    minimum_price = _yaml_number(config_text, "min_price")
    minimum_turnover_amount = _yaml_number(config_text, "min_turnover_amount_cny")
    minimum_history_days = int(_yaml_number(config_text, "minimum_history_days"))
    target_return_pct = _yaml_number(config_text, "target_return_pct")
    stop_loss_pct = _yaml_number(config_text, "stop_loss_pct")
    max_holding_sessions = int(_yaml_number(config_text, "max_holding_sessions"))
    if minimum_history_days < MIN_HISTORY_SESSIONS:
        raise RuntimeError("私有策略最低历史长度不足以支撑当前20日特征。")
    if max_holding_sessions != 5:
        raise RuntimeError("当前系统交易契约要求最大持有五个交易日。")
    if target_return_pct <= 0 or stop_loss_pct <= 0:
        raise RuntimeError("私有策略止盈止损参数必须为正。")
    if not np.isclose(sum(float(v) for v in model_module.WEIGHTS.values()), 1.0, atol=1e-9):
        raise RuntimeError("策略因子权重之和不为 1。")
    if int(precision_module.MAX_CANDIDATES) != MAX_CANDIDATES:
        raise RuntimeError("当前策略最大候选数与系统研究契约不一致。")
    return {
        "manifest": manifest,
        "version": version,
        "commit": commit,
        "score_universe": model_module.score_universe,
        "admit_candidates": model_module.admit_candidates,
        "weights": dict(model_module.WEIGHTS),
        "admission_policy": str(model_module.ADMISSION_POLICY),
        "precision_module": precision_module,
        "minimum_price": minimum_price,
        "minimum_turnover_amount": minimum_turnover_amount,
        "minimum_history_days": minimum_history_days,
        "target_return_pct": target_return_pct,
        "stop_loss_pct": stop_loss_pct,
        "max_holding_sessions": max_holding_sessions,
    }


def build_latest_frame(files: list[Path], signal_date: str) -> tuple[pd.DataFrame, int, str]:
    dates = [path.name[:10] for path in files]
    if signal_date not in dates:
        raise ValueError(f"信号日期不在历史数据范围内：{signal_date}")
    end_index = dates.index(signal_date)
    # 当前私有策略最高使用20日回报及10日隔夜特征；预留充分滚动窗口。
    start_index = max(0, end_index - MIN_HISTORY_SESSIONS - 2)
    state = FeatureState()
    frame: pd.DataFrame | None = None
    for path in files[start_index : end_index + 1]:
        frame = state.build(read_daily(path))
    if frame is None or frame.empty:
        raise RuntimeError("信号日没有可评分股票。")
    return frame, end_index + 1, dates[0]


def eligible_universe(frame: pd.DataFrame, minimum_price: float, minimum_turnover_amount: float) -> pd.DataFrame:
    """在横截面评分前执行市场范围、价格、流动性和历史长度门禁。"""
    if frame.empty:
        return frame.copy()
    scored = frame.copy()
    names = scored.get("name", pd.Series("", index=scored.index)).fillna("").astype(str).str.upper()
    close = pd.to_numeric(scored.get("close"), errors="coerce")
    amount = pd.to_numeric(scored.get("amount"), errors="coerce")
    paused = pd.to_numeric(scored.get("is_paused", pd.Series(0, index=scored.index)), errors="coerce").fillna(0)
    is_st = pd.to_numeric(scored.get("is_st", pd.Series(0, index=scored.index)), errors="coerce").fillna(0)
    mask = (
        ~names.str.contains(r"ST|退", case=False, regex=True)
        & close.gt(minimum_price)
        & amount.ge(minimum_turnover_amount)
        & paused.le(0)
        & is_st.le(0)
    )
    for column in ("return_20d_pct", "amount_20d", "volatility_10d_pct", "volume_ratio_5d"):
        if column not in scored.columns:
            raise RuntimeError(f"特征表缺少必要历史特征：{column}")
        mask &= pd.to_numeric(scored[column], errors="coerce").notna()
    for column in ("market_breadth_pct", "market_median_return_pct"):
        if column not in scored.columns:
            raise RuntimeError(f"特征表缺少必要市场状态特征：{column}")
        mask &= pd.to_numeric(scored[column], errors="coerce").notna()
    return scored.loc[mask].copy()


def _number(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = pd.to_numeric(pd.Series([row.get(key, default)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) and np.isfinite(float(value)) else default


def payload_row(row: pd.Series, rank: int) -> dict[str, Any]:
    flags: list[str] = []
    if _number(row, "volatility_10d_pct") > 8:
        flags.append("高波动")
    if _number(row, "change_pct") < -7:
        flags.append("当日大跌")
    if _number(row, "return_5d_pct") > 15:
        flags.append("短期过热")
    if _number(row, "limit_up_5d_count") >= 2:
        flags.append("近期涨停密集")
    if _number(row, "intraday_return_pct") > 8:
        flags.append("日内追高风险")
    return {
        "rank": rank,
        "symbol": str(row["symbol"]).zfill(6),
        "name": str(row["name"]),
        "price": round(_number(row, "close"), 3),
        "change_pct": round(_number(row, "change_pct"), 3),
        "return_3d_pct": round(_number(row, "return_3d_pct"), 3),
        "return_5d_pct": round(_number(row, "return_5d_pct"), 3),
        "return_10d_pct": round(_number(row, "return_10d_pct"), 3),
        "volume_ratio_5d": round(_number(row, "volume_ratio_5d"), 3),
        "turnover_pct": round(_number(row, "turnover_pct"), 3),
        "amount": round(_number(row, "amount"), 2),
        "volatility_10d_pct": round(_number(row, "volatility_10d_pct"), 3),
        "close_strength": round(_number(row, "close_strength", 0.5), 3),
        "overnight_1d_pct": round(_number(row, "overnight_1d_pct"), 3),
        "overnight_3d_pct": round(_number(row, "overnight_3d_pct"), 3),
        "overnight_5d_pct": round(_number(row, "overnight_5d_pct"), 3),
        "intraday_return_pct": round(_number(row, "intraday_return_pct"), 3),
        "limit_up_5d_count": round(_number(row, "limit_up_5d_count"), 3),
        "score": round(_number(row, "score"), 3),
        "precision_probability": round(_number(row, "precision_probability"), 6),
        "admission_tier": str(row.get("admission_tier", "研究观察")),
        "flags": flags,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成当前研究策略候选池；不启用正式生产或自动下单。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--signal-date", default=None)
    parser.add_argument("--private-strategy", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None, help="旧版兼容参数；不允许覆盖策略自身的准入规则")
    args = parser.parse_args()
    if args.threshold is not None:
        raise ValueError("--threshold 已停用；准入阈值必须来自唯一私有策略源。")

    private_root = args.private_strategy or Path(
        os.environ.get("AQUANT_PRIVATE_STRATEGY_PATH", ROOT / "private-strategy")
    ).resolve()
    strategy = load_current_strategy(private_root)
    files = history_files()
    dates = [path.name[:10] for path in files]
    signal_date = args.signal_date or dates[-1]
    if signal_date not in dates:
        raise ValueError(f"信号日期不在历史数据范围内：{signal_date}")

    raw_frame, files_used, data_start = build_latest_frame(files, signal_date)
    market_breadth_all = pd.to_numeric(raw_frame["market_breadth_pct"], errors="coerce").dropna()
    market_median_all = pd.to_numeric(raw_frame["market_median_return_pct"], errors="coerce").dropna()
    if market_breadth_all.empty or market_median_all.empty:
        raise RuntimeError("信号日缺少有效市场广度或市场中位收益，拒绝生成候选。")
    breadth = float(market_breadth_all.iloc[0])
    median_return = float(market_median_all.iloc[0])

    eligible = eligible_universe(raw_frame, strategy["minimum_price"], strategy["minimum_turnover_amount"])
    if eligible.empty:
        raise RuntimeError("通过硬性市场范围、流动性和历史长度门禁的股票数量为零。")
    scored = strategy["score_universe"](eligible)
    selected = strategy["admit_candidates"](scored).copy()
    if len(selected) > MAX_CANDIDATES:
        raise RuntimeError("策略输出候选超过系统允许上限。")
    candidates = [payload_row(row, rank + 1) for rank, (_, row) in enumerate(selected.iterrows())]

    precision = strategy["precision_module"]
    risk_off = (
        breadth < float(precision.MIN_MARKET_BREADTH_PCT)
        or median_return < float(precision.MIN_MARKET_MEDIAN_RETURN_PCT)
    )
    regime = "risk_off" if risk_off else (
        "risk_on" if median_return > 0.3 else "neutral"
    )
    primary_count = sum(row["admission_tier"] == "primary_089" for row in candidates)
    fallback_count = sum(row["admission_tier"] == "coverage_fallback_088" for row in candidates)
    rescue_count = sum(row["admission_tier"] == "daily_top1_rescue" for row in candidates)

    features = list(precision.PRECISION_COEFFICIENTS.keys()) + [
        "market_breadth_pct", "market_median_return_pct"
    ]
    result = {
        "schema_version": 3,
        "system_version": strategy["version"],
        "lifecycle_status": "research_only",
        "as_of": f"{signal_date}T18:00:00+08:00",
        "timezone": "Asia/Shanghai",
        "source": "Aquant-Public data/history",
        "status": "ready",
        "strategy_source": "Aquant-Private/main",
        "strategy_version": strategy["version"],
        "strategy_commit": strategy["commit"],
        "model_definition": {
            "type": "private_short_term_precision_policy",
            "features": features,
            "primary_probability_threshold": float(precision.PRECISION_PROBABILITY_THRESHOLD),
            "maximum_candidates": MAX_CANDIDATES,
            "objective": "研究阶段：验证成本后可执行净收益、收益风险结构与参数稳健性",
            "release_status": "not_released",
        },
        "market_scope": "沪深主板；排除ST、退市、停牌、价格不高于2元及近期成交额低于3000万元的股票",
        "universe": "仅对通过硬性资格、价格、流动性与20日历史长度门禁的股票进行横截面评分",
        "lookback_trading_days": 20,
        "signal_horizon": "T收盘信号 → T+1开盘买入 → T+2起最早卖出 → 最长5个交易日",
        "holding_window_sessions": [2, 5],
        "risk_controls": {
            "net_win_threshold_pct": strategy["target_return_pct"],
            "stop_loss_pct": strategy["stop_loss_pct"],
            "max_positions": MAX_CANDIDATES,
            "sell_start_session": 2,
            "max_holding_sessions": strategy["max_holding_sessions"],
        },
        "market": {
            "breadth_pct": round(breadth, 3),
            "median_return_pct": round(median_return, 3),
            "regime": regime,
        },
        "diagnostics": {
            "scorable_rows": int(len(eligible)),
            "eligible_rows": int(len(eligible)),
            "candidate_count": len(candidates),
            "candidate_probability_min": min((r["precision_probability"] for r in candidates), default=None),
            "candidate_probability_max": max((r["precision_probability"] for r in candidates), default=None),
            "primary_candidate_count": primary_count,
            "coverage_fallback_used": fallback_count == 1,
            "daily_top1_rescue_used": rescue_count == 1,
            "risk_off_no_trade": risk_off,
        },
        "candidates": candidates,
        "research_observation_candidates": [
            payload_row(row, rank + 1)
            for rank, (_, row) in enumerate(scored.head(10).iterrows())
        ],
        "factor_weights": strategy["weights"],
        "future_function": False,
        "candidate_admission_policy": strategy["admission_policy"],
        "audit": {
            "hard_eligibility_applied_before_scoring": True,
            "strategy_source_locked_to_private": True,
            "short_term_features_only": True,
            "market_gate_applied": True,
            "no_future_features": True,
            "entry_is_T_plus_1_open": True,
            "exit_starts_T_plus_2": True,
            "strict_profit_label": True,
            "final_holdout_not_used_for_candidate_selection": True,
            "formal_production_changed": False,
            "automatic_ordering": False,
        },
        "history_files_used": files_used,
        "history_window_start": data_start,
        "history_window_end": signal_date,
        "inference_data_only_to_signal_date": True,
        "model_artifact": "Aquant-Private/strategy/model.py",
        "production_release": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ready",
        "lifecycle_status": "research_only",
        "system_version": strategy["version"],
        "strategy_commit": strategy["commit"],
        "signal_date": signal_date,
        "scorable_rows": len(eligible),
        "candidate_count": len(candidates),
        "market_regime": regime,
        "candidates": [
            {"symbol": row["symbol"], "name": row["name"], "probability": row["precision_probability"]}
            for row in candidates
        ],
        "production_release": False,
        "automatic_ordering": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
