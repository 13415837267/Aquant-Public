from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.market_scope import is_main_board_symbol

DEFAULT_CANDIDATES = ROOT / "data" / "candidates.json"
REQUIRED_CANDIDATE_FIELDS = {
    "rank", "symbol", "name", "price", "change_pct",
    "return_3d_pct", "return_5d_pct", "return_10d_pct",
    "volume_ratio_5d", "turnover_pct", "amount",
    "volatility_10d_pct", "close_strength", "score",
    "precision_probability", "admission_tier",
}


def finite(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} 必须是数值") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{field} 必须为有限数值")
    return number


def date_only(value: object, field: str) -> str:
    try:
        return date.fromisoformat(str(value)[:10]).isoformat()
    except ValueError as exc:
        raise RuntimeError(f"{field} 日期无效") from exc


def current_private_contract() -> dict | None:
    root_text = os.environ.get("AQUANT_PRIVATE_STRATEGY_PATH")
    if not root_text:
        return None
    root = Path(root_text).resolve()
    if not (root / "strategy" / "version.py").is_file():
        raise RuntimeError("个人策略仓库缺少版本文件")
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    version_module = importlib.import_module("strategy.version")
    model_module = importlib.import_module("strategy.model")
    precision_module = importlib.import_module("strategy.precision_config")
    commit = subprocess.check_output(["git", "-C", root_text, "rev-parse", "HEAD"], text=True).strip()
    return {
        "version": str(version_module.STRATEGY_VERSION),
        "commit": commit,
        "policy": str(model_module.ADMISSION_POLICY),
        "max_candidates": int(precision_module.MAX_CANDIDATES),
        "weights": {str(key): float(value) for key, value in model_module.WEIGHTS.items()},
    }


def private_provenance() -> tuple[str | None, str | None]:
    contract = current_private_contract()
    if contract is None:
        return None, None
    return contract["version"], contract["commit"]


def validate_candidates(payload, private_version=None, private_commit=None):
    if not isinstance(payload, dict):
        raise RuntimeError("候选快照必须是对象")
    if payload.get("status") != "ready" or payload.get("future_function") is not False:
        raise RuntimeError("候选快照未通过就绪状态或点时数据审计")
    as_of = str(payload.get("as_of") or "")
    if not as_of.endswith("T18:00:00+08:00"):
        raise RuntimeError("候选生成时间戳无效")
    if date_only(as_of, "as_of") != date_only(payload.get("history_window_end"), "history_window_end"):
        raise RuntimeError("候选生成日期必须与历史窗口结束日期一致")
    if date_only(payload.get("history_window_start"), "history_window_start") > date_only(payload.get("history_window_end"), "history_window_end"):
        raise RuntimeError("历史窗口起始日期不能晚于结束日期")
    if not isinstance(payload.get("history_files_used"), int) or payload["history_files_used"] < 20:
        raise RuntimeError("实际使用的历史文件数量不得少于20")
    if payload.get("lookback_trading_days") != 20:
        raise RuntimeError("当前短线特征窗口必须为20个交易日")
    if payload.get("signal_horizon") != "T收盘信号 → T+1开盘买入 → T+2起最早卖出 → 最长5个交易日":
        raise RuntimeError("短线信号与交易时序约束不一致")
    if payload.get("strategy_source") != "Aquant-Private/main":
        raise RuntimeError("策略来源不是个人策略仓库 main 分支")

    contract = current_private_contract()
    if contract is not None:
        if private_version is not None and str(private_version) != contract["version"]:
            raise RuntimeError("系统版本清单与个人策略仓库最新版本不一致")
        if private_commit is not None and str(private_commit) != contract["commit"]:
            raise RuntimeError("系统版本清单与个人策略仓库最新提交不一致")
        private_version = contract["version"]
        private_commit = contract["commit"]

    version = str(payload.get("strategy_version") or "")
    commit = str(payload.get("strategy_commit") or "")
    if not version or not commit:
        raise RuntimeError("缺少本轮策略来源版本或提交记录")
    if payload.get("system_version") != version:
        raise RuntimeError("系统版本与策略版本不一致")
    if private_version is not None and version != private_version:
        raise RuntimeError("候选策略版本与本轮实际策略版本不一致")
    if private_commit is not None and commit != private_commit:
        raise RuntimeError("候选策略提交与本轮实际策略提交不一致")

    policy = str(payload.get("candidate_admission_policy") or "")
    if not policy:
        raise RuntimeError("缺少本轮策略的候选准入政策")
    if contract is not None and policy != contract["policy"]:
        raise RuntimeError("候选准入政策与个人策略仓库最新策略不一致")

    weights = payload.get("factor_weights")
    if not isinstance(weights, dict) or not weights or any(not isinstance(key, str) or not key for key in weights):
        raise RuntimeError("因子权重格式无效")
    weight_values = {key: finite(value, f"因子权重.{key}") for key, value in weights.items()}
    if abs(sum(weight_values.values()) - 1.0) > 1e-9:
        raise RuntimeError("因子权重之和必须为1")
    if contract is not None:
        expected_weights = contract["weights"]
        if set(weight_values) != set(expected_weights) or any(
            not math.isclose(weight_values[key], expected_weights[key], rel_tol=0.0, abs_tol=1e-9)
            for key in expected_weights
        ):
            raise RuntimeError("候选因子权重与个人策略仓库最新策略不一致")

    model_definition = payload.get("model_definition")
    if not isinstance(model_definition, dict):
        raise RuntimeError("缺少策略模型定义")
    if model_definition.get("type") != "private_short_term_precision_policy":
        raise RuntimeError("当前策略模型接口类型无效")
    expected_maximum = contract["max_candidates"] if contract else model_definition.get("maximum_candidates")
    if model_definition.get("maximum_candidates") != expected_maximum:
        raise RuntimeError("候选数量上限与本轮策略不一致")
    try:
        maximum_candidates = int(model_definition.get("maximum_candidates"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("候选数量上限无效") from exc
    if maximum_candidates < 1 or maximum_candidates > 100:
        raise RuntimeError("候选数量上限超出系统安全范围")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > maximum_candidates:
        raise RuntimeError("候选数量无效")
    risk = payload.get("risk_controls")
    if not isinstance(risk, dict):
        raise RuntimeError("缺少本轮策略风险控制参数")
    if finite(risk.get("net_win_threshold_pct"), "风险控制.止盈阈值") <= 0:
        raise RuntimeError("策略止盈阈值必须为正数")
    if finite(risk.get("stop_loss_pct"), "风险控制.止损阈值") <= 0:
        raise RuntimeError("策略止损阈值必须为正数")
    if risk.get("max_positions") != maximum_candidates:
        raise RuntimeError("持仓上限与本轮策略候选数量上限不一致")
    if risk.get("max_holding_sessions") != 5 or payload.get("holding_window_sessions") != [2, 5]:
        raise RuntimeError("A股 T+1/T+2 与最长五个交易日约束不一致")

    previous = math.inf
    seen = set()
    for rank, row in enumerate(candidates, 1):
        missing = REQUIRED_CANDIDATE_FIELDS - set(row)
        if missing:
            raise RuntimeError(f"第{rank}个候选缺少字段：{sorted(missing)}")
        if row.get("rank") != rank:
            raise RuntimeError("候选排名必须连续且从1开始")
        symbol = str(row.get("symbol", "")).zfill(6)
        if not is_main_board_symbol(symbol):
            raise RuntimeError(f"{symbol} 不属于沪深主板范围")
        if symbol in seen:
            raise RuntimeError("候选池存在重复股票代码")
        seen.add(symbol)
        score = finite(row["score"], f"{symbol}.评分")
        probability = finite(row["precision_probability"], f"{symbol}.精度概率")
        if not 0 <= score <= 100:
            raise RuntimeError(f"{symbol} 评分超出0至100范围")
        if not 0 <= probability <= 1:
            raise RuntimeError(f"{symbol} 精度概率超出0至1范围")
        tier = str(row.get("admission_tier") or "").strip()
        if not tier:
            raise RuntimeError(f"{symbol} 缺少准入层级说明")
        if score > previous + 1e-9:
            raise RuntimeError("候选未按评分从高到低排序")
        previous = score
        for field in REQUIRED_CANDIDATE_FIELDS - {"rank", "symbol", "name", "admission_tier"}:
            finite(row[field], f"{symbol}.{field}")
        if not str(row.get("name") or "").strip():
            raise RuntimeError(f"{symbol} 股票名称为空")

    market = payload.get("market")
    if not isinstance(market, dict) or market.get("regime") not in {"risk_on", "neutral", "risk_off"}:
        raise RuntimeError("市场状态无效")
    diag = payload.get("diagnostics")
    if not isinstance(diag, dict) or diag.get("candidate_count") != len(candidates):
        raise RuntimeError("候选诊断数量与实际数量不一致")
    if diag.get("risk_off_no_trade") != (market["regime"] == "risk_off"):
        raise RuntimeError("风险规避状态与候选准入审计不一致")
    audit = payload.get("audit", {})
    for key in (
        "hard_eligibility_applied_before_scoring",
        "strategy_source_tracks_latest_private_main",
        "short_term_features_only",
        "market_gate_applied",
    ):
        if audit.get(key) is not True:
            raise RuntimeError(f"策略审计未通过：{key}")
    return {
        "status": "pass",
        "as_of": as_of,
        "candidate_count": len(candidates),
        "strategy_version": version,
        "strategy_commit": commit,
        "market_regime": market["regime"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_CANDIDATES)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    pv, pc = private_provenance()
    print(json.dumps(validate_candidates(payload, pv, pc), ensure_ascii=False))


if __name__ == "__main__":
    main()
