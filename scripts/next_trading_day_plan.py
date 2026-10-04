from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.backfill_history import api_client

DEFAULT_CANDIDATES = ROOT / "data" / "candidates.json"
DEFAULT_OUTPUT = ROOT / "data" / "next_trading_day_plan.json"
DEFAULT_PRODUCTION = ROOT / "data" / "production_status.json"
TZ_NAME = "Asia/Shanghai"
PLAN_SCHEMA_VERSION = 3


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"缺少文件: {path.relative_to(ROOT)}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON 无效: {path.relative_to(ROOT)}") from exc


def _calendar_dates(signal_date: date, horizon_days: int = 60) -> list[str]:
    api = api_client()
    end = signal_date + timedelta(days=horizon_days)
    raw = api.trade_days(
        day_start=signal_date.strftime("%Y%m%d"),
        day_end=end.strftime("%Y%m%d"),
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError("交易日历为空，无法生成下一交易日计划")
    frame = raw.copy() if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
    date_col = next(
        (c for c in ("trade_date", "cal_date", "date", "日期") if c in frame.columns),
        frame.columns[0] if len(frame.columns) == 1 else None,
    )
    if date_col is None:
        raise RuntimeError(f"无法识别交易日历日期列: {list(frame.columns)}")
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    if "is_open" in frame.columns:
        opened = pd.to_numeric(frame["is_open"], errors="coerce").fillna(0).astype(bool)
        dates = dates.where(opened)
    out = sorted(set(dates.dropna().dt.strftime("%Y-%m-%d").tolist()))
    return [d for d in out if d >= signal_date.isoformat()]


def _next_sessions(signal_date: date) -> tuple[str, str]:
    future = [d for d in _calendar_dates(signal_date) if d > signal_date.isoformat()]
    if len(future) < 2:
        raise RuntimeError("交易日历未来窗口不足两个交易日")
    return future[0], future[1]


def _production_release(production: dict) -> bool:
    return bool(
        production.get("status") == "released"
        and production.get("release_gate") is True
        and production.get("system_audit") is True
    )


def build_plan(candidates: dict, production: dict) -> dict:
    if candidates.get("status") != "ready":
        raise RuntimeError("只有 ready 候选快照才能生成下一交易日计划")
    signal_date = date.fromisoformat(str(candidates["as_of"])[:10])
    execution_date, earliest_exit_date = _next_sessions(signal_date)
    rows = []
    for candidate in candidates.get("candidates", []):
        rows.append(
            {
                "rank": int(candidate["rank"]),
                "symbol": str(candidate["symbol"]).zfill(6),
                "name": str(candidate["name"]),
                "price": float(candidate["price"]),
                "score": float(candidate["score"]),
                "precision_probability": float(candidate["precision_probability"]),
                "admission_tier": str(candidate["admission_tier"]),
                "signal_date": signal_date.isoformat(),
                "execution_date": execution_date,
                "earliest_exit_date": earliest_exit_date,
                "net_win_threshold_pct": 1.0,
                "stop_loss_pct": 3.0,
                "flags": list(candidate.get("flags", [])),
            }
        )
    released = _production_release(production)
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "generated_at": pd.Timestamp.now(tz=TZ_NAME).isoformat(),
        "timezone": TZ_NAME,
        "next_trading_day": execution_date,
        "signal_date": signal_date.isoformat(),
        "data_cutoff": signal_date.isoformat(),
        "status": "production_plan" if released else "research_plan",
        "production_release": released,
        "title": f"{execution_date[:4]}年{int(execution_date[5:7])}月{int(execution_date[8:10])}日下一交易日计划",
        "summary": (
            f"{signal_date.isoformat()} 收盘候选快照直接作为 {execution_date} 开盘执行计划。"
            "候选身份、排序、分数与精度概率全部逐项继承 data/candidates.json，"
            "不重新选股、不重新打分，不读取执行日或之后的行情数据，因此不存在未来函数。"
        ),
        "strategy_version": candidates.get("strategy_version"),
        "strategy_commit": candidates.get("strategy_commit"),
        "candidate_policy": candidates.get("candidate_admission_policy"),
        "market": candidates.get("market"),
        "candidate_count": len(rows),
        "candidates": rows,
        "steps": [
            {
                "time": f"{execution_date} 盘前",
                "action": f"仅核对计划候选与开盘前可执行性及统一硬风控；不使用 {execution_date} 收盘以后产生的行情数据。",
            },
            {
                "time": f"{execution_date} 开盘",
                "action": f"若仅作为研究/模拟执行，按 {signal_date.isoformat()} 收盘形成的信号对应 T+1 开盘执行；正式生产门禁未通过时禁止实盘。",
            },
            {
                "time": f"{execution_date} 收盘后",
                "action": f"把 {execution_date} 作为新的 T 日，以该日收盘及此前数据重新生成下一交易日计划。",
            },
            {
                "time": f"{earliest_exit_date} 起",
                "action": f"对 {execution_date} 新信号严格执行 T+2 最早卖出规则；最长持有 5 个完整交易日。",
            },
        ],
        "hard_rules": [
            "下一交易日计划的候选唯一来源为 data/candidates.json，网页不得维护第二套候选列表。",
            "只使用信号日收盘及之前已经存在的数据；不得使用未来交易日的价格、最高、最低、成交量或结果标签参与选股。",
            "T 日收盘产生信号，T+1 开盘执行，T+2 起最早卖出，最长持有 5 个完整交易日。",
            "风险规避状态或全部候选未通过统一硬风控时允许空池。",
            "production_release 为 false 时，计划只能用于研究/模拟，不得视为正式生产信号。",
        ],
        "source_snapshot": {
            "path": "data/candidates.json",
            "as_of": candidates.get("as_of"),
            "strategy_version": candidates.get("strategy_version"),
            "strategy_commit": candidates.get("strategy_commit"),
        },
    }
    validate_plan(plan, candidates, production)
    return plan


def validate_plan(plan: dict, candidates: dict, production: dict) -> dict:
    if not isinstance(plan, dict):
        raise RuntimeError("下一交易日计划必须是对象")
    if plan.get("status") not in {"research_plan", "production_plan"}:
        raise RuntimeError("下一交易日计划状态无效")
    if plan.get("signal_date") != str(candidates.get("as_of", ""))[:10]:
        raise RuntimeError("计划 signal_date 与候选快照 as_of 不一致")
    if plan.get("data_cutoff") != plan.get("signal_date"):
        raise RuntimeError("计划 data_cutoff 必须等于 signal_date")
    for key in ("strategy_version", "strategy_commit"):
        if plan.get(key) != candidates.get(key):
            raise RuntimeError(f"计划 {key} 与候选快照不一致")
    source = plan.get("source_snapshot") or {}
    expected_source = {
        "path": "data/candidates.json",
        "as_of": candidates.get("as_of"),
        "strategy_version": candidates.get("strategy_version"),
        "strategy_commit": candidates.get("strategy_commit"),
    }
    if any(source.get(k) != v for k, v in expected_source.items()):
        raise RuntimeError("计划 source_snapshot 与候选快照不一致")
    source_rows = candidates.get("candidates", [])
    planned = plan.get("candidates")
    if plan.get("candidate_count") != len(source_rows) or not isinstance(planned, list) or len(planned) != len(source_rows):
        raise RuntimeError("计划候选数量与候选快照不一致")
    execution = date.fromisoformat(plan["next_trading_day"])
    signal = date.fromisoformat(plan["signal_date"])
    if execution <= signal:
        raise RuntimeError("next_trading_day 必须晚于 signal_date")
    for expected, actual in zip(source_rows, planned):
        for key in ("rank", "symbol", "name", "price", "score", "precision_probability", "admission_tier"):
            if actual.get(key) != expected.get(key):
                raise RuntimeError(f"计划候选字段不一致: {key}")
        if actual.get("signal_date") != plan["signal_date"] or actual.get("execution_date") != plan["next_trading_day"]:
            raise RuntimeError("计划候选日期不一致")
        if date.fromisoformat(actual.get("earliest_exit_date", "1900-01-01")) <= execution:
            raise RuntimeError("最早退出必须晚于执行日")
        if actual.get("net_win_threshold_pct") != 1.0 or actual.get("stop_loss_pct") != 3.0:
            raise RuntimeError("计划硬风控阈值不一致")
    expected_release = _production_release(production)
    if plan.get("production_release") is not expected_release:
        raise RuntimeError("计划 production_release 与生产状态不一致")
    if plan.get("status") != ("production_plan" if expected_release else "research_plan"):
        raise RuntimeError("计划状态与生产状态不一致")
    return {
        "status": "pass",
        "signal_date": plan["signal_date"],
        "next_trading_day": plan["next_trading_day"],
        "candidate_count": plan["candidate_count"],
        "strategy_version": plan["strategy_version"],
        "strategy_commit": plan["strategy_commit"],
    }


def set_production_release(path: Path, released: bool) -> None:
    plan = _load_json(path)
    plan["production_release"] = bool(released)
    plan["status"] = "production_plan" if released else "research_plan"
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--production-status", type=Path, default=DEFAULT_PRODUCTION)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    candidates = _load_json(args.candidates)
    production = _load_json(args.production_status)
    if args.validate:
        print(json.dumps(validate_plan(_load_json(args.output), candidates, production), ensure_ascii=False))
        return
    plan = build_plan(candidates, production)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ready",
        "signal_date": plan["signal_date"],
        "next_trading_day": plan["next_trading_day"],
        "candidate_count": plan["candidate_count"],
        "strategy_version": plan["strategy_version"],
        "strategy_commit": plan["strategy_commit"],
        "production_release": plan["production_release"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
