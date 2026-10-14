from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.next_trading_day_plan import set_production_release, validate_plan
from scripts.validate_candidates import validate_candidates

SYSTEM_VERSION_PATH = ROOT / "config" / "system_version.json"
CANDIDATES_PATH = ROOT / "data" / "candidates.json"
PLAN_PATH = ROOT / "data" / "next_trading_day_plan.json"
PRODUCTION_STATUS_PATH = ROOT / "data" / "production_status.json"
AUDIT_PATH = ROOT / "data" / "system_audit.json"
RELEASE_EVIDENCE_PATH = ROOT / "data" / "backtest" / "strategy_release_gate.json"

REQUIRED_RELEASE_CHECKS = {
    "validation_minimum_trades",
    "validation_positive_compound_return",
    "validation_positive_mean_trade_net_return",
    "validation_profit_factor",
    "validation_max_drawdown",
    "supplementary_positive_compound_return",
    "supplementary_positive_mean_trade_net_return",
    "supplementary_profit_factor",
    "supplementary_max_drawdown",
    "cost_stress_test",
    "adjacent_parameter_robustness",
    "shadow_minimum_days",
    "shadow_minimum_trades",
    "shadow_positive_compound_return",
    "shadow_positive_mean_trade_net_return",
    "shadow_max_drawdown",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"缺少必要文件：{path.relative_to(ROOT)}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON格式无效：{path.relative_to(ROOT)}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON根节点必须是对象：{path.relative_to(ROOT)}")
    return value


def required_marker(path: str, expected_status: str | None = None) -> dict:
    value = load_json(ROOT / path)
    if expected_status is not None and value.get("status") != expected_status:
        raise RuntimeError(f"{path} 状态不是 {expected_status}")
    return value


def evaluate_release_evidence(manifest: dict) -> dict:
    if not RELEASE_EVIDENCE_PATH.exists():
        return {
            "present": False,
            "passed": False,
            "checks": {},
            "reason": "尚无与当前策略版本及提交号匹配的完整发布证据。",
        }
    evidence = load_json(RELEASE_EVIDENCE_PATH)
    checks = evidence.get("checks", {})
    current_version = (
        evidence.get("system_version") == manifest.get("system_version")
        and evidence.get("strategy_version") == manifest.get("strategy_version")
        and evidence.get("strategy_commit") == manifest.get("strategy_commit")
    )
    required_checks_passed = (
        set(REQUIRED_RELEASE_CHECKS).issubset(checks)
        and all(checks.get(key) is True for key in REQUIRED_RELEASE_CHECKS)
    )
    passed = (
        evidence.get("status") == "passed"
        and current_version
        and required_checks_passed
    )
    return {
        "present": True,
        "passed": bool(passed),
        "current_version_match": bool(current_version),
        "checks": {key: checks.get(key) is True for key in sorted(REQUIRED_RELEASE_CHECKS)},
        "reason": "发布证据通过" if passed else "发布证据缺失、版本不匹配或至少一项硬门槛未通过。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="核验研究阶段版本一致性；没有发布证据时始终保持未发布状态。")
    parser.add_argument("--output", default="data/system_audit.json")
    parser.add_argument("--allow-release", action="store_true", help="即便显式传入，也必须同时满足版本清单与完整发布证据")
    args = parser.parse_args()

    manifest = load_json(SYSTEM_VERSION_PATH)
    system_version = str(manifest.get("system_version") or "")
    strategy_version = str(manifest.get("strategy_version") or "")
    strategy_commit = str(manifest.get("strategy_commit") or "")
    if not system_version or system_version != strategy_version:
        raise RuntimeError("系统版本与策略版本不一致，拒绝通过系统审计。")
    if not strategy_commit:
        raise RuntimeError("系统版本清单缺少策略提交号。")
    if manifest.get("strategy_source_repo") != "13415837267/Aquant-Private" or manifest.get("strategy_source_branch") != "main":
        raise RuntimeError("策略来源必须指向个人策略仓库 main 分支。")

    history_complete = required_marker("data/history/_BACKFILL_COMPLETE", "complete")
    history_state = required_marker("data/history/_BACKFILL_STATE.json", "complete")
    fundamentals_complete = required_marker("data/fundamentals/_FUNDAMENTALS_COMPLETE")
    fundamentals_state = load_json(ROOT / "data" / "fundamentals" / "_FUNDAMENTALS_STATE.json")
    if fundamentals_state.get("status") != "complete" or fundamentals_complete.get("point_in_time_fields") != ["report_date", "pub_date"]:
        raise RuntimeError("财务数据未通过公告时点字段审计。")

    candidates = load_json(CANDIDATES_PATH)
    production_before = load_json(PRODUCTION_STATUS_PATH)
    candidate_current = bool(
        candidates.get("status") == "ready"
        and candidates.get("future_function") is False
        and candidates.get("system_version") == system_version
        and candidates.get("strategy_version") == strategy_version
        and candidates.get("strategy_commit") == strategy_commit
    )
    candidate_check = {"status": "stale", "reason": "当前候选快照尚未按系统版本重新生成。"}
    if candidate_current:
        candidate_check = validate_candidates(candidates, strategy_version, strategy_commit)

    plan_current = False
    plan_check: dict = {"status": "stale", "reason": "当前下一交易日计划尚未按当前候选快照重建。"}
    if candidate_current:
        plan = load_json(PLAN_PATH)
        if (
            plan.get("strategy_version") == strategy_version
            and plan.get("strategy_commit") == strategy_commit
            and plan.get("source_snapshot", {}).get("as_of") == candidates.get("as_of")
        ):
            plan_check = validate_plan(plan, candidates, production_before)
            plan_current = plan_check.get("status") == "pass"

    evidence = evaluate_release_evidence(manifest)
    lifecycle_approved = manifest.get("lifecycle_status") == "production_approved" and manifest.get("release_status") == "approved"
    production_released = bool(
        args.allow_release
        and lifecycle_approved
        and evidence.get("passed") is True
        and candidate_current
        and plan_current
    )
    release_status = "released" if production_released else "research_only"

    production_payload = {
        "schema_version": 4,
        "status": release_status,
        "system_version": system_version,
        "strategy_version": strategy_version,
        "lifecycle_status": "released" if production_released else "research",
        "production_version": system_version if production_released else None,
        "production_strategy_commit": strategy_commit if production_released else None,
        "as_of": candidates.get("as_of") if production_released else None,
        "release_gate": production_released,
        "system_audit": bool(candidate_current and plan_current and history_complete.get("status") == "complete" and history_state.get("status") == "complete"),
        "strategy_quality_gate_passed": production_released,
        "strategy_source_repo": manifest["strategy_source_repo"],
        "strategy_source_branch": manifest["strategy_source_branch"],
        "strategy_source_commit": strategy_commit,
        "candidate_current": candidate_current,
        "plan_current": plan_current,
        "release_evidence_passed": bool(evidence.get("passed")),
        "automatic_ordering": False,
        "note": (
            "当前版本已通过版本、候选和计划一致性审计，并已具备完整发布证据；正式发布仍由单独授权控制。"
            if production_released
            else "当前系统仍处于研究/模拟阶段；系统版本与策略版本一一对应，但未完成正式发布门禁。禁止实盘下单与自动订单执行。"
        ),
    }
    PRODUCTION_STATUS_PATH.write_text(json.dumps(production_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if plan_current:
        set_production_release(PLAN_PATH, production_released)
        plan = load_json(PLAN_PATH)
        plan_check = validate_plan(plan, candidates, production_payload)

    payload = {
        "schema_version": 4,
        "status": "current_research_audit" if candidate_current and plan_current else "research_version_alignment_pending_audit",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "timezone": "Asia/Shanghai",
        "system_version": system_version,
        "strategy_version": strategy_version,
        "strategy_source": "Aquant-Private/main",
        "strategy_commit": strategy_commit,
        "lifecycle_status": production_payload["lifecycle_status"],
        "historical_database": "complete_incremental",
        "fundamentals_database": "complete_pit",
        "candidate_layer": "current" if candidate_current else "stale_until_current_strategy_regeneration",
        "candidate_check": candidate_check,
        "plan_check": plan_check,
        "production_version": production_payload["production_version"],
        "production_gate": "passed" if production_released else "research_only",
        "release_evidence": evidence,
        "future_function": False,
        "automatic_ordering": False,
        "audit": {
            "system_and_strategy_version_match": system_version == strategy_version,
            "strategy_commit_used_as_snapshot_provenance": bool(strategy_commit),
            "next_run_resolves_latest_private_main": True,
            "candidate_snapshot_matches_current_strategy": candidate_current,
            "next_trading_day_plan_matches_current_strategy": plan_current,
            "legacy_neural_model_is_archived": True,
            "history_completion_checked": True,
            "fundamentals_point_in_time_checked": True,
            "production_release_enabled": production_released,
            "automatic_ordering_disabled": True,
            "cloud_runtime_only": True,
        },
        "note": (
            "每次候选运行都重新读取 Aquant-Private/main 最新提交；提交号只记录本次快照来源，不固定历史提交。旧候选或旧模型与当前策略不一致时不展示；当前未发布正式版本，也不自动下单。"
        ),
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
