from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_READY = [
    "data/backtest/latest.json",
    "data/backtest/walk_forward.json",
    "data/backtest/sensitivity.json",
    "data/backtest/factor_ablation.json",
    "data/backtest/factor_ablation_walk_forward.json",
    "data/backtest/regime_factor_diagnostics.json",
]

def load(path: str) -> dict:
    p = ROOT / path
    if not p.exists():
        raise RuntimeError(f"missing required artifact: {path}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON artifact: {path}") from exc

def require_ready(path: str, payload: dict) -> None:
    if payload.get("status") != "ready":
        raise RuntimeError(f"{path}: status is not ready")
    if payload.get("future_function") is not False:
        raise RuntimeError(f"{path}: future_function audit failed")

def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Aquant production/research layer readiness")
    parser.add_argument("--output", default="data/system_audit.json")
    args = parser.parse_args()

    history = load("data/history/_BACKFILL_COMPLETE")
    history_state = load("data/history/_BACKFILL_STATE.json")
    if history.get("status") != "complete" or history_state.get("status") != "complete":
        raise RuntimeError("historical market database is not complete")
    if history.get("trading_days") != history_state.get("trading_days"):
        raise RuntimeError("history completion metadata mismatch")

    fund_complete = load("data/fundamentals/_FUNDAMENTALS_COMPLETE")
    fund_state = load("data/fundamentals/_FUNDAMENTALS_STATE.json")
    if fund_state.get("status") != "complete":
        raise RuntimeError("fundamentals database is not complete")
    if fund_complete.get("point_in_time_fields") != ["report_date", "pub_date"]:
        raise RuntimeError("fundamentals PIT metadata is invalid")

    candidates = load("data/candidates.json")
    portfolio = load("data/portfolio.json")
    execution = load("data/execution_plan.json")
    require_ready("data/candidates.json", candidates)
    require_ready("data/portfolio.json", portfolio)
    if execution.get("status") != "ready_for_next_open_recheck":
        raise RuntimeError("execution plan is not ready_for_next_open_recheck")
    if execution.get("audit", {}).get("future_function") is not False:
        raise RuntimeError("execution plan future-function audit failed")

    for key in ["strategy_source", "strategy_version", "strategy_commit"]:
        if candidates.get(key) != portfolio.get(key) or candidates.get(key) != execution.get(key):
            raise RuntimeError(f"strategy metadata mismatch: {key}")

    if len(candidates.get("candidates", [])) != candidates.get("diagnostics", {}).get("candidate_count"):
        raise RuntimeError("candidate count audit mismatch")
    if float(portfolio.get("gross_target_weight", 0)) > 1.0 + 1e-8:
        raise RuntimeError("portfolio gross target exceeds 100%")
    if portfolio.get("audit", {}).get("long_only") is not True or portfolio.get("audit", {}).get("leverage") is not False:
        raise RuntimeError("portfolio leverage audit failed")
    if float(execution.get("summary", {}).get("turnover", 0)) > float(execution.get("turnover_cap", 0)) + 1e-8:
        raise RuntimeError("execution plan turnover cap audit failed")

    for path in REQUIRED_READY:
        require_ready(path, load(path))

    constrained_path = ROOT / "data/backtest/execution_constrained.json"
    cloud_marker = ROOT / "data/backtest/_CONSTRAINED_CLOUD_RUN.json"
    constrained_state = "pending_cloud_evidence"
    if constrained_path.exists() and cloud_marker.exists():
        constrained = load("data/backtest/execution_constrained.json")
        marker = load("data/backtest/_CONSTRAINED_CLOUD_RUN.json")
        require_ready("data/backtest/execution_constrained.json", constrained)
        if marker.get("status") != "success" or marker.get("run_id") is None:
            raise RuntimeError("constrained cloud marker is invalid")
        if constrained.get("strategy_commit") != candidates.get("strategy_commit"):
            raise RuntimeError("constrained strategy commit differs from production candidate commit")
        constrained_state = "cloud_verified"

    payload = {
        "schema_version": 1,
        "status": "ready",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "historical_database": "complete",
        "fundamentals_database": "complete_pit",
        "candidate_layer": "ready",
        "portfolio_layer": "ready",
        "execution_plan_layer": "ready_for_next_open_recheck",
        "research_artifacts": "ready",
        "constrained_backtest": constrained_state,
        "strategy_source": candidates["strategy_source"],
        "strategy_version": candidates["strategy_version"],
        "strategy_commit": candidates["strategy_commit"],
        "future_function": False,
        "audit": {
            "strategy_metadata_locked": True,
            "cross_layer_strategy_consistency": True,
            "history_completion_cross_checked": True,
            "fundamentals_pit_metadata_checked": True,
            "portfolio_no_leverage_checked": True,
            "execution_turnover_checked": True,
        },
    }

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))

if __name__ == "__main__":
    main()
