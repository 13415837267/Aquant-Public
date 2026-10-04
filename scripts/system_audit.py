from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.validate_candidates import validate_candidates
from scripts.next_trading_day_plan import set_production_release, validate_plan

REQUIRED_READY=[
    "data/backtest/short_term_latest.json",
    "data/backtest/short_term_sensitivity.json",
    "data/backtest/short_term_exit_diagnostics.json",
    "data/backtest/short_term_release_validation.json",
]

def load(path):
    p=ROOT/path
    if not p.exists(): raise RuntimeError(f"missing required artifact: {path}")
    try: return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc: raise RuntimeError(f"invalid JSON: {path}") from exc

def ready(path,p):
    if p.get("status")!="ready": raise RuntimeError(f"{path}: not ready")
    if p.get("future_function") is not False: raise RuntimeError(f"{path}: PIT audit failed")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="data/system_audit.json")
    args=ap.parse_args()
    hc=load("data/history/_BACKFILL_COMPLETE"); hs=load("data/history/_BACKFILL_STATE.json")
    if hc.get("status")!="complete" or hs.get("status")!="complete": raise RuntimeError("historical database incomplete")
    fc=load("data/fundamentals/_FUNDAMENTALS_COMPLETE"); fs=load("data/fundamentals/_FUNDAMENTALS_STATE.json")
    if fs.get("status")!="complete" or fc.get("point_in_time_fields")!=["report_date","pub_date"]:
        raise RuntimeError("fundamentals database is not complete/PIT")
    candidates=load("data/candidates.json"); ready("data/candidates.json",candidates); validate_candidates(candidates)
    production_status=load("data/production_status.json")
    if production_status.get("status") not in {"not_released","release_candidate","released"}:
        raise RuntimeError("invalid production status")
    for path in REQUIRED_READY: ready(path,load(path))
    research=load(REQUIRED_READY[0])
    if int(research.get("signal_days",0))<500: raise RuntimeError("short-term research coverage too small")
    if int(research.get("candidate_days",0))<50: raise RuntimeError("short-term candidate sample too small")

    release=load("data/backtest/short_term_release_validation.json")
    research_strategy_version=release.get("strategy_version")
    research_strategy_commit=release.get("strategy_commit")
    validation = release.get("validation", {})
    final_holdout = release.get("final_holdout", {})
    for label, window in (("validation", validation), ("final_holdout", final_holdout)):
        if int(window.get("signal_days",0)) < 100:
            raise RuntimeError(f"{label} coverage too small")
        if int(window.get("candidate_days",0)) < 50:
            raise RuntimeError(f"{label} candidate sample too small")

    round_trip_cost_bps = float(release.get("cost_bps", 3.0) + release.get("slippage_bps", 2.0)) * 2.0
    production_gate = {
        "validation_scope": "2025_validation_and_2026_final_holdout",
        "min_forward_3d_mean_return_pct": 0.10,
        "min_forward_5d_mean_return_pct": 0.10,
        "min_forward_5d_positive_rate_pct": 50.0,
        "min_managed_trade_mean_return_pct": 0.10,
        "min_managed_trade_win_rate_pct": 45.0,
        "max_managed_trade_drawdown_pct": -40.0,
        "round_trip_cost_bps": round_trip_cost_bps,
    }

    def window_checks(window):
        return {
            "forward_3d_mean": bool(
                window.get("forward_3d",{}).get("mean_return_pct") is not None
                and float(window["forward_3d"]["mean_return_pct"]) >= production_gate["min_forward_3d_mean_return_pct"]
            ),
            "forward_5d_mean": bool(
                window.get("forward_5d",{}).get("mean_return_pct") is not None
                and float(window["forward_5d"]["mean_return_pct"]) >= production_gate["min_forward_5d_mean_return_pct"]
            ),
            "forward_5d_positive_rate": bool(
                window.get("forward_5d",{}).get("positive_rate_pct") is not None
                and float(window["forward_5d"]["positive_rate_pct"]) >= production_gate["min_forward_5d_positive_rate_pct"]
            ),
            "managed_trade_mean": bool(
                window.get("managed_trade",{}).get("mean_return_pct") is not None
                and float(window["managed_trade"]["mean_return_pct"]) >= production_gate["min_managed_trade_mean_return_pct"]
            ),
            "managed_trade_win_rate": bool(
                window.get("managed_trade",{}).get("win_rate_pct") is not None
                and float(window["managed_trade"]["win_rate_pct"]) >= production_gate["min_managed_trade_win_rate_pct"]
            ),
            "managed_trade_drawdown": bool(
                window.get("managed_trade",{}).get("max_drawdown_pct") is not None
                and float(window["managed_trade"]["max_drawdown_pct"]) >= production_gate["max_managed_trade_drawdown_pct"]
            ),
        }

    production_gate["checks"] = {
        "validation": window_checks(validation),
        "final_holdout": window_checks(final_holdout),
    }
    production_gate["passed"] = bool(
        all(production_gate["checks"]["validation"].values())
        and all(production_gate["checks"]["final_holdout"].values())
        and release.get("release_gate_scope") == "final_holdout_only"
    )
    for path in REQUIRED_READY:
        p=load(path)
        if p.get("strategy_version")!=research_strategy_version or p.get("strategy_commit")!=research_strategy_commit:
            raise RuntimeError(f"research strategy provenance mismatch: {path}")
    candidate_matches_research = bool(
        candidates.get("strategy_version")==research_strategy_version
        and candidates.get("strategy_commit")==research_strategy_commit
    )
    if not candidate_matches_research:
        raise RuntimeError("研究结果与候选快照策略版本或提交不一致")
    plan = load("data/next_trading_day_plan.json")
    plan_check = validate_plan(plan, candidates, production_status)
    payload={
        "schema_version":2,"status":"ready","audited_at":datetime.now(timezone.utc).isoformat(),
        "historical_database":"complete","fundamentals_database":"complete_pit",
        "candidate_layer":"ready","short_term_research":"ready",
        "production_gate":"passed" if production_gate["passed"] else "research_only",
        "strategy_source":"Aquant-Private/main","strategy_version":research_strategy_version,
        "strategy_commit":research_strategy_commit,"future_function":False,
        "production_candidate_strategy_version":candidates.get("strategy_version"),
        "production_candidate_strategy_commit":candidates.get("strategy_commit"),
        "production_horizon":"T_close -> T+1_open -> max_5_sessions",
        "audit":{
            "strategy_metadata_locked":True,"cross_layer_strategy_consistency":candidate_matches_research and plan_check.get("status")=="pass",
            "history_completion_cross_checked":True,"fundamentals_pit_metadata_checked":True,
            "candidate_pool_integrity_checked":True,"short_term_research_coverage_checked":True,
            "execution_assumptions_explicit":True,
            "production_gate_evaluated":True,
            "production_gate_passed":production_gate["passed"]
        }
    }
    production_released = bool(production_gate["passed"])
    production_payload = {
        "schema_version": 1,
        "status": "released" if production_released else "not_released",
        "production_version": research_strategy_version if production_released else None,
        "production_strategy_commit": research_strategy_commit if production_released else None,
        "as_of": candidates.get("as_of") if production_released else None,
        "release_gate": production_released,
        "system_audit": True,
        "note": "正式生产版本已通过发布门槛与系统审计。" if production_released else "当前没有正式生产策略；研究候选未达到正式发布条件。"
    }
    (ROOT/"data/production_status.json").write_text(
        json.dumps(production_payload,ensure_ascii=False,indent=2)+"\\n",encoding="utf-8"
    )
    set_production_release(ROOT/"data/next_trading_day_plan.json", production_released)
    out=ROOT/args.output; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False))
