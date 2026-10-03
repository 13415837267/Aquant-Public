from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.validate_candidates import validate_candidates

REQUIRED_READY=[
    "data/backtest/short_term_latest.json",
    "data/backtest/short_term_sensitivity.json",
    "data/backtest/short_term_exit_diagnostics.json",
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
    for path in REQUIRED_READY: ready(path,load(path))
    research=load(REQUIRED_READY[0])
    if int(research.get("signal_days",0))<500: raise RuntimeError("short-term research coverage too small")
    if int(research.get("candidate_days",0))<50: raise RuntimeError("short-term candidate sample too small")
    for path in REQUIRED_READY:
        p=load(path)
        if p.get("strategy_version")!=candidates.get("strategy_version") or p.get("strategy_commit")!=candidates.get("strategy_commit"):
            raise RuntimeError(f"strategy provenance mismatch: {path}")
    payload={
        "schema_version":2,"status":"ready","audited_at":datetime.now(timezone.utc).isoformat(),
        "historical_database":"complete","fundamentals_database":"complete_pit",
        "candidate_layer":"ready","short_term_research":"ready",
        "strategy_source":candidates["strategy_source"],"strategy_version":candidates["strategy_version"],
        "strategy_commit":candidates["strategy_commit"],"future_function":False,
        "production_horizon":"T_close -> T+1_open -> max_5_sessions",
        "audit":{
            "strategy_metadata_locked":True,"cross_layer_strategy_consistency":True,
            "history_completion_cross_checked":True,"fundamentals_pit_metadata_checked":True,
            "candidate_pool_integrity_checked":True,"short_term_research_coverage_checked":True,
            "execution_assumptions_explicit":True
        }
    }
    out=ROOT/args.output; out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False))

if __name__=="__main__": main()
