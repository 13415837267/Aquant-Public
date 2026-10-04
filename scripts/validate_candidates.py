from __future__ import annotations
import argparse
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
    "rank","symbol","name","price","change_pct",
    "return_3d_pct","return_5d_pct","return_10d_pct",
    "volume_ratio_5d","turnover_pct","amount",
    "volatility_10d_pct","close_strength","score","precision_probability","admission_tier",
}
EXPECTED_WEIGHTS = {"momentum_short","overnight_structure","volume_activity","price_strength","liquidity","safety"}
LEGACY_WEIGHTS = {"momentum_short","volume_activity","price_strength","liquidity","safety"}

def finite(value: object, field: str) -> float:
    try: number = float(value)
    except (TypeError, ValueError) as exc: raise RuntimeError(f"{field} is not numeric") from exc
    if not math.isfinite(number): raise RuntimeError(f"{field} is not finite")
    return number

def date_only(value: object, field: str) -> str:
    try: return date.fromisoformat(str(value)[:10]).isoformat()
    except ValueError as exc: raise RuntimeError(f"{field} has invalid date") from exc

def private_provenance():
    root_text = os.environ.get("AQUANT_PRIVATE_STRATEGY_PATH")
    if not root_text: return None, None
    root = Path(root_text).resolve()
    vf = root / "strategy" / "version.py"
    if not vf.exists(): raise RuntimeError("Private strategy version file missing")
    version = None
    for line in vf.read_text(encoding="utf-8").splitlines():
        if line.startswith("STRATEGY_VERSION") and "=" in line:
            version = line.split("=",1)[1].strip().strip('"').strip("'")
    commit = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"], text=True).strip()
    return version, commit

def validate_candidates(payload, private_version=None, private_commit=None):
    if not isinstance(payload, dict): raise RuntimeError("candidate snapshot must be an object")
    if payload.get("status") != "ready" or payload.get("future_function") is not False:
        raise RuntimeError("candidate readiness/PIT audit failed")
    as_of = str(payload.get("as_of") or "")
    if not as_of.endswith("T18:00:00+08:00"): raise RuntimeError("invalid production timestamp")
    if date_only(as_of,"as_of") != date_only(payload.get("history_window_end"),"history_window_end"):
        raise RuntimeError("as_of must match history_window_end")
    if date_only(payload.get("history_window_start"),"history_window_start") > date_only(payload.get("history_window_end"),"history_window_end"):
        raise RuntimeError("invalid history window")
    if not isinstance(payload.get("history_files_used"), int) or payload["history_files_used"] < 20:
        raise RuntimeError("history_files_used must be >= 20")
    if payload.get("lookback_trading_days") != 20: raise RuntimeError("short-term lookback must be 20")
    if payload.get("signal_horizon") != "T收盘信号 → T+1开盘买入 → T+2起最早卖出 → 最长5个交易日":
        raise RuntimeError("short-term signal horizon mismatch")
    if payload.get("strategy_source") != "Aquant-Private/main": raise RuntimeError("strategy source mismatch")
    version = str(payload.get("strategy_version") or "")
    commit = str(payload.get("strategy_commit") or "")
    if not version or not commit: raise RuntimeError("strategy provenance missing")
    if private_version is not None and version != private_version: raise RuntimeError("strategy version mismatch")
    if private_commit is not None and commit != private_commit: raise RuntimeError("strategy commit mismatch")
    weights = payload.get("factor_weights")
    if not isinstance(weights, dict) or set(weights) not in (EXPECTED_WEIGHTS, LEGACY_WEIGHTS): raise RuntimeError("factor weights are invalid")
    weight_keys = set(weights)
    if abs(sum(finite(weights[k], f"factor_weights.{k}") for k in weight_keys)-1.0) > 1e-9:
        raise RuntimeError("factor weights must sum to 1")
    candidates = payload.get("candidates")
    policy = str(payload.get("candidate_admission_policy") or "")
    if policy != "precision_top_2_with_089_gate_and_088_daily_fallback":
        raise RuntimeError("candidate admission policy is invalid")
    max_candidates = 2
    if not isinstance(candidates, list) or len(candidates) > max_candidates: raise RuntimeError("invalid candidate count")
    previous = math.inf
    seen = set()
    for rank,row in enumerate(candidates,1):
        missing = REQUIRED_CANDIDATE_FIELDS - set(row)
        if missing: raise RuntimeError(f"candidate #{rank} missing {sorted(missing)}")
        if row.get("rank") != rank: raise RuntimeError("candidate ranks are invalid")
        symbol = str(row.get("symbol","")).zfill(6)
        if not is_main_board_symbol(symbol): raise RuntimeError(f"{symbol} outside production scope")
        if symbol in seen: raise RuntimeError("duplicate candidate symbol")
        seen.add(symbol)
        score = finite(row["score"], f"{symbol}.score")
        probability = finite(row["precision_probability"], f"{symbol}.precision_probability")
        if not 0 <= score <= 100: raise RuntimeError(f"{symbol} score outside 0..100")
        if not 0 <= probability <= 1: raise RuntimeError(f"{symbol} precision_probability outside 0..1")
        tier = str(row.get("admission_tier") or "")
        if tier not in {"primary_089","coverage_fallback_088"}: raise RuntimeError(f"{symbol} admission tier invalid")
        if tier == "primary_089" and probability < 0.89 - 1e-9: raise RuntimeError(f"{symbol} primary probability gate failed")
        if tier == "coverage_fallback_088" and probability < 0.88 - 1e-9: raise RuntimeError(f"{symbol} fallback probability gate failed")
        if score > previous + 1e-9: raise RuntimeError("candidates not sorted by score")
        previous = score
        for field in REQUIRED_CANDIDATE_FIELDS - {"rank","symbol","name"}: finite(row[field], f"{symbol}.{field}")
        if not str(row.get("name") or "").strip(): raise RuntimeError(f"{symbol} has empty name")
    market = payload.get("market")
    if not isinstance(market, dict) or market.get("regime") not in {"risk_on","neutral","risk_off"}:
        raise RuntimeError("invalid market regime")
    diag = payload.get("diagnostics")
    if not isinstance(diag, dict) or diag.get("candidate_count") != len(candidates):
        raise RuntimeError("candidate diagnostics mismatch")
    fallback_count = sum(1 for row in candidates if str(row.get("admission_tier")) == "coverage_fallback_088")
    if fallback_count > 1: raise RuntimeError("coverage fallback must select at most one candidate")
    if diag.get("coverage_fallback_used") != (fallback_count == 1): raise RuntimeError("coverage fallback audit mismatch")
    if diag.get("primary_candidate_count") != sum(1 for row in candidates if str(row.get("admission_tier")) == "primary_089"):
        raise RuntimeError("primary candidate diagnostics mismatch")
    if diag.get("risk_off_no_trade") != (market["regime"] == "risk_off"):
        raise RuntimeError("risk-off admission audit mismatch")
    for key in ("hard_eligibility_applied_before_scoring","strategy_source_locked_to_private","short_term_features_only","market_gate_applied"):
        if payload.get("audit",{}).get(key) is not True: raise RuntimeError(f"audit failed: {key}")
    return {"status":"pass","as_of":as_of,"candidate_count":len(candidates),"strategy_version":version,"strategy_commit":commit,"market_regime":market["regime"]}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=DEFAULT_CANDIDATES)
    args=parser.parse_args()
    payload=json.loads(args.input.read_text(encoding="utf-8"))
    pv,pc=private_provenance()
    print(json.dumps(validate_candidates(payload,pv,pc),ensure_ascii=False))

if __name__=="__main__": main()
