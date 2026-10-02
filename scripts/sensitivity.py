"""Parameter sensitivity analysis for the frozen Private strategy.

Only portfolio construction / execution-cost assumptions vary. Historical data,
PIT rules and the Private strategy commit remain fixed.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import backtest as base

ROOT = base.ROOT
OUT_FILE = ROOT / "data" / "backtest" / "sensitivity.json"

def run_one(top_n: int, cost_bps: float, slippage_bps: float, start: str, end: str) -> dict:
    p = base.run_backtest(start=start, end=end, top_n=top_n,
                          cost_bps=cost_bps, slippage_bps=slippage_bps)
    m = p["overall"]
    return {
        "top_n": top_n, "transaction_cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "total_return_pct": m["total_return_pct"],
        "annualized_return_pct": m["annualized_return_pct"],
        "annualized_volatility_pct": m["annualized_volatility_pct"],
        "sharpe": m["sharpe"], "max_drawdown_pct": m["max_drawdown_pct"],
        "win_rate_pct": m["win_rate_pct"],
        "average_turnover_pct": m["average_turnover_pct"],
        "total_turnover_pct": m["total_turnover_pct"],
        "strategy_version": p["strategy_version"],
        "strategy_commit": p["strategy_commit"],
        "future_function": p["future_function"],
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",default="2015-01-05")
    ap.add_argument("--end",default="2026-09-29")
    ap.add_argument("--top-n",default="10,20,30,50,100")
    ap.add_argument("--cost-bps",default="3,5,10")
    ap.add_argument("--slippage-bps",default="2")
    ap.add_argument("--output",default=str(OUT_FILE))
    a=ap.parse_args()
    top_ns=[int(x) for x in a.top_n.split(",") if x]
    costs=[float(x) for x in a.cost_bps.split(",") if x]
    slips=[float(x) for x in a.slippage_bps.split(",") if x]
    rows=[]
    for c in costs:
        for s in slips:
            for n in top_ns:
                print(f"[sensitivity] top_n={n} cost={c} slip={s}")
                rows.append(run_one(n,c,s,a.start,a.end))
    commits={r["strategy_commit"] for r in rows}
    versions={r["strategy_version"] for r in rows}
    if len(commits)!=1 or len(versions)!=1 or any(r["future_function"] for r in rows):
        raise ValueError("strategy/PIT consistency audit failed")
    out={"schema_version":1,"status":"ready","method":"fixed_strategy_parameter_sensitivity",
         "start":a.start,"end":a.end,"results":rows,
         "audit":{"fixed_strategy":True,"future_adjusted_factor_not_used":True}}
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"rows":len(rows),"strategy_commit":next(iter(commits))}))
if __name__=="__main__": main()
