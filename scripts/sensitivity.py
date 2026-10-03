"""Parameter sensitivity analysis for the current Private strategy.

The workflow publishes a synchronized, warmup-excluded result snapshot.

Only portfolio-size and execution-cost assumptions vary. Historical data, PIT rules,
and the Private strategy commit remain fixed.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import backtest as base

ROOT = base.ROOT
OUT_FILE = ROOT / "data" / "backtest" / "sensitivity.json"

def run_one(top_n: int, cost_bps: float, slippage_bps: float, payload: dict) -> dict:
    daily = pd.DataFrame(payload["daily"])
    if daily.empty:
        raise ValueError("sensitivity payload has no daily rows")
    daily = daily.copy()
    daily["net_return"] = (
        pd.to_numeric(daily["gross_return"], errors="coerce").fillna(0.0)
        - pd.to_numeric(daily["turnover"], errors="coerce").fillna(0.0)
        * (cost_bps + slippage_bps) / 10000.0
    )
    trade_start = payload.get("trade_start")
    performance = (
        daily[daily["date"].astype(str) >= str(trade_start)].reset_index(drop=True)
        if trade_start
        else daily
    )
    if performance.empty:
        raise ValueError("sensitivity payload has no performance rows")
    m = base.metrics(performance)
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
        "strategy_version": payload["strategy_version"],
        "strategy_commit": payload["strategy_commit"],
        "future_function": payload["future_function"],
        "daily_sessions": int(len(daily)),
        "performance_sessions": int(len(performance)),
        "warmup_sessions": int(len(daily) - len(performance)),
    }
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",default="2015-01-05")
    ap.add_argument("--end",default="2026-09-29")
    ap.add_argument("--top-n",default="1,2,3")
    ap.add_argument("--cost-bps",default="3,5,10")
    ap.add_argument("--slippage-bps",default="2")
    ap.add_argument("--output",default=str(OUT_FILE))
    a=ap.parse_args()
    top_ns=[int(x) for x in a.top_n.split(",") if x]
    costs=[float(x) for x in a.cost_bps.split(",") if x]
    slips=[float(x) for x in a.slippage_bps.split(",") if x]
    rows=[]
    for n in top_ns:
        print(f"[sensitivity] replay top_n={n} once")
        payload = base.run_backtest(start=a.start, end=a.end, top_n=n,
                                    cost_bps=0.0, slippage_bps=0.0)
        for c in costs:
            for sl in slips:
                print(f"[sensitivity] derive top_n={n} cost={c} slip={sl}")
                rows.append(run_one(n, c, sl, payload))
    commits={r["strategy_commit"] for r in rows}
    versions={r["strategy_version"] for r in rows}
    expected_rows = len(top_ns) * len(costs) * len(slips)
    if (
        len(rows) != expected_rows
        or len(commits) != 1
        or len(versions) != 1
        or any(r["future_function"] for r in rows)
        or any(
            r["performance_sessions"] + r["warmup_sessions"] != r["daily_sessions"]
            for r in rows
        )
    ):
        raise ValueError("strategy/PIT consistency audit failed")
    out={"schema_version":1,"status":"ready","method":"fixed_strategy_parameter_sensitivity",
         "start":a.start,"end":a.end,"results":rows,
         "audit":{
             "fixed_strategy":True,
             "future_adjusted_factor_not_used":True,
             "warmup_sessions": int(rows[0]["warmup_sessions"]) if rows else 0,
             "performance_sessions": int(rows[0]["performance_sessions"]) if rows else 0,
             "metrics_exclude_warmup": True,
             "parameter_grid_size": int(expected_rows),
         }}
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"rows":len(rows),"strategy_commit":next(iter(commits))}))
if __name__=="__main__": main()
