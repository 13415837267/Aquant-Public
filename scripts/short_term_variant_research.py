"""Compare several short-term signal structures on the same historical universe.

This is research-only. It deliberately does not change the production strategy.
"""
from __future__ import annotations

import argparse, gzip, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from scripts.short_term_research import FeatureState, history_files, read_daily
from scripts.market_scope import is_main_board_symbol


VARIANTS = {
    "baseline_momentum": {
        "description":"current positive short momentum + volume + strength",
        "formula":"0.35*momentum+0.25*volume+0.15*strength+0.15*liquidity+0.10*safety",
    },
    "short_reversal": {
        "description":"short-term reversal + volume confirmation",
        "formula":"0.40*reversal+0.25*volume+0.15*strength_reversal+0.10*liquidity+0.10*safety",
    },
    "trend_pullback": {
        "description":"positive 10d trend with 1-3d pullback and volume",
        "formula":"0.30*trend10+0.25*reversal_short+0.20*volume+0.15*strength+0.10*safety",
    },
    "breakout_strength": {
        "description":"short momentum + volume expansion + close strength",
        "formula":"0.35*momentum+0.30*volume+0.20*strength+0.10*liquidity+0.05*safety",
    },
    "hybrid_pullback": {
        "description":"10d momentum, 3d reversal, strong volume, low risk",
        "formula":"0.25*trend10+0.25*reversal3+0.25*volume+0.15*strength+0.10*safety",
    },
}


def rank(s: pd.Series, ascending=True):
    return s.replace([np.inf,-np.inf],np.nan).rank(pct=True,ascending=ascending,method="average").fillna(0.5)


def variant_scores(df: pd.DataFrame) -> dict[str,pd.Series]:
    r1=pd.to_numeric(df["return_1d_pct"],errors="coerce")
    r3=pd.to_numeric(df["return_3d_pct"],errors="coerce")
    r5=pd.to_numeric(df["return_5d_pct"],errors="coerce")
    r10=pd.to_numeric(df["return_10d_pct"],errors="coerce")
    vol=pd.to_numeric(df["volume_ratio_5d"],errors="coerce")
    strength=pd.to_numeric(df["close_strength"],errors="coerce")
    liq=rank(np.log1p(pd.to_numeric(df["amount_20d"],errors="coerce").clip(lower=0)))
    safety=rank(pd.to_numeric(df["volatility_10d_pct"],errors="coerce"),ascending=False)
    momentum=0.10*rank(r1)+0.20*rank(r3)+0.40*rank(r5)+0.30*rank(r10)
    volume=0.65*rank(vol)+0.35*rank(df["turnover_pct"])
    strength_rank=0.65*rank(strength)+0.35*rank(df["change_pct"])
    reversal=0.40*rank(-r1)+0.35*rank(-r3)+0.25*rank(-r5)
    strength_reversal=0.70*rank(-strength)+0.30*rank(-df["change_pct"])
    trend10=rank(r10)
    reversal3=rank(-r3)
    out={}
    out["baseline_momentum"]=100*(0.35*momentum+0.25*volume+0.15*strength_rank+0.15*liq+0.10*safety)
    out["short_reversal"]=100*(0.40*reversal+0.25*volume+0.15*strength_reversal+0.10*liq+0.10*safety)
    out["trend_pullback"]=100*(0.30*trend10+0.25*(0.6*rank(-r1)+0.4*rank(-r3))+0.20*volume+0.15*strength_rank+0.10*safety)
    out["breakout_strength"]=100*(0.35*momentum+0.30*volume+0.20*strength_rank+0.10*liq+0.05*safety)
    out["hybrid_pullback"]=100*(0.25*trend10+0.25*reversal3+0.25*volume+0.15*strength_rank+0.10*safety)
    return out


def forward_return(day_frames, symbol, entry_index, horizon):
    if entry_index>=len(day_frames): return None
    row=day_frames[entry_index].loc[day_frames[entry_index]["symbol"].eq(symbol)]
    if row.empty: return None
    entry=float(row.iloc[0]["open"]) if pd.notna(row.iloc[0].get("open")) else np.nan
    target_index=entry_index+horizon-1
    if target_index>=len(day_frames): return None
    target=day_frames[target_index].loc[day_frames[target_index]["symbol"].eq(symbol)]
    if target.empty or pd.isna(target.iloc[0].get("close")) or not np.isfinite(entry) or entry<=0: return None
    return (float(target.iloc[0]["close"])/entry-1.0)*100.0


def run(args):
    files=history_files()
    dates=[p.name[:10] for p in files]
    if args.start not in dates or args.end not in dates: raise ValueError("research dates must be trading dates")
    start_i,end_i=dates.index(args.start),dates.index(args.end)
    max_future=5
    if end_i+max_future>=len(files): raise ValueError("end date needs five future trading sessions")
    begin=max(0,start_i-20)
    active=files[begin:end_i+max_future+1]
    cache={}
    def get(i):
        if i not in cache: cache[i]=read_daily(active[i])
        return cache[i]
    state=FeatureState()
    results={name:{"signal_days":0,"candidate_days":0,"forward_1d":[],"forward_3d":[],"forward_5d":[]} for name in VARIANTS}
    for i in range(len(active)-max_future):
        signal_date=active[i].name[:10]
        frame=state.build(get(i))
        if signal_date<args.start or signal_date>args.end or frame.empty: continue
        scored_map=variant_scores(frame)
        for name, scores in scored_map.items():
            results[name]["signal_days"]+=1
            tmp=frame.copy(); tmp["_score"]=scores
            breadth=float(tmp["market_breadth_pct"].iloc[0]); median=float(tmp["market_median_return_pct"].iloc[0])
            if breadth<35 or median<-0.30:
                continue
            top=tmp.sort_values(["_score","volume_ratio_5d","amount","symbol"],ascending=[False,False,False,True]).head(3)
            if top.empty: continue
            results[name]["candidate_days"]+=1
            future=[get(i+j) for j in range(1,max_future+1)]
            for symbol in top["symbol"].astype(str).str.zfill(6):
                for horizon,key in ((1,"forward_1d"),(3,"forward_3d"),(5,"forward_5d")):
                    value=forward_return(future,symbol,0,horizon)
                    if value is not None:
                        results[name][key].append(value)

    summary=[]
    for name,data in results.items():
        row={"variant":name,"description":VARIANTS[name]["description"],"formula":VARIANTS[name]["formula"],
             "signal_days":data["signal_days"],"candidate_days":data["candidate_days"],
             "candidate_day_rate_pct":data["candidate_days"]/data["signal_days"]*100 if data["signal_days"] else 0}
        for horizon,key in ((1,"forward_1d"),(3,"forward_3d"),(5,"forward_5d")):
            arr=np.asarray(data[key],dtype=float)
            row[f"{horizon}d_samples"]=int(len(arr))
            row[f"{horizon}d_mean_return_pct"]=float(arr.mean()) if len(arr) else None
            row[f"{horizon}d_median_return_pct"]=float(np.median(arr)) if len(arr) else None
            row[f"{horizon}d_positive_rate_pct"]=float((arr>0).mean()*100) if len(arr) else None
            row[f"{horizon}d_p25_pct"]=float(np.quantile(arr,0.25)) if len(arr) else None
            row[f"{horizon}d_p75_pct"]=float(np.quantile(arr,0.75)) if len(arr) else None
        summary.append(row)

    out={"schema_version":1,"status":"ready","method":"short_term_signal_variant_comparison",
         "start":args.start,"end":args.end,"variants":summary,
         "notes":{"research_only":True,"entry":"T+1_open","future_function":False,
                  "market_gate":"breadth>=35% and median daily return>=-0.30%",
                  "top_n":3}}
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"ready","variants":len(summary),"output":str(path)},ensure_ascii=False))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",required=True); ap.add_argument("--end",required=True)
    ap.add_argument("--output",default=str(ROOT/"data/backtest/short_term_variants.json"))
    run(ap.parse_args())
