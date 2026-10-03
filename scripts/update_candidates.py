from __future__ import annotations

import gzip, importlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.market_scope import is_main_board_symbol

DATA_DIR=ROOT/"data"
HISTORY_DIR=DATA_DIR/"history"
DATA_FILE=DATA_DIR/"candidates.json"
PRIVATE_STRATEGY_PATH=os.environ.get("AQUANT_PRIVATE_STRATEGY_PATH")
PRIVATE_STRATEGY_COMMIT=os.environ.get("AQUANT_PRIVATE_STRATEGY_COMMIT")

REQUIRED_COLUMNS={"symbol","date","high","low","close","volume","amount","pct_chg","turnover_pct","is_paused","is_st"}

def _load_current_names():
    path=DATA_DIR/"universe.json"
    if not path.exists(): return {}
    try: rows=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return {}
    return {str(r.get("symbol","")).zfill(6):str(r.get("name","")).strip()
            for r in rows if is_main_board_symbol(r.get("ts_code")) and r.get("symbol") and r.get("name")}

def _load_private_strategy():
    if not PRIVATE_STRATEGY_PATH: raise RuntimeError("AQUANT_PRIVATE_STRATEGY_PATH is required")
    root=Path(PRIVATE_STRATEGY_PATH).resolve()
    if not (root/"strategy"/"model.py").exists(): raise RuntimeError("Private strategy not found")
    sys.path.insert(0,str(root))
    model=importlib.import_module("strategy.model")
    version=importlib.import_module("strategy.version")
    commit=PRIVATE_STRATEGY_COMMIT or subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    return model,str(version.STRATEGY_VERSION),commit

def _history_files():
    files=sorted(HISTORY_DIR.glob("????/*.csv.gz"),key=lambda p:p.name[:10],reverse=True)
    if not files: files=sorted(HISTORY_DIR.glob("????-??-??.csv.gz"),key=lambda p:p.name[:10],reverse=True)
    if not files: raise RuntimeError("No historical daily files found")
    return files

def _read(path:Path):
    with gzip.open(path,"rt",encoding="utf-8") as fh: frame=pd.read_csv(fh)
    missing=REQUIRED_COLUMNS-set(frame.columns)
    if missing: raise RuntimeError(f"{path.name} missing required columns: {sorted(missing)}")
    frame["symbol"]=frame["symbol"].astype(str).str.extract(r"(\d{6})")[0].str.zfill(6)
    frame=frame.loc[frame["symbol"].map(is_main_board_symbol)].copy()
    frame["date"]=pd.to_datetime(frame["date"],errors="coerce").dt.strftime("%Y-%m-%d")
    file_date=path.name[:10]
    if frame["date"].ne(file_date).any(): raise RuntimeError(f"{path.name} date mismatch")
    for c in ["high","low","close","volume","amount","pct_chg","turnover_pct","is_paused","is_st"]:
        frame[c]=pd.to_numeric(frame[c],errors="coerce")
    frame["is_paused"]=frame["is_paused"].fillna(0); frame["is_st"]=frame["is_st"].fillna(0)
    return frame.drop_duplicates(["symbol","date"],keep="last")

def _prepare(history:pd.DataFrame,latest_date:str):
    hist=history.sort_values(["symbol","date"]).copy()
    hist["daily_ret"]=hist["pct_chg"]/100.0
    for w in (1,3,5,10,20):
        hist[f"return_{w}d_pct"]=hist.groupby("symbol")["daily_ret"].transform(
            lambda s:(1.0+s).rolling(w,min_periods=w).apply(np.prod,raw=True).sub(1.0).mul(100.0)
        )
    hist["volatility_10d_pct"]=hist.groupby("symbol")["daily_ret"].transform(
        lambda s:s.rolling(10,min_periods=10).std().mul(100.0)
    )
    prior5=hist.groupby("symbol")["volume"].transform(
        lambda s:s.shift(1).rolling(5,min_periods=5).mean()
    )
    hist["volume_ratio_5d"]=hist["volume"]/prior5.replace(0,np.nan)
    hist["amount_20d"]=hist.groupby("symbol")["amount"].transform(
        lambda s:s.rolling(20,min_periods=20).mean()
    )
    rng=(hist["high"]-hist["low"]).replace(0,np.nan)
    hist["close_strength"]=((hist["close"]-hist["low"])/rng).clip(0,1)

    latest=hist.loc[hist["date"].eq(latest_date)].copy()
    names=_load_current_names()
    latest["name"]=latest["symbol"].map(names).fillna(latest["symbol"])
    excluded=latest["name"].str.contains(r"ST|退",case=False,na=False)
    eligible=(~excluded & latest["is_st"].eq(0) & latest["is_paused"].eq(0) &
              latest["close"].gt(2) & latest["amount"].ge(3e7))
    market=latest.loc[eligible]
    breadth=float((market["pct_chg"]>0).mean()*100) if not market.empty else 0.0
    median_ret=float(market["pct_chg"].median()) if not market.empty else 0.0
    usable=eligible & latest["return_10d_pct"].notna() & latest["volume_ratio_5d"].notna() & latest["volatility_10d_pct"].notna() & latest["amount_20d"].notna()
    latest=latest.loc[usable].copy()
    latest["market_breadth_pct"]=breadth; latest["market_median_return_pct"]=median_ret
    return latest.reset_index(drop=True),{
        "latest_main_board_rows":int(len(latest)),
        "latest_basic_eligible_rows":int(eligible.sum()),
        "latest_basic_filter_exclusions":int((~eligible).sum()),
        "scorable_rows":int(len(latest)),
    }

def build_candidates(history,strategy_model,strategy_version,strategy_commit):
    latest_date=str(history["date"].dropna().max())
    frame,diag=_prepare(history,latest_date)
    frame=frame.rename(columns={"pct_chg":"change_pct"})
    if frame.empty: raise RuntimeError("No usable rows after short-term eligibility/features")
    scored=strategy_model.score_universe(frame)
    admission=getattr(strategy_model,"admit_candidates",None)
    if not callable(admission): raise RuntimeError("Private strategy must expose admit_candidates")
    selected=admission(scored)
    rows=[]
    for rank,row in enumerate(selected.itertuples(index=False),1):
        flags=[]
        if float(row.volatility_10d_pct)>8: flags.append("高波动")
        if float(row.change_pct)<-7: flags.append("当日跌幅<-7%")
        if float(row.return_5d_pct)>15: flags.append("5日过热")
        rows.append({
            "rank":rank,"symbol":str(row.symbol).zfill(6),"name":row.name,"price":round(float(row.close),3),
            "change_pct":round(float(row.change_pct),3),"return_3d_pct":round(float(row.return_3d_pct),3),
            "return_5d_pct":round(float(row.return_5d_pct),3),"return_10d_pct":round(float(row.return_10d_pct),3),
            "volume_ratio_5d":round(float(row.volume_ratio_5d),3),"turnover_pct":round(float(row.turnover_pct),3),
            "amount":round(float(row.amount),2),"volatility_10d_pct":round(float(row.volatility_10d_pct),3),
            "close_strength":round(float(row.close_strength),3),"score":round(float(row.score),3),"flags":flags
        })
    breadth=float(frame["market_breadth_pct"].iloc[0]); median=float(frame["market_median_return_pct"].iloc[0])
    regime="risk_on" if breadth>=55 and median>0.3 else ("neutral" if breadth>=35 and median>=-0.3 else "risk_off")
    return {
        "as_of":f"{latest_date}T18:00:00+08:00","timezone":"Asia/Shanghai","source":"Aquant-Public data/history",
        "status":"ready","strategy_source":"Aquant-Private/main","strategy_version":strategy_version,"strategy_commit":strategy_commit,
        "market_scope":"沪深主板：000001-004999.SZ（排除001001-001199 CDR）+ 600/601/603/605.SH",
        "universe":"沪深主板；排除 ST/退市相关标的、停牌、价格≤2元、最近交易日成交额<3000万元",
        "lookback_trading_days":20,"signal_horizon":"T收盘信号 → T+1开盘进入 → 最长5个交易日",
        "holding_window_sessions":[1,5],"risk_controls":{"target_return_pct":6.0,"stop_loss_pct":3.0,"max_positions":3},
        "market":{"breadth_pct":round(breadth,3),"median_return_pct":round(median,3),"regime":regime},
        "diagnostics":{"history_rows":int(len(history)),**diag,"candidate_count":len(rows),"risk_off_no_trade":regime=="risk_off"},
        "candidates":rows,"factor_weights":getattr(strategy_model,"WEIGHTS",None),"future_function":False,
        "candidate_admission_policy":"dynamic_top_score_3_with_market_gate",
        "audit":{"hard_eligibility_applied_before_scoring":True,"strategy_source_locked_to_private":True,"short_term_features_only":True,"market_gate_applied":True}
    }

def main():
    model,version,commit=_load_private_strategy()
    files=_history_files()[:21]
    frames=[_read(p) for p in files]
    history=pd.concat(frames,ignore_index=True)
    snapshot=build_candidates(history,model,version,commit)
    dates=[p.name[:10] for p in files]
    snapshot["history_files_used"]=len(dates)
    snapshot["history_window_start"]=min(dates); snapshot["history_window_end"]=max(dates)
    DATA_FILE.parent.mkdir(parents=True,exist_ok=True)
    DATA_FILE.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"ready","rows":len(snapshot["candidates"]),"strategy_version":version,"strategy_commit":commit,"as_of":snapshot["as_of"]},ensure_ascii=False))

if __name__=="__main__": main()
