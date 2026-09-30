from __future__ import annotations
import gzip, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import akshare as ak

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"; HISTORY=DATA/"history"
TZ=ZoneInfo("Asia/Shanghai")
START_DAYS=1095
WORKERS=4
MIN_ROWS_PER_STOCK=20

def fetch_spot():
    for fn in (ak.stock_zh_a_spot, ak.stock_zh_a_spot_em):
        for attempt in range(3):
            try:
                df=fn()
                if df is not None and not df.empty:
                    df=df.copy()
                    df["代码"]=df["代码"].astype(str).str.extract(r"(\d+)")[0].str.zfill(6)
                    df=df[df["代码"].str.len().eq(6)]
                    if "名称" in df.columns:
                        name=df["名称"].astype(str).str.upper()
                        before=len(df)
                        df=df[~name.str.contains(r"ST|退",regex=True,na=False)].copy()
                        print(f"UNIVERSE FILTER: removed {before-len(df)} ST/delisted-related symbols; remaining={len(df)}")
                    return df.drop_duplicates("代码")
            except Exception as exc:
                print(f"WARN universe {getattr(fn,'__name__',fn)} attempt {attempt+1}: {exc}")
                time.sleep(2**attempt)
    raise RuntimeError("No A-share universe source is reachable")

def market_symbol(code):
    code=str(code).zfill(6)
    return ("sh" if code.startswith("6") else "sz")+code

def history_one(symbol,start,end):
    for attempt in range(3):
        try:
            df=ak.stock_zh_a_daily(symbol=market_symbol(symbol),start_date=start,end_date=end,adjust="qfq")
            if df is None or df.empty:
                return None
            aliases={
                "date":["日期","date"],"open":["开盘","open"],"high":["最高","high"],
                "low":["最低","low"],"close":["收盘","close"],"volume":["成交量","volume"],
                "amount":["成交额","amount"],"turnover_pct":["换手率","turnover"]
            }
            def col(key):
                for name in aliases[key]:
                    if name in df.columns: return df[name]
                return pd.Series(index=df.index,dtype="float64")
            dates=col("date")
            if dates.isna().all(): return None
            out=pd.DataFrame({
                "date":pd.to_datetime(dates,errors="coerce").dt.strftime("%Y-%m-%d"),
                "symbol":str(symbol).zfill(6),
                "open":pd.to_numeric(col("open"),errors="coerce"),
                "high":pd.to_numeric(col("high"),errors="coerce"),
                "low":pd.to_numeric(col("low"),errors="coerce"),
                "close":pd.to_numeric(col("close"),errors="coerce"),
                "volume":pd.to_numeric(col("volume"),errors="coerce"),
                "amount":pd.to_numeric(col("amount"),errors="coerce"),
                "turnover_pct":pd.to_numeric(col("turnover_pct"),errors="coerce")
            }).dropna(subset=["date","close"])
            return out if len(out)>=MIN_ROWS_PER_STOCK else None
        except Exception as exc:
            if attempt==2: print(f"WARN {symbol}: {exc}")
            time.sleep(1.5*(attempt+1))
    return None

def write_daily_gzip(all_df):
    HISTORY.mkdir(parents=True,exist_ok=True)
    for day,g in all_df.groupby("date",sort=True):
        path=HISTORY/f"{day}.csv.gz"
        g.sort_values("symbol").to_csv(path,index=False,compression="gzip")
    return len(all_df["date"].unique())

def write_universe(raw):
    cols=[c for c in ["代码","名称","最新价","成交额"] if c in raw.columns]
    raw[cols].sort_values("代码").to_json(DATA/"universe.json",orient="records",force_ascii=False,indent=2)

def initial_backfill(raw):
    end=datetime.now(TZ).date(); start=end-timedelta(days=START_DAYS)
    raw=raw.copy()
    symbols=raw["代码"].dropna().astype(str).str.zfill(6).drop_duplicates().tolist()
    print(f"FULL MARKET UNIVERSE: {len(symbols)} symbols")
    frames=[]; failed=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures={pool.submit(history_one,s,start.strftime("%Y%m%d"),end.strftime("%Y%m%d")):s for s in symbols}
        for i,f in enumerate(as_completed(futures),1):
            code=futures[f]
            try: df=f.result()
            except Exception as exc:
                print(f"WARN future {code}: {exc}"); df=None
            if df is not None: frames.append(df)
            else: failed.append(code)
            if i%100==0: print(f"BACKFILL {i}/{len(symbols)} success={len(frames)} failed={len(failed)}")
    if not frames: raise RuntimeError("Historical backfill returned no data")
    all_df=pd.concat(frames,ignore_index=True)
    days=write_daily_gzip(all_df)
    write_universe(raw)
    marker=HISTORY/"_BACKFILL_COMPLETE"
    marker.write_text(json.dumps({
        "completed_at":datetime.now(TZ).isoformat(),
        "symbols_requested":len(symbols),"symbols_with_history":len(frames),
        "symbols_failed":len(failed),"rows":len(all_df),
        "trading_days":days,"start":str(start),"end":str(end),
        "format":"daily CSV gzip","source":"AKShare / Sina stock_zh_a_daily","universe_filter":"exclude ST/*ST and delisted-related names"
    },ensure_ascii=False,indent=2),encoding="utf-8")
    if failed: print("FAILED SAMPLE:",",".join(failed[:100]))
    print(json.dumps({"mode":"initial-backfill","symbols":len(symbols),"success":len(frames),"failed":len(failed),"rows":len(all_df),"trading_days":days},ensure_ascii=False))

def incremental(raw,today):
    path=HISTORY/f"{today}.csv.gz"
    symbols=raw["代码"].dropna().astype(str).str.zfill(6).drop_duplicates().tolist()
    frames=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures={pool.submit(history_one,s,today.strftime("%Y%m%d"),today.strftime("%Y%m%d")):s for s in symbols}
        for f in as_completed(futures):
            df=f.result()
            if df is not None: frames.append(df)
    if frames:
        df=pd.concat(frames,ignore_index=True)
        df.to_csv(path,index=False,compression="gzip")
        write_universe(raw)
    print(json.dumps({"mode":"incremental","date":str(today),"rows":sum(len(x) for x in frames),"symbols":len(frames)},ensure_ascii=False))

def main():
    raw=fetch_spot()
    marker=HISTORY/"_BACKFILL_COMPLETE"
    if not marker.exists():
        initial_backfill(raw); return
    today=datetime.now(TZ).date()
    try:
        dates=set(pd.to_datetime(ak.tool_trade_date_hist_sina()["trade_date"]).dt.date)
        if today not in dates:
            print(json.dumps({"mode":"incremental","changed":False,"skipped":True,"date":str(today)},ensure_ascii=False)); return
    except Exception as exc:
        print(f"WARN trade calendar: {exc}")
    incremental(raw,today)

if __name__=="__main__":
    main()
