from __future__ import annotations
import json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
import akshare as ak

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; HISTORY=DATA/'history'; SNAP=DATA/'candidates.json'
TZ=ZoneInfo('Asia/Shanghai'); START_DAYS=1095; MAX_SYMBOLS=500; WORKERS=8

def fetch_spot():
    return ak.stock_zh_a_spot_em()

def history_one(symbol,start,end):
    try:
        df=ak.stock_zh_a_hist(symbol=symbol,period='daily',start_date=start,end_date=end,adjust='qfq')
        if df.empty: return None
        code=str(symbol).zfill(6); out=pd.DataFrame({
          'date':pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d'),
          'symbol':code,
          'open':pd.to_numeric(df['开盘'],errors='coerce'),'high':pd.to_numeric(df['最高'],errors='coerce'),
          'low':pd.to_numeric(df['最低'],errors='coerce'),'close':pd.to_numeric(df['收盘'],errors='coerce'),
          'volume':pd.to_numeric(df['成交量'],errors='coerce'),'amount':pd.to_numeric(df['成交额'],errors='coerce'),
          'turnover_pct':pd.to_numeric(df.get('换手率'),errors='coerce') if '换手率' in df else None})
        return out.dropna(subset=['close'])
    except Exception as exc:
        print(f'WARN {symbol}: {exc}')
        return None

def write_history(raw):
    end=datetime.now(TZ).date(); start=end-timedelta(days=START_DAYS)
    raw=raw.copy(); raw['代码']=raw['代码'].astype(str).str.extract(r'(\d+)')[0].str.zfill(6)
    raw['成交额']=pd.to_numeric(raw.get('成交额'),errors='coerce')
    raw=raw[raw['成交额'].fillna(0)>=2e7]
    raw=raw[~raw['名称'].astype(str).str.contains(r'ST|退',na=False)].sort_values('成交额',ascending=False).head(MAX_SYMBOLS)
    symbols=raw['代码'].tolist(); frames=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures={pool.submit(history_one,s,start.strftime('%Y%m%d'),end.strftime('%Y%m%d')):s for s in symbols}
        for i,f in enumerate(as_completed(futures),1):
            df=f.result()
            if df is not None: frames.append(df)
            if i%50==0: print(f'BACKFILL {i}/{len(symbols)}')
    if not frames: raise RuntimeError('Historical backfill returned no data')
    all_df=pd.concat(frames,ignore_index=True); HISTORY.mkdir(parents=True,exist_ok=True)
    for day,g in all_df.groupby('date'):
        p=HISTORY/f'{day}.json'; g.to_json(p,orient='records',force_ascii=False)
    return len(symbols),len(all_df),start,end

def main():
    raw=fetch_spot(); marker=HISTORY/'_BACKFILL_COMPLETE'
    if not marker.exists():
        n,rows,start,end=write_history(raw)
        marker.write_text(json.dumps({'completed_at':datetime.now(TZ).isoformat(),'symbols':n,'rows':rows,'start':str(start),'end':str(end)},ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'mode':'initial-backfill','symbols':n,'rows':rows,'start':str(start),'end':str(end)},ensure_ascii=False)); return
    today=datetime.now(TZ).date(); dates=set(pd.to_datetime(ak.tool_trade_date_hist_sina()['trade_date']).dt.date)
    if today not in dates:
        print(json.dumps({'mode':'incremental','changed':False,'skipped':True,'date':str(today)},ensure_ascii=False)); return
    # Incremental: fetch the same liquid universe and append only today's rows.
    raw=fetch_spot(); rows=[]
    for _,r in raw.iterrows():
        code=str(r.get('代码','')).zfill(6); name=str(r.get('名称',''))
        if not code or ('ST' in name or '退' in name): continue
        rows.append({'date':str(today),'symbol':code,'close':r.get('最新价'),'amount':r.get('成交额'),'turnover_pct':r.get('换手率')})
    pd.DataFrame(rows).to_json(HISTORY/f'{today}.json',orient='records',force_ascii=False)
    print(json.dumps({'mode':'incremental','changed':True,'rows':len(rows),'date':str(today)},ensure_ascii=False))

if __name__=='__main__': main()
