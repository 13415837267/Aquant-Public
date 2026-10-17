"""全市场 T 日全因子与 T+1/T+2 开盘三分类研究。"""
from __future__ import annotations
import argparse, json, math, re, shutil, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.tree import DecisionTreeClassifier, _tree

ROOT=Path(__file__).resolve().parents[1]
HISTORY=ROOT/"data/history"; FUNDAMENTALS=ROOT/"data/fundamentals"
CONFIG=ROOT/"config/tplus2_research.json"
TECH=("return_1d_pct","return_3d_pct","return_5d_pct","return_10d_pct","return_20d_pct",
"overnight_1d_pct","overnight_3d_pct","overnight_5d_pct","overnight_10d_pct",
"volume_ratio_5d","amount_20d","volatility_10d_pct","close_strength","intraday_return_pct",
"limit_up_5d_count","turnover_pct","change_pct","close_vs_ma5_pct","close_vs_ma10_pct",
"close_vs_ma20_pct","close_vs_ma60_pct","ma5_vs_ma10_pct","ma5_vs_ma20_pct","ma10_vs_ma20_pct",
"ma20_vs_ma60_pct","ma5_slope_5d_pct","ma20_slope_5d_pct","return_60d_pct","breakout_20d_pct",
"breakout_60d_pct","distance_from_high_20d_pct","distance_from_low_20d_pct","volume_ratio_10d",
"volume_ratio_20d","amount_ratio_5d","amount_ratio_20d","body_pct","upper_shadow_pct",
"lower_shadow_pct","range_pct","volatility_5d_pct","volatility_20d_pct",
"downside_volatility_10d_pct","drawdown_20d_pct","bollinger_position_20d",
"bollinger_width_20d_pct","atr_14_pct","rsi_6","rsi_14","macd_line_pct","macd_signal_pct",
"macd_histogram_pct","kdj_k_9","kdj_d_9","kdj_j_9","limit_up_10d_count")
MARKET=("market_breadth_pct","market_median_return_pct","market_return_dispersion_pct",
"market_above_ma20_pct","market_above_ma60_pct","market_positive_5d_pct")
TABLES=("indicator","income","balance","cash_flow")
CLASSES=("盈利","亏损","平盘"); CLASS_CODE={"亏损":0,"盈利":1,"平盘":2}; CODE_CLASS={v:k for k,v in CLASS_CODE.items()}
IDENTITY={"symbol","date","name","exchange","list_status","ts_code","trade_date","report_date","pub_date"}


def load_config():
    c=json.loads(CONFIG.read_text(encoding="utf-8"))
    if c.get("版本")!=1: raise ValueError("研究配置版本不正确")
    if c["往返交易成本基点"]<0 or c["平盘容忍百分比"]<0: raise ValueError("交易成本或平盘区间非法")
    if not 0<c["训练期结束比例"]<c["验证期结束比例"]<1: raise ValueError("时间切分比例非法")
    return c


def history_files():
    files=sorted(set(HISTORY.glob("????/*.csv.gz"))|set(HISTORY.glob("????-??-??.csv.gz")),key=lambda p:p.name[:10])
    if len(files)<30: raise RuntimeError(f"历史行情文件不足：{len(files)}")
    dates=[p.name[:10] for p in files]
    if len(dates)!=len(set(dates)): raise RuntimeError("历史交易日文件重复")
    return files


def normalize_code(v):
    m=re.search(r"(\d{6})",str(v)); return m.group(1) if m else ""


def discover_raw_columns(files):
    candidates=set();year_files={}
    for p in files:
        candidates.update(c for c in pd.read_csv(p,compression="gzip",nrows=0).columns if c not in IDENTITY)
        year_files.setdefault(p.name[:4],p)
    observed={};numeric={}
    for p in year_files.values():
        df=pd.read_csv(p,compression="gzip",nrows=200)
        for c in candidates.intersection(df.columns):
            valid=df[c].notna();n=int(valid.sum())
            if n:
                observed[c]=observed.get(c,0)+n
                numeric[c]=numeric.get(c,0)+int(pd.to_numeric(df.loc[valid,c],errors="coerce").notna().sum())
    out=sorted(c for c,n in observed.items() if n>=10 and numeric.get(c,0)/n>=.98)
    required={"open","high","low","close","pre_close","pct_chg","volume","amount"}
    if required-set(out): raise RuntimeError(f"行情核心字段缺失：{sorted(required-set(out))}")
    return out


def load_panels(files,columns):
    symbols=set()
    for i,p in enumerate(files,1):
        symbols.update(c for c in pd.read_csv(p,compression="gzip",usecols=["symbol"])["symbol"].map(normalize_code) if c)
        if i%500==0: print(f"[证券目录] {i}/{len(files)}",flush=True)
    symbols=sorted(symbols);pos={s:i for i,s in enumerate(symbols)}
    panel={c:np.full((len(symbols),len(files)),np.nan,dtype=np.float32) for c in columns}
    present=np.zeros((len(symbols),len(files)),dtype=bool)
    for j,p in enumerate(files):
        df=pd.read_csv(p,compression="gzip");ids=df["symbol"].map(normalize_code).map(pos);ok=ids.notna().to_numpy()
        df=df.loc[ok].copy();df["_pos"]=ids.loc[ok].astype(np.int64).to_numpy();df=df.drop_duplicates("_pos",keep="last")
        rows=df["_pos"].to_numpy(dtype=np.int64);present[rows,j]=True
        for c in columns:
            if c in df: panel[c][rows,j]=pd.to_numeric(df[c],errors="coerce").to_numpy(dtype=np.float32)
        if (j+1)%100==0: print(f"[历史面板] {j+1}/{len(files)} 日期={p.name[:10]}",flush=True)
    return symbols,panel,present


class PITEvents:
    def __init__(self,names,symbols,dates,values):
        self.names=names;self.symbols=symbols;self.dates=dates;self.values=values;self.cursor=0
        self.latest={};self.empty=np.full(len(names),np.nan,dtype=np.float32)
    def advance(self,asof):
        cutoff=np.datetime64(asof,"D")
        while self.cursor<len(self.dates) and self.dates[self.cursor]<=cutoff:
            s=str(self.symbols[self.cursor])
            if s:self.latest[s]=self.values[self.cursor]
            self.cursor+=1
    def matrix(self,symbols):
        return np.asarray([self.latest.get(s,self.empty) for s in symbols],dtype=np.float32) if self.names else np.empty((len(symbols),0),dtype=np.float32)


def load_pit_events():
    events=[]
    for table in TABLES:
        frames=[]
        for p in sorted((FUNDAMENTALS/table).glob("????q?.csv.gz")):
            d=pd.read_csv(p,compression="gzip")
            if d.empty or "symbol" not in d or "pub_date" not in d:continue
            d["symbol"]=d["symbol"].map(normalize_code);d["pub_date"]=pd.to_datetime(d["pub_date"],errors="coerce").dt.floor("D")
            d["report_date"]=pd.to_datetime(d["report_date"],errors="coerce") if "report_date" in d else pd.NaT
            d=d.loc[(d.symbol!="")&d.pub_date.notna()]
            if not d.empty:frames.append(d)
        if not frames:
            print(f"[点时财务] {table} 没有可用公告记录",flush=True);continue
        d=pd.concat(frames,ignore_index=True,sort=False);fields=[]
        for c in d.columns:
            if c in IDENTITY:continue
            nonnull=d[c].notna();n=int(nonnull.sum())
            if n<10:continue
            val=pd.to_numeric(d[c],errors="coerce")
            if int((val.notna()&nonnull).sum())/n>=.98:fields.append(c);d[c]=val
        if not fields:continue
        d=d.sort_values(["pub_date","report_date","symbol"],kind="mergesort")
        events.append(PITEvents([f"fund_{table}_{c}" for c in fields],d.symbol.to_numpy(str),
            d.pub_date.to_numpy(dtype="datetime64[D]"),np.column_stack([d[c].to_numpy(np.float32) for c in fields])))
        print(f"[点时财务] {table} 公告记录={len(d)} 因子={len(fields)}",flush=True)
    return events


def window(panel,i,n): return panel[:,i-n+1:i+1] if i+1>=n else None

def mean_window(panel,i,n):
    out=np.full(panel.shape[0],np.nan,dtype=np.float32);w=window(panel,i,n)
    if w is not None:
        ok=np.isfinite(w).all(axis=1);out[ok]=w[ok].mean(axis=1).astype(np.float32)
    return out

def compound(panel,i,n):
    out=np.full(panel.shape[0],np.nan,dtype=np.float32);w=window(panel,i,n)
    if w is not None:
        ok=np.isfinite(w).all(axis=1);out[ok]=((np.prod(1+w[ok].astype(float)/100,axis=1)-1)*100).astype(np.float32)
    return out

def ratio(a,b):
    out=np.full(len(a),np.nan,dtype=np.float32);ok=np.isfinite(a)&np.isfinite(b)&(b>0)
    out[ok]=((a[ok]/b[ok]-1)*100).astype(np.float32);return out

def rsi(panel,i,n):
    out=np.full(panel.shape[0],np.nan,dtype=np.float32);w=window(panel,i,n)
    if w is None:return out
    ok=np.isfinite(w).all(axis=1)
    if not ok.any():return out
    x=w[ok].astype(float);up=np.maximum(x,0).mean(axis=1);down=np.maximum(-x,0).mean(axis=1)
    val=np.full(len(x),50.);has_loss=down>0;val[has_loss&(up<=0)]=0
    active=has_loss&(up>0);val[active]=100-100/(1+up[active]/down[active]);val[(~has_loss)&(up>0)]=100
    out[np.flatnonzero(ok)]=val.astype(np.float32);return out


def feature_day(i,symbols,p,raw,overnight,ema,kdj):
    n=len(symbols);op,hi,lo,cl=(p[k][:,i] for k in ("open","high","low","close"))
    pc,ret,vol,amt=(p[k][:,i] for k in ("pre_close","pct_chg","volume","amount"))
    f={"raw_"+c:p[c][:,i] for c in raw}
    f["return_1d_pct"]=ret
    for days in (3,5,10,20,60):f[f"return_{days}d_pct"]=compound(p["pct_chg"],i,days)
    f["overnight_1d_pct"]=overnight[:,i]
    for days in (3,5,10):f[f"overnight_{days}d_pct"]=compound(overnight,i,days)
    for days in (5,10,20):
        prev=p["volume"][:,i-days:i] if i>=days else None;out=np.full(n,np.nan,dtype=np.float32)
        if prev is not None:
            ok=np.isfinite(prev).all(axis=1);avg=np.full(n,np.nan,dtype=np.float32);avg[ok]=prev[ok].mean(axis=1)
            good=np.isfinite(vol)&(avg>0);out[good]=vol[good]/avg[good]
        f[f"volume_ratio_{days}d"]=out
    f["amount_20d"]=mean_window(p["amount"],i,20)
    for days in (5,20):
        prev=p["amount"][:,i-days:i] if i>=days else None;out=np.full(n,np.nan,dtype=np.float32)
        if prev is not None:
            ok=np.isfinite(prev).all(axis=1);avg=np.full(n,np.nan,dtype=np.float32);avg[ok]=prev[ok].mean(axis=1)
            good=np.isfinite(amt)&(avg>0);out[good]=amt[good]/avg[good]
        f[f"amount_ratio_{days}d"]=out
    f["turnover_pct"]=p["turnover_pct"][:,i] if "turnover_pct" in p else np.full(n,np.nan,dtype=np.float32);f["change_pct"]=ret
    cr=hi-lo;f["close_strength"]=np.where(np.isfinite(cr)&(cr>0),(cl-lo)/np.where(cr>0,cr,1),.5).astype(np.float32)
    f["intraday_return_pct"]=ratio(cl,op)
    limits=p["high_limit"] if "high_limit" in p else None
    is_limit=((np.isfinite(p["close"])&np.isfinite(limits)&(limits>0)&(p["close"]>=limits*(1-1e-6))).astype(np.float32)
              if limits is not None else np.zeros_like(p["close"],dtype=np.float32))
    for days in (5,10):
        w=window(is_limit,i,days);f[f"limit_up_{days}d_count"]=np.sum(w,axis=1,dtype=np.float32) if w is not None else np.full(n,np.nan,dtype=np.float32)
    ma={}
    for days in (5,10,20,60):ma[days]=mean_window(p["close"],i,days);f[f"close_vs_ma{days}_pct"]=ratio(cl,ma[days])
    for a,b in ((5,10),(5,20),(10,20),(20,60)):f[f"ma{a}_vs_ma{b}_pct"]=ratio(ma[a],ma[b])
    for days in (5,20):
        prev=p["close"][:,i-days-4:i-4] if i>=days+4 else None;old=np.full(n,np.nan,dtype=np.float32)
        if prev is not None:
            ok=np.isfinite(prev).all(axis=1);old[ok]=prev[ok].mean(axis=1)
        f[f"ma{days}_slope_5d_pct"]=ratio(ma[days],old)
    for days in (20,60):
        prev=p["high"][:,i-days:i] if i>=days else None;out=np.full(n,np.nan,dtype=np.float32)
        if prev is not None:
            ok=np.isfinite(prev).all(axis=1);out[ok]=((cl[ok]/np.max(prev[ok],axis=1)-1)*100).astype(np.float32)
        f[f"breakout_{days}d_pct"]=out
    for name,source,func in (("distance_from_high_20d_pct","high",np.max),("distance_from_low_20d_pct","low",np.min)):
        w=window(p[source],i,20);out=np.full(n,np.nan,dtype=np.float32)
        if w is not None:
            ok=np.isfinite(w).all(axis=1);out[np.flatnonzero(ok)]=((cl[ok]/func(w[ok],axis=1)-1)*100).astype(np.float32)
        f[name]=out
    den=np.where(np.isfinite(pc)&(pc>0),pc,cl);candle=np.isfinite(op)&np.isfinite(hi)&np.isfinite(lo)&np.isfinite(cl)&np.isfinite(den)&(den>0)
    for name,v in (("body_pct",cl-op),("upper_shadow_pct",hi-np.maximum(op,cl)),("lower_shadow_pct",np.minimum(op,cl)-lo),("range_pct",hi-lo)):
        out=np.full(n,np.nan,dtype=np.float32);out[candle]=(v[candle]/den[candle]*100).astype(np.float32);f[name]=out
    for days in (5,10,20):
        w=window(p["pct_chg"],i,days);out=np.full(n,np.nan,dtype=np.float32)
        if w is not None:
            ok=np.isfinite(w).all(axis=1);out[ok]=np.std(w[ok],axis=1,ddof=1).astype(np.float32)
        f[f"volatility_{days}d_pct"]=out
    w=window(p["pct_chg"],i,10);out=np.full(n,np.nan,dtype=np.float32)
    if w is not None:
        ok=np.isfinite(w).all(axis=1);out[ok]=np.sqrt(np.mean(np.square(np.minimum(w[ok],0)),axis=1)).astype(np.float32)
    f["downside_volatility_10d_pct"]=out
    w=window(p["close"],i,20);dd=np.full(n,np.nan,dtype=np.float32);bp=dd.copy();bw=dd.copy()
    if w is not None:
        ok=np.isfinite(w).all(axis=1)&np.all(w>0,axis=1)
        if ok.any():
            x=w[ok].astype(float);mid=x.mean(axis=1);sd=x.std(axis=1);upper=mid+2*sd;lower=mid-2*sd
            dd[ok]=(np.min(x/np.maximum.accumulate(x,axis=1)-1,axis=1)*100).astype(np.float32)
            bw[ok]=np.where(mid>0,(upper-lower)/mid*100,np.nan).astype(np.float32)
            pos=np.full(len(x),.5);nz=upper>lower;cs=cl[ok];pos[nz]=(cs[nz]-lower[nz])/(upper[nz]-lower[nz]);bp[np.flatnonzero(ok)]=pos.astype(np.float32)
    f["drawdown_20d_pct"],f["bollinger_position_20d"],f["bollinger_width_20d_pct"]=dd,bp,bw
    atr=np.full(n,np.nan,dtype=np.float32)
    if i>=13:
        h=p["high"][:,i-13:i+1];l=p["low"][:,i-13:i+1];pr=p["pre_close"][:,i-13:i+1]
        ok=np.isfinite(h).all(axis=1)&np.isfinite(l).all(axis=1)&np.isfinite(pr).all(axis=1)
        if ok.any():
            tr=np.maximum.reduce([h[ok]-l[ok],np.abs(h[ok]-pr[ok]),np.abs(l[ok]-pr[ok])])
            atr[ok]=(tr.mean(axis=1)/np.maximum(cl[ok],1e-12)*100).astype(np.float32)
    f["atr_14_pct"]=atr;f["rsi_6"],f["rsi_14"]=rsi(p["pct_chg"],i,6),rsi(p["pct_chg"],i,14)
    e12,e26,sig=ema["12"],ema["26"],ema["signal"];ok=np.isfinite(cl)&(cl>0)
    first=ok&~np.isfinite(e12)
    e12[first]=cl[first];e26[first]=cl[first];sig[first]=0
    active=ok&~first
    if active.any():
        e12[active]=cl[active]*(2/13)+e12[active]*(11/13)
        e26[active]=cl[active]*(2/27)+e26[active]*(25/27)
        m=e12-e26;sig[active]=m[active]*.2+sig[active]*.8
    m=e12-e26;denom=np.where(ok,cl,1)
    f["macd_line_pct"]=np.where(ok,m/denom*100,np.nan).astype(np.float32)
    f["macd_signal_pct"]=np.where(ok,sig/denom*100,np.nan).astype(np.float32)
    f["macd_histogram_pct"]=np.where(ok,(m-sig)/denom*100,np.nan).astype(np.float32)
    wh,wl=window(p["high"],i,9),window(p["low"],i,9);kc=np.full(n,np.nan,dtype=np.float32);dc=kc.copy()
    if wh is not None:
        valid=np.isfinite(wh).all(axis=1)&np.isfinite(wl).all(axis=1)&np.isfinite(cl)
        if valid.any():
            up=np.max(wh[valid],axis=1);low=np.min(wl[valid],axis=1);rsv=np.full(len(up),50.);nz=up>low
            rsv[nz]=(cl[valid][nz]-low[nz])/(up[nz]-low[nz])*100
            pk=kdj["k"][valid].astype(float);pdv=kdj["d"][valid].astype(float)
            if i==0:pk[:]=50;pdv[:]=50
            kv=(2*pk+rsv)/3;dv=(2*pdv+kv)/3
            kdj["k"][np.flatnonzero(valid)]=kv;kdj["d"][np.flatnonzero(valid)]=dv
            kc[np.flatnonzero(valid)]=kv;dc[np.flatnonzero(valid)]=dv
    f["kdj_k_9"],f["kdj_d_9"],f["kdj_j_9"]=kc,dc,(3*kc-2*dc).astype(np.float32)
    vr=np.isfinite(ret);v20=np.isfinite(f["close_vs_ma20_pct"]);v60=np.isfinite(f["close_vs_ma60_pct"]);v5=np.isfinite(f["return_5d_pct"])
    market={
        "market_breadth_pct":float(((ret>0)&vr).sum()/max(int(vr.sum()),1)*100),
        "market_median_return_pct":float(np.median(ret[vr])) if vr.any() else math.nan,
        "market_return_dispersion_pct":float(np.std(ret[vr],ddof=1)) if vr.sum()>1 else math.nan,
        "market_above_ma20_pct":float((f["close_vs_ma20_pct"][v20]>0).mean()*100) if v20.any() else math.nan,
        "market_above_ma60_pct":float((f["close_vs_ma60_pct"][v60]>0).mean()*100) if v60.any() else math.nan,
        "market_positive_5d_pct":float(((f["return_5d_pct"]>0)&v5).sum()/max(int(v5.sum()),1)*100)}
    for name,value in market.items():f[name]=np.full(n,value,dtype=np.float32)
    for name in TECH:f.setdefault(name,np.full(n,np.nan,dtype=np.float32))
    result=pd.DataFrame(f);result.insert(0,"symbol",symbols);return result


def classify_net_return(net,band):
    if not math.isfinite(float(net)):return "不可确认"
    if net>band:return "盈利"
    if net< -band:return "亏损"
    return "平盘"


def label_day(entry,exit_price,entry_paused,exit_paused,upper,lower,costs,band):
    n=len(entry);valid=np.isfinite(entry)&(entry>0)&np.isfinite(exit_price)&(exit_price>0)
    gross=np.full(n,np.nan,dtype=np.float32);gross[valid]=((exit_price[valid]/entry[valid]-1)*100).astype(np.float32)
    net=gross-np.float32(costs/100);theory=np.full(n,"不可确认",dtype=object);finite=np.isfinite(net);theory[finite]="平盘"
    theory[finite&(net>band)]="盈利";theory[finite&(net< -band)]="亏损"
    es=np.full(n,"可买入",dtype=object);xs=np.full(n,"可卖出",dtype=object)
    es[~np.isfinite(entry)|(entry<=0)]="T+1开盘价缺失或无效";xs[~np.isfinite(exit_price)|(exit_price<=0)]="T+2开盘价缺失或无效"
    es[~np.isfinite(entry_paused)|(entry_paused>0)]="T+1停牌或状态缺失";xs[~np.isfinite(exit_paused)|(exit_paused>0)]="T+2停牌或状态缺失"
    es[np.isfinite(entry)&np.isfinite(upper)&(upper>0)&(entry>=upper*(1-1e-6))]="T+1开盘涨停，保守排除"
    xs[np.isfinite(exit_price)&np.isfinite(lower)&(lower>0)&(exit_price<=lower*(1+1e-6))]="T+2开盘跌停，保守排除"
    es[~np.isfinite(upper)|(upper<=0)]="T+1涨停价缺失，无法确认成交";xs[~np.isfinite(lower)|(lower<=0)]="T+2跌停价缺失，无法确认成交"
    can=valid&(es=="可买入")&(xs=="可卖出");result=np.full(n,"不可执行",dtype=object);result[can]=theory[can]
    status=np.full(n,"理论收益可算，但交易不可确认",dtype=object);status[can]="可执行";status[~valid]="价格缺失"
    return pd.DataFrame({"buy_open":entry.astype(np.float32),"sell_open":exit_price.astype(np.float32),
        "gross_return_pct":gross,"round_trip_cost_bps":np.full(n,costs,dtype=np.float32),
        "net_return_pct":net.astype(np.float32),"theoretical_result_class":theory,"result_class":result,
        "entry_status":es,"exit_status":xs,"label_status":status})


def rank_frame(df,names,config):
    ranks=df.reindex(columns=names).rank(axis=0,method="average",pct=True)
    for name,rule in config["市场因子缩放"].items():
        if name in ranks and name in df:
            ranks[name]=(pd.to_numeric(df[name],errors="coerce")/float(rule["除数"])+float(rule.get("偏移",0))).clip(
                lower=float(rule.get("最小值",0)),upper=float(rule.get("最大值",1)))
    return ranks.replace([np.inf,-np.inf],np.nan).fillna(.5).to_numpy(np.float32)


class Moments:
    def __init__(self,n):self.rows=0;self.count=np.zeros(n,dtype=np.int64);self.total=np.zeros(n);self.square=np.zeros(n)
    def update(self,x):
        if not len(x):return
        a=np.asarray(x,float);ok=np.isfinite(a);self.rows+=len(a);self.count+=ok.sum(axis=0)
        clean=np.where(ok,a,0);self.total+=clean.sum(axis=0);self.square+=(clean*clean).sum(axis=0)
    def report(self,names,sample):
        out=[]
        for i,name in enumerate(names):
            c=int(self.count[i]);m=float(self.total[i]/c) if c else None
            v=max(float(self.square[i]/c)-m*m,0) if c else None
            a=sample[:,i] if len(sample) else np.empty(0);a=a[np.isfinite(a)];q=np.quantile(a,[.25,.5,.75]) if len(a) else [math.nan]*3
            out.append({"因子":name,"样本数":self.rows,"有效值数":c,"缺失率百分比":round((1-c/self.rows)*100,4) if self.rows else None,
                "均值":round(m,8) if m is not None else None,"标准差":round(math.sqrt(v),8) if v is not None else None,
                "抽样第一四分位":round(float(q[0]),8) if math.isfinite(float(q[0])) else None,
                "抽样中位数":round(float(q[1]),8) if math.isfinite(float(q[1])) else None,
                "抽样第三四分位":round(float(q[2]),8) if math.isfinite(float(q[2])) else None})
        return out


class Reservoir:
    def __init__(self,cap,n,rng):
        self.cap=cap;self.rng=rng;self.keys=np.empty(0,np.float32);self.x=np.empty((0,n),np.float32)
        self.labels=np.empty(0,np.int8);self.returns=np.empty(0,np.float32)
    def add(self,x,labels=None,returns=None):
        if not len(x):return
        keys=np.r_[self.keys,self.rng.random(len(x),dtype=np.float32)];xx=np.concatenate((self.x,np.asarray(x,np.float32)))
        yy=np.r_[self.labels,np.asarray(labels if labels is not None else np.full(len(x),-1),np.int8)]
        rr=np.r_[self.returns,np.asarray(returns if returns is not None else np.full(len(x),np.nan),np.float32)]
        if len(keys)>self.cap:
            ix=np.argpartition(keys,-self.cap)[-self.cap:];keys,xx,yy,rr=keys[ix],xx[ix],yy[ix],rr[ix]
        self.keys,self.x,self.labels,self.returns=keys,xx,yy,rr


def pick(ix,cap,rng):return ix if len(ix)<=cap else rng.choice(ix,cap,replace=False)

def metrics(returns,classes):
    ok=np.isfinite(returns);x=np.asarray(returns[ok],float);c=np.asarray(classes)[ok]
    if not len(x):return {"样本数":0,"平均净收益百分比":None,"盈利因子":None}
    pos=float(x[x>0].sum());neg=float(-x[x<0].sum());pf=pos/neg if neg>0 else ("无亏损样本" if pos>0 else None)
    return {"样本数":len(x),"盈利标签数":int((c=="盈利").sum()),"亏损标签数":int((c=="亏损").sum()),"平盘标签数":int((c=="平盘").sum()),
        "盈利标签比例百分比":round(float((c=="盈利").mean()*100),4),"平均净收益百分比":round(float(x.mean()),6),
        "净收益为正比例百分比":round(float((x>0).mean()*100),4),"正收益金额合计":round(pos,6),"负收益绝对值合计":round(neg,6),
        "盈利因子":round(pf,6) if isinstance(pf,(int,float)) else pf}

def positive_probability(model,x):
    return model.predict_proba(x)[:,list(model.classes_).index(1)].astype(np.float32) if 1 in model.classes_ else np.zeros(len(x),np.float32)

def choose_threshold(model,res,config):
    if not len(res.x):return None,[],False
    prob=positive_probability(model,res.x);labels=np.asarray([CODE_CLASS.get(int(c),"不可确认") for c in res.labels]);out=[]
    for threshold in config["公式概率阈值网格"]:
        m=prob>=float(threshold);row=metrics(res.returns[m],labels[m]);row["概率阈值"]=float(threshold);pf=row.get("盈利因子")
        row["通过验证门槛"]=bool(row.get("样本数",0)>=config["验证期候选最低样本数"] and
            isinstance(row.get("平均净收益百分比"),(int,float)) and row["平均净收益百分比"]>config["验证期最低平均净收益百分比"] and
            isinstance(pf,(int,float)) and pf>=config["验证期最低盈利因子"]);out.append(row)
    good=[x for x in out if x["通过验证门槛"]]
    if not good:return None,out,False
    good.sort(key=lambda r:(r["平均净收益百分比"],r["盈利因子"],r["样本数"]),reverse=True)
    return good[0]["概率阈值"],out,True

def get_rules(model,train_x,names,threshold):
    if not len(train_x):return []
    probs=positive_probability(model,train_x);leaf_ids=model.apply(train_x);stat={}
    for node in np.unique(leaf_ids):
        m=leaf_ids==node;stat[int(node)]=(float(probs[m].mean()),int(m.sum()))
    tree=model.tree_;out=[]
    def walk(node,conditions):
        fi=int(tree.feature[node])
        if fi==_tree.TREE_UNDEFINED:
            x=stat.get(node)
            if x and x[0]>=threshold:out.append({"leaf_id":node,"训练抽样盈利概率":round(x[0],6),"训练抽样样本数":x[1],"conditions":conditions})
            return
        name=names[fi];cut=round(float(tree.threshold[node]),8)
        walk(int(tree.children_left[node]),conditions+[{"factor":name,"operator":"<=","threshold":cut}])
        walk(int(tree.children_right[node]),conditions+[{"factor":name,"operator":">","threshold":cut}])
    walk(0,[]);return out

def evaluate_split(paths,model,names,threshold,split,config):
    n=win=loss=flat=pos_count=0;sum_ret=pos=neg=0.;dates=set()
    cols=["signal_date","split","split_eligible","result_class","net_return_pct"]+names
    for p in paths:
        pf=pq.ParquetFile(p)
        for rg in range(pf.num_row_groups):
            d=pf.read_row_group(rg,columns=cols).to_pandas()
            mask=(d.split.to_numpy()==split)&d.split_eligible.to_numpy(bool)&d.result_class.isin(CLASSES).to_numpy()&pd.to_numeric(d.net_return_pct,errors="coerce").notna().to_numpy()
            if not mask.any():continue
            all_ranks=rank_frame(d,names,config)
            probs=positive_probability(model,all_ranks[mask])
            x=d.loc[mask].reset_index(drop=True);x=x.loc[probs>=threshold]
            if x.empty:continue
            r=pd.to_numeric(x.net_return_pct,errors="coerce").to_numpy(float);c=x.result_class.astype(str).to_numpy();ok=np.isfinite(r);r,c=r[ok],c[ok]
            n+=len(r);win+=int((c=="盈利").sum());loss+=int((c=="亏损").sum());flat+=int((c=="平盘").sum())
            sum_ret+=float(r.sum());pos+=float(r[r>0].sum());neg+=float(-r[r<0].sum());pos_count+=int((r>0).sum())
            dates.update(x.loc[ok,"signal_date"].astype(str).tolist())
    pf=pos/neg if neg>0 else ("无亏损样本" if pos>0 else None)
    return {"评估口径":"完整全量信号样本；不是现金约束组合净值回测","时间分段":split,"选中信号样本数":n,"有信号交易日数":len(dates),
        "盈利标签数":win,"亏损标签数":loss,"平盘标签数":flat,"盈利标签比例百分比":round(win/n*100,4) if n else None,
        "平均净收益百分比":round(sum_ret/n,6) if n else None,"净收益为正比例百分比":round(pos_count/n*100,4) if n else None,
        "盈利因子":round(pf,6) if isinstance(pf,(int,float)) else pf}

def split_ranges(n,c):
    a=min(max(int(n*c["训练期结束比例"]),1),n-2);b=min(max(int(n*c["验证期结束比例"]),a+1),n-1)
    return {"训练期":(0,a),"验证期":(a,b),"最终留出期":(b,n)}


def main():
    parser=argparse.ArgumentParser(description="全市场 T 日全因子与 T+1/T+2 开盘三分类研究")
    parser.add_argument("--output",default="data/research/latest_report.json");parser.add_argument("--samples-dir",default="data/research/samples")
    args=parser.parse_args();started=time.time();cfg=load_config();files=history_files();all_dates=[p.name[:10] for p in files];dates=all_dates[:-2]
    splits=split_ranges(len(dates),cfg);by_day={};end_at={}
    for name,(a,b) in splits.items():
        end_at[name]=b
        for i in range(a,b):by_day[i]=name
    raw=discover_raw_columns(files);symbols,p,present=load_panels(files,raw)
    print(f"[预检] 行情={all_dates[0]}至{all_dates[-1]}，交易日={len(files)}，证券={len(symbols)}，原始数值因子={len(raw)}",flush=True)
    overnight=np.full_like(p["open"],np.nan);ok=np.isfinite(p["open"])&np.isfinite(p["pre_close"])&(p["pre_close"]>0)
    overnight[ok]=((p["open"][ok]/p["pre_close"][ok]-1)*100).astype(np.float32)
    events=load_pit_events();fin_names=[n for e in events for n in e.names]
    names=["raw_"+x for x in raw]+list(TECH)+fin_names+list(MARKET)
    if len(names)!=len(set(names)):raise RuntimeError("因子字段重名")
    nf=len(names);outdir=Path(args.samples_dir)
    if outdir.exists():shutil.rmtree(outdir)
    outdir.mkdir(parents=True,exist_ok=True);writers={};parquets=[];rng=np.random.default_rng(cfg["随机种子"])
    moments={(sp,cl):Moments(nf) for sp in splits for cl in CLASSES}
    quant={(sp,cl):Reservoir(cfg["每组分位数抽样上限"],nf,rng) for sp in splits for cl in CLASSES}
    counts={sp:{cl:0 for cl in CLASSES} for sp in splits}
    sample_counts={sp:{"完整样本行数":0,"可执行三分类样本数":0,"理论结果可算但不可执行数":0,"价格缺失数":0,"跨分段标签剔除数":0} for sp in splits}
    train=Reservoir(cfg["每个时间分段最大模型样本数"],nf,rng);valid_res=Reservoir(cfg["每个时间分段最大模型样本数"],nf,rng)
    ema={x:np.full(len(symbols),np.nan,np.float32) for x in ("12","26")};ema["signal"]=np.zeros(len(symbols),np.float32)
    kdj={"k":np.full(len(symbols),50,np.float32),"d":np.full(len(symbols),50,np.float32)}
    try:
        for i,date in enumerate(dates):
            active=np.flatnonzero(present[:,i]);daily_symbols=[symbols[j] for j in active]
            f_all=feature_day(i,symbols,p,raw,overnight,ema,kdj)
            f=f_all.iloc[active].reset_index(drop=True)
            extra=[]
            for e in events:
                e.advance(date)
                if e.names:extra.append(pd.DataFrame(e.matrix(daily_symbols),columns=e.names))
            if extra:f=pd.concat([f,pd.concat(extra,axis=1)],axis=1)
            f=f.reindex(columns=["symbol"]+names);raw_x=f[names].to_numpy(np.float32)
            pause_panel=p["is_paused"] if "is_paused" in p else np.full_like(p["close"],np.nan)
            upper_panel=p["high_limit"] if "high_limit" in p else np.full_like(p["close"],np.nan)
            lower_panel=p["low_limit"] if "low_limit" in p else np.full_like(p["close"],np.nan)
            labels=label_day(p["open"][active,i+1],p["open"][active,i+2],
                pause_panel[active,i+1],pause_panel[active,i+2],
                upper_panel[active,i+1],lower_panel[active,i+2],
                float(cfg["往返交易成本基点"]),float(cfg["平盘容忍百分比"]))
            sp=by_day[i];eligible=True if sp=="最终留出期" else i+2<end_at[sp]
            cls=labels.result_class.astype(str).to_numpy();theory=labels.theoretical_result_class.astype(str).to_numpy()
            ret=pd.to_numeric(labels.net_return_pct,errors="coerce").to_numpy(np.float32)
            sc=sample_counts[sp];sc["完整样本行数"]+=len(active);sc["可执行三分类样本数"]+=int(np.isin(cls,CLASSES).sum())
            sc["理论结果可算但不可执行数"]+=int(np.isin(theory,CLASSES).sum()-np.isin(cls,CLASSES).sum())
            sc["价格缺失数"]+=int((theory=="不可确认").sum())
            if not eligible:sc["跨分段标签剔除数"]+=int(np.isin(cls,CLASSES).sum())
            if eligible:
                for label in CLASSES:
                    m=cls==label;counts[sp][label]+=int(m.sum())
                    if m.any():moments[(sp,label)].update(raw_x[m]);quant[(sp,label)].add(raw_x[m])
                ranks=rank_frame(f,names,cfg)
                if sp=="训练期":
                    pos=pick(np.flatnonzero(cls=="盈利"),cfg["每日每类最大训练抽样数"],rng)
                    neg=pick(np.flatnonzero(cls=="亏损"),cfg["每日每类最大训练抽样数"],rng);chosen=np.r_[pos,neg]
                    if len(chosen):train.add(ranks[chosen],np.array([int(cls[j]=="盈利") for j in chosen],np.int8),ret[chosen])
                elif sp=="验证期":
                    ix=np.flatnonzero(np.isin(cls,CLASSES)&np.isfinite(ret));chosen=pick(ix,cfg["每日最大验证抽样数"],rng)
                    if len(chosen):valid_res.add(ranks[chosen],np.array([CLASS_CODE[cls[j]] for j in chosen],np.int8),ret[chosen])
            out=f.copy()
            for c in labels.columns:out[c]=labels[c].to_numpy()
            out.insert(0,"signal_date",date);out["buy_date"]=all_dates[i+1];out["sell_date"]=all_dates[i+2]
            out["split"]=sp;out["split_eligible"]=bool(eligible)
            out=out.reindex(columns=["signal_date","symbol","buy_date","sell_date"]+list(labels.columns)+["split","split_eligible"]+names)
            year=date[:4];path=outdir/f"year={year}.parquet";table=pa.Table.from_pandas(out,preserve_index=False)
            if year not in writers:writers[year]=pq.ParquetWriter(path,table.schema,compression="zstd",use_dictionary=True,write_statistics=True);parquets.append(path)
            writers[year].write_table(table,row_group_size=max(1,len(out)))
            if (i+1)%int(cfg["进度日志间隔交易日"])==0:print(f"[样本构建] {i+1}/{len(dates)} 日期={date} 当日股票={len(active)} 因子={nf}",flush=True)
    finally:
        for w in writers.values():w.close()
    print(f"[统计完成] 样本行={sum(x['完整样本行数'] for x in sample_counts.values())} 因子={nf}",flush=True)
    model=None;model_names=[];threshold=None;thresholds=[];passed=False;evals={}
    if len(train.x) and len(np.unique(train.labels))>=2:
        wm,lm=moments[("训练期","盈利")],moments[("训练期","亏损")]
        coverage=(wm.count+lm.count)/max(wm.rows+lm.rows,1);variable=np.std(train.x,axis=0)>1e-6
        ix=[i for i in range(nf) if coverage[i]>=cfg["最低训练期因子覆盖率"] and variable[i]]
        model_names=[names[i] for i in ix]
        if ix:
            model=DecisionTreeClassifier(max_depth=int(cfg["决策树最大深度"]),min_samples_leaf=int(cfg["决策树每叶最小样本数"]),class_weight="balanced",random_state=int(cfg["随机种子"]))
            model.fit(train.x[:,ix],train.labels);v=Reservoir(cfg["每个时间分段最大模型样本数"],len(ix),rng)
            v.x=valid_res.x[:,ix];v.labels=valid_res.labels.copy();v.returns=valid_res.returns.copy()
            threshold,thresholds,passed=choose_threshold(model,v,cfg)
            if threshold is not None:
                for sp in splits:evals[sp]=evaluate_split(parquets,model,model_names,threshold,sp,cfg)
    rules=[]
    if model is not None and threshold is not None:rules=get_rules(model,train.x[:,[names.index(n) for n in model_names]],model_names,threshold)
    hold=evals.get("最终留出期",{});pf=hold.get("盈利因子")
    hold_pass=bool(passed and hold.get("选中信号样本数",0)>=cfg["最终留出期候选最低样本数"] and
        isinstance(hold.get("平均净收益百分比"),(int,float)) and hold["平均净收益百分比"]>0 and
        isinstance(pf,(int,float)) and pf>=cfg["最终留出期最低盈利因子"])
    comparisons={}
    for sp in splits:
        stats={cl:moments[(sp,cl)].report(names,quant[(sp,cl)].x) for cl in CLASSES}
        maps={cl:{r["因子"]:r for r in stats[cl]} for cl in CLASSES};rows=[]
        for name in names:
            a,b,c=(maps[cl][name] for cl in CLASSES);effect=None
            if a["均值"] is not None and b["均值"] is not None:
                pooled=math.sqrt(max(((a["标准差"] or 0)**2+(b["标准差"] or 0)**2)/2,0))
                effect=(a["均值"]-b["均值"])/pooled if pooled>1e-12 else 0.
            rows.append({"因子":name,"盈利组":a,"亏损组":b,"平盘组":c,
                "盈利减亏损标准化均值差":round(effect,6) if effect is not None else None,
                "盈利组均值高于亏损组":bool(effect>0) if effect is not None else None})
        rows.sort(key=lambda x:abs(x["盈利减亏损标准化均值差"]) if x["盈利减亏损标准化均值差"] is not None else -1,reverse=True)
        comparisons[sp]={"结果样本数":counts[sp],"全因子比较":rows,"差异最大的前二十个因子":rows[:20]}
    split_report={}
    for sp,(a,b) in splits.items():
        latest=b-3 if sp!="最终留出期" else b-1
        split_report[sp]={"开始日期":dates[a],"分段信号结束日期":dates[b-1],
            "参与差异分析最晚信号日期":dates[max(a,latest)],"跨分段标签样本已剔除":sp!="最终留出期"}
    formula={"status":"validation_passed" if passed else ("validation_rejected" if model is not None else "unavailable"),
        "selected_probability_threshold":threshold,"model_type":"最大深度受限的可解释决策树","model_max_depth":int(cfg["决策树最大深度"]),
        "features":model_names,"selected_leaf_rules":rules,"validation_threshold_metrics":thresholds,
        "validation_passed":passed,"final_holdout_passed":hold_pass,"final_holdout_metrics":hold,
        "eligible_for_production":False,"automatic_ordering":False,
        "说明":"正式发布保持关闭；历史研究通过仍需现金约束组合风险和未来影子模拟验收。"}
    report={"schema_version":1,"状态":"全因子分组研究报告已生成",
        "研究目标":{"信号":"T日收盘","买入":"T+1开盘","卖出":"T+2开盘",
        "净收益公式":"(T+2开盘价 / T+1开盘价 - 1) × 100% - 往返交易成本",
        "往返交易成本基点":cfg["往返交易成本基点"],"平盘容忍区间百分比":cfg["平盘容忍百分比"],
        "三分类规则":"净收益高于正平盘边界为盈利，低于负边界为亏损，区间含边界为平盘",
        "不可执行状态":"记录理论收益，同时单独标注停牌、开盘涨跌停、价格缺失","未来函数":False},
        "数据范围":{"历史数据库开始日期":all_dates[0],"历史数据库最后信号日期":dates[-1],"历史行情文件数":len(files),
        "信号交易日数":len(dates),"原始数值行情因子数":len(raw),"衍生技术因子数":len(TECH),"市场因子数":len(MARKET),
        "点时财务因子数":len(fin_names),"可用因子总数":nf,"证券代码总数":len(symbols),
        "完整样本行数":sum(x["完整样本行数"] for x in sample_counts.values()),
        "可执行三分类样本数":sum(x["可执行三分类样本数"] for x in sample_counts.values()),
        "因子清单":names,"样本文件格式":"按年份分区的Parquet；每个信号日一个行组，保存实际因子值和收益标签"},
        "时间分段":split_report,"分组对比":comparisons,
        "单因子差异解释":{"标准化均值差":"（盈利组均值 - 亏损组均值）/ 两组方差平均值的平方根",
        "说明":"均值、标准差和缺失率按全量样本累计；四分位数使用固定种子有界样本；关联不代表因果"},
        "验证期公式挖掘":{"验证抽样阈值比较":thresholds,"选择概率阈值":threshold,"验证通过":passed},
        "首版公式候选":formula,"全量公式评估":evals,"各分段样本概览":sample_counts,
        "模型样本抽样":{"训练样本数":len(train.x),"验证样本数":len(valid_res.x),"模型因子数":len(model_names),
        "规则搜索限制":"训练期拟合、验证期选阈值、最终留出期只评估，不反向改公式"},
        "耗时秒数":round(time.time()-started,2),"正式发布":False,"自动下单":False}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);tmp=output.with_suffix(output.suffix+".tmp")
    tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8");tmp.replace(output)
    print(json.dumps({"状态":report["状态"],"数据范围":report["数据范围"],"时间分段":report["时间分段"],
        "首版公式候选":{"status":formula["status"],"验证通过":passed,"最终留出通过":hold_pass,"阈值":threshold,"规则数":len(rules),"正式发布":False},
        "报告路径":str(output),"样本目录":str(outdir),"耗时秒数":report["耗时秒数"]},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__": main()
