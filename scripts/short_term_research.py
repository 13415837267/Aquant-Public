"""Short-term 1-5 session research for the production candidate strategy.

Research trigger: Aquant-Private/main current strategy regression validation.
"""
from __future__ import annotations

import argparse, gzip, importlib, json, os, subprocess, sys, time, traceback
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.selection_factor_catalog import load_research_config
RESEARCH_CONFIG = load_research_config()
HISTORY_DIR = ROOT / "data" / "history"
OUT_DIR = ROOT / "data" / "backtest"
MAX_HOLD = int(RESEARCH_CONFIG["最大前瞻交易日数"])
MIN_EXIT_DAY = int(RESEARCH_CONFIG["最早允许退出日序号"])
MAX_HOLDING_SESSIONS = MAX_HOLD
FEATURE_WARMUP_SESSIONS = 61
FEATURE_SCHEMA_VERSION = 2
ENTRY_LIMIT_UP_BLOCK = bool(RESEARCH_CONFIG["开盘涨停禁止入场"])
STOP_PCT = float(RESEARCH_CONFIG["止损幅度百分比"])
WIN_THRESHOLD_PCT = float(RESEARCH_CONFIG["短线目标净收益百分比"])
ROUND_TRIP_COST_BPS = float(RESEARCH_CONFIG["往返交易成本基点"])
TARGET_PCT = WIN_THRESHOLD_PCT + ROUND_TRIP_COST_BPS / 100.0


def executable_entry_mask(symbols, entry_day: pd.DataFrame) -> np.ndarray:
    """Return entries executable at T+1 open under the production entry rules."""
    count = len(symbols)
    if count == 0 or entry_day.empty or "symbol" not in entry_day or "open" not in entry_day:
        return np.zeros(count, dtype=bool)

    keys = pd.Index(
        pd.Series(symbols, dtype="string").astype(str)
        .str.extract(r"(\d{6})")[0].fillna("").str.zfill(6)
    )
    day = entry_day.copy()
    day["_symbol_key"] = (
        day["symbol"].astype(str).str.extract(r"(\d{6})")[0].fillna("").str.zfill(6)
    )
    day = day.drop_duplicates("_symbol_key", keep="last").set_index("_symbol_key")
    entry = pd.to_numeric(day["open"], errors="coerce").reindex(keys).to_numpy(dtype=np.float64)
    executable = np.isfinite(entry) & (entry > 0)

    if ENTRY_LIMIT_UP_BLOCK and "high_limit" in day:
        high_limit = pd.to_numeric(day["high_limit"], errors="coerce").reindex(keys).to_numpy(dtype=np.float64)
        blocked = np.isfinite(high_limit) & (entry >= high_limit * (1.0 - 1e-6))
        executable &= ~blocked
    return executable


def history_files():
    paths = sorted(HISTORY_DIR.glob("????-??-??.csv.gz"), key=lambda p: p.name[:10])
    if not paths:
        paths = sorted(HISTORY_DIR.glob("????/*.csv.gz"), key=lambda p: p.name[:10])
    if not paths:
        raise RuntimeError("no historical files")
    return paths


def read_daily(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(fh)
    df = df.loc[df["symbol"].astype(str).map(lambda x: len(x) >= 6)].copy()
    df["symbol"] = df["symbol"].astype(str).str.extract(r"(\d{6})")[0].str.zfill(6)
    numeric = ["open","high","low","close","volume","amount","pct_chg","turnover_pct",
               "is_paused","is_st","high_limit","low_limit"]
    for c in numeric:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    from scripts.market_scope import is_main_board_symbol
    df = df.loc[df["symbol"].map(is_main_board_symbol)].drop_duplicates("symbol", keep="last")
    if df.empty:
        raise RuntimeError(f"no main-board rows in {path.name}")
    return df


def load_strategy():
    path = os.environ.get("AQUANT_PRIVATE_STRATEGY_PATH", "").strip()
    if not path:
        raise RuntimeError("AQUANT_PRIVATE_STRATEGY_PATH is required")
    root = Path(path).resolve()
    sys.path.insert(0, str(root))
    model = importlib.import_module("strategy.model")
    version = importlib.import_module("strategy.version")
    commit = os.environ.get("AQUANT_PRIVATE_STRATEGY_COMMIT", "").strip()
    if not commit:
        commit = subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"], text=True).strip()
    return model, str(version.STRATEGY_VERSION), commit


class FeatureState:
    """按证券保存滚动历史，只用当前及过去已完成日线构造因子。"""
    def __init__(self):
        self.returns={}; self.volumes={}; self.amounts={}; self.closes={}
        self.highs={}; self.lows={}; self.overnights={}; self.limitups={}
        self.ema12={}; self.ema26={}; self.macd_signal={}; self.kdj_k={}; self.kdj_d={}

    def build(self, day: pd.DataFrame) -> pd.DataFrame:
        rows=[]
        for r in day.itertuples(index=False):
            s=str(r.symbol).zfill(6)
            ret=float(r.pct_chg)/100.0 if pd.notna(r.pct_chg) else np.nan
            volume=float(r.volume) if pd.notna(r.volume) else np.nan
            amount=float(r.amount) if pd.notna(r.amount) else np.nan
            close=float(r.close) if pd.notna(r.close) else np.nan
            high=float(r.high) if pd.notna(r.high) else np.nan
            low=float(r.low) if pd.notna(r.low) else np.nan
            op=float(r.open) if pd.notna(r.open) else np.nan
            names=("returns","volumes","amounts","closes","highs","lows","overnights","limitups")
            state={name:getattr(self,name).setdefault(s,deque(maxlen=FEATURE_WARMUP_SESSIONS)) for name in names}
            prior=state["closes"][-1] if state["closes"] else np.nan
            overnight=(op/prior-1)*100 if np.isfinite(op) and np.isfinite(prior) and prior>0 else np.nan
            hl=float(r.high_limit) if pd.notna(r.high_limit) else np.nan
            limit_up=int(np.isfinite(hl) and np.isfinite(close) and close>=hl*(1-1e-6))
            for name,value in (("returns",ret),("volumes",volume),("amounts",amount),("closes",close),("highs",high),("lows",low),("overnights",overnight),("limitups",limit_up)):
                state[name].append(value)
            rv=list(state["returns"]); vv=list(state["volumes"]); av=list(state["amounts"])
            cv=list(state["closes"]); hv=list(state["highs"]); lv=list(state["lows"])
            ov=list(state["overnights"]); lu=list(state["limitups"])

            def mean(values,n):
                x=np.asarray(values[-n:],dtype=np.float64)
                return float(x.mean()) if len(x)==n and np.isfinite(x).all() else np.nan
            def rel(a,b):
                return (a/b-1)*100 if np.isfinite(a) and np.isfinite(b) and b>0 else np.nan
            def compounded(values,n,scale=1.0):
                x=np.asarray(values[-n:],dtype=np.float64)
                return (float(np.prod(1+x/scale))-1)*scale if len(x)==n and np.isfinite(x).all() else np.nan
            def ratio(values,n,current):
                base=mean(values[:-1],n)
                return current/base if np.isfinite(current) and np.isfinite(base) and base>0 else np.nan
            ma={n:mean(cv,n) for n in (5,10,20,60)}
            pairs={(5,10):rel(ma[5],ma[10]),(5,20):rel(ma[5],ma[20]),(10,20):rel(ma[10],ma[20]),(20,60):rel(ma[20],ma[60])}
            def calc_rsi(n):
                x=np.asarray(rv[-n:],dtype=np.float64)
                if len(x)!=n or not np.isfinite(x).all(): return np.nan
                gains=float(np.maximum(x,0).mean()); losses=float(np.maximum(-x,0).mean())
                if losses<=0: return 100.0 if gains>0 else 50.0
                if gains<=0: return 0.0
                return 100.0-100.0/(1.0+gains/losses)
            vol={}
            for n in (5,10,20):
                x=np.asarray(rv[-n:],dtype=np.float64)
                vol[n]=float(np.std(x,ddof=1)*100) if len(x)==n and np.isfinite(x).all() and n>1 else np.nan
            down=np.asarray(rv[-10:],dtype=np.float64)
            downside=float(np.sqrt(np.mean(np.square(np.minimum(down,0))))*100) if len(down)==10 and np.isfinite(down).all() else np.nan
            close20=cv[-20:]; high20=hv[-20:]; low20=lv[-20:]
            width=high-low if np.isfinite(high) and np.isfinite(low) else np.nan
            strength=(close-low)/width if np.isfinite(close) and np.isfinite(low) and np.isfinite(width) and width>0 else 0.5
            breakout20=rel(close,max(hv[-21:-1])) if len(hv)>=21 and np.isfinite(hv[-21:-1]).all() else np.nan
            breakout60=rel(close,max(hv[-61:-1])) if len(hv)>=61 and np.isfinite(hv[-61:-1]).all() else np.nan
            distance_high=rel(close,max(high20)) if len(high20)==20 and np.isfinite(high20).all() else np.nan
            distance_low=rel(close,min(low20)) if len(low20)==20 and np.isfinite(low20).all() else np.nan
            bbpos=bbwidth=np.nan
            if len(close20)==20 and np.isfinite(close20).all():
                middle=float(np.mean(close20)); std=float(np.std(close20,ddof=0)); upper=middle+2*std; lower=middle-2*std
                if middle>0: bbwidth=(upper-lower)/middle*100
                bbpos=(close-lower)/(upper-lower) if upper>lower else 0.5
            dd20=np.nan
            if len(close20)==20 and np.isfinite(close20).all():
                x=np.asarray(close20,dtype=float); dd20=float(np.min(x/np.maximum.accumulate(x)-1)*100)
            atr=np.nan
            if len(cv)>=15 and close>0:
                hs=np.asarray(hv[-14:],dtype=float); ls=np.asarray(lv[-14:],dtype=float); prev=np.asarray(cv[-15:-1],dtype=float)
                if np.isfinite(hs).all() and np.isfinite(ls).all() and np.isfinite(prev).all():
                    atr=float(np.mean(np.maximum(hs-ls,np.maximum(np.abs(hs-prev),np.abs(ls-prev))))/close*100)
            macd1=macd2=macd3=np.nan
            if np.isfinite(close) and close>0:
                e12=2/13*close+11/13*self.ema12.get(s,close)
                e26=2/27*close+25/27*self.ema26.get(s,close)
                line=e12-e26; signal=2/10*line+8/10*self.macd_signal.get(s,line)
                self.ema12[s]=e12; self.ema26[s]=e26; self.macd_signal[s]=signal
                if len(cv)>=26: macd1=line/close*100; macd2=signal/close*100; macd3=(line-signal)/close*100
            k=d=j=np.nan; hs9=np.asarray(hv[-9:],dtype=float); ls9=np.asarray(lv[-9:],dtype=float)
            if len(hs9)==9 and np.isfinite(hs9).all() and np.isfinite(ls9).all() and np.isfinite(close):
                hi9=float(hs9.max()); lo9=float(ls9.min()); rsv=(close-lo9)/(hi9-lo9)*100 if hi9>lo9 else 50.0
                k=(2*self.kdj_k.get(s,50.0)+rsv)/3; d=(2*self.kdj_d.get(s,50.0)+k)/3; j=3*k-2*d
                self.kdj_k[s]=k; self.kdj_d[s]=d
            rows.append({
                "symbol":s,"close":close,"high":high,"low":low,"amount":amount,
                "turnover_pct":float(r.turnover_pct) if pd.notna(r.turnover_pct) else np.nan,
                "change_pct":float(r.pct_chg) if pd.notna(r.pct_chg) else np.nan,
                "return_1d_pct":compounded(rv,1)*100.0,"return_3d_pct":compounded(rv,3)*100.0,"return_5d_pct":compounded(rv,5)*100.0,
                "return_10d_pct":compounded(rv,10)*100.0,"return_20d_pct":compounded(rv,20)*100.0,
                "return_60d_pct":rel(close,cv[-61]) if len(cv)>=61 else np.nan,
                "overnight_1d_pct":overnight,"overnight_3d_pct":compounded(ov,3,100),"overnight_5d_pct":compounded(ov,5,100),"overnight_10d_pct":compounded(ov,10,100),
                "volume_ratio_5d":ratio(vv,5,volume),"volume_ratio_10d":ratio(vv,10,volume),"volume_ratio_20d":ratio(vv,20,volume),
                "amount_20d":mean(av,20),"amount_ratio_5d":ratio(av,5,amount),"amount_ratio_20d":ratio(av,20,amount),
                "volatility_5d_pct":vol[5],"volatility_10d_pct":vol[10],"volatility_20d_pct":vol[20],"downside_volatility_10d_pct":downside,
                "close_strength":strength,"intraday_return_pct":rel(close,op),"limit_up_close_flag":limit_up,
                "limit_up_5d_count":float(sum(lu[-5:])),"limit_up_10d_count":float(sum(lu[-10:])),
                "close_vs_ma5_pct":rel(close,ma[5]),"close_vs_ma10_pct":rel(close,ma[10]),"close_vs_ma20_pct":rel(close,ma[20]),"close_vs_ma60_pct":rel(close,ma[60]),
                "ma5_vs_ma10_pct":pairs[(5,10)],"ma5_vs_ma20_pct":pairs[(5,20)],"ma10_vs_ma20_pct":pairs[(10,20)],"ma20_vs_ma60_pct":pairs[(20,60)],
                "ma5_slope_5d_pct":rel(ma[5],mean(cv[-10:-5],5)),"ma20_slope_5d_pct":rel(ma[20],mean(cv[-25:-5],20)),
                "breakout_20d_pct":breakout20,"breakout_60d_pct":breakout60,"distance_from_high_20d_pct":distance_high,"distance_from_low_20d_pct":distance_low,
                "body_pct":rel(close,op),"upper_shadow_pct":(high-max(op,close))/close*100 if np.isfinite(op) and np.isfinite(close) and np.isfinite(high) and close>0 else np.nan,
                "lower_shadow_pct":(min(op,close)-low)/close*100 if np.isfinite(op) and np.isfinite(close) and np.isfinite(low) and close>0 else np.nan,
                "range_pct":(high-low)/close*100 if np.isfinite(high) and np.isfinite(low) and close>0 else np.nan,
                "drawdown_20d_pct":dd20,"rsi_6":calc_rsi(6),"rsi_14":calc_rsi(14),"bollinger_position_20d":bbpos,"bollinger_width_20d_pct":bbwidth,
                "atr_14_pct":atr,"macd_line_pct":macd1,"macd_signal_pct":macd2,"macd_histogram_pct":macd3,"kdj_k_9":k,"kdj_d_9":d,"kdj_j_9":j,
                "is_paused":float(r.is_paused) if pd.notna(r.is_paused) else 0.0,"is_st":float(r.is_st) if pd.notna(r.is_st) else 0.0,
            })
        frame=pd.DataFrame(rows)
        if frame.empty: return frame
        names={}
        universe_path=ROOT/"data"/"universe.json"
        if universe_path.exists():
            try:
                universe=json.loads(universe_path.read_text(encoding="utf-8"))
                names.update({str(row.get("symbol","")).zfill(6):str(row.get("name","")).strip() for row in universe if row.get("symbol") and str(row.get("name","")).strip()})
            except (OSError,json.JSONDecodeError): pass
        if "name" in day.columns: names.update(dict(zip(day["symbol"].astype(str).str.zfill(6),day["name"].astype(str))))
        frame["name"]=frame["symbol"].map(names).fillna(frame["symbol"])
        eligible=(~frame["name"].str.contains(r"ST|退",case=False,na=False)&frame["is_st"].eq(0)&frame["is_paused"].eq(0)&frame["close"].gt(2)&frame["amount"].ge(3e7))
        market=frame.loc[eligible]
        frame["market_breadth_pct"]=float((market["change_pct"]>0).mean()*100) if not market.empty else 0.0
        frame["market_median_return_pct"]=float(market["change_pct"].median()) if not market.empty else 0.0
        frame["market_return_dispersion_pct"]=float(market["change_pct"].std(ddof=1)) if len(market)>1 else 0.0
        a20=market["close_vs_ma20_pct"].dropna(); a60=market["close_vs_ma60_pct"].dropna(); pos5=market["return_5d_pct"].dropna()
        frame["market_above_ma20_pct"]=float((a20>0).mean()*100) if not a20.empty else 0.0
        frame["market_above_ma60_pct"]=float((a60>0).mean()*100) if not a60.empty else 0.0
        frame["market_positive_5d_pct"]=float((pos5>0).mean()*100) if not pos5.empty else 0.0
        usable=eligible&frame["return_10d_pct"].notna()&frame["volume_ratio_5d"].notna()&frame["volatility_10d_pct"].notna()
        return frame.loc[usable].copy()



def managed_trade(symbol: str, future_days: list[pd.DataFrame], round_trip_cost_bps: float = ROUND_TRIP_COST_BPS):
    if not future_days: return None
    first = future_days[0]
    row = first.loc[first["symbol"].eq(symbol)]
    if row.empty or pd.isna(row.iloc[0].get("open")): return None
    entry = float(row.iloc[0]["open"])
    if entry <= 0 or not np.isfinite(entry): return None
    high_limit = row.iloc[0].get("high_limit", np.nan)
    if ENTRY_LIMIT_UP_BLOCK and pd.notna(high_limit):
        high_limit = float(high_limit)
        if np.isfinite(high_limit) and entry >= high_limit * (1.0 - 1e-6):
            return None
    target_gross_pct = WIN_THRESHOLD_PCT + round_trip_cost_bps / 100.0
    stop = entry*(1-STOP_PCT/100.0); target = entry*(1+target_gross_pct/100.0)

    for day_no, day in enumerate(future_days[:MAX_HOLD], 1):
        if day_no < MIN_EXIT_DAY:
            continue
        rr = day.loc[day["symbol"].eq(symbol)]
        if rr.empty: continue
        r = rr.iloc[0]
        op = float(r.open) if pd.notna(r.open) else np.nan
        hi = float(r.high) if pd.notna(r.high) else np.nan
        lo = float(r.low) if pd.notna(r.low) else np.nan
        cl = float(r.close) if pd.notna(r.close) else np.nan
        # T+1 是建仓日，项目硬规则要求 T+2 才允许退出。
        # 因此即使 T+1 出现止损/止盈价，也不能在研究中提前平仓。
        if day_no < MIN_EXIT_DAY:
            continue
        if np.isfinite(lo) and lo <= stop:
            return {"gross_return_pct":(stop/entry-1)*100,"holding_days":day_no,"exit_reason":"stop"}
        if np.isfinite(hi) and hi >= target:
            return {"gross_return_pct":(target/entry-1)*100,"holding_days":day_no,"exit_reason":"target"}
        if day_no == MAX_HOLD and np.isfinite(cl):
            return {"gross_return_pct":(cl/entry-1)*100,"holding_days":day_no,"exit_reason":"max_hold"}
    return None


def _state_to_jsonable(state):
    names=("returns","volumes","amounts","closes","highs","lows","overnights","limitups")
    payload={name:{symbol:list(values) for symbol,values in getattr(state,name).items()} for name in names}
    for name in ("ema12","ema26","macd_signal","kdj_k","kdj_d"):
        payload[name]=dict(getattr(state,name))
    payload["feature_schema_version"]=FEATURE_SCHEMA_VERSION
    return payload


def _state_from_jsonable(state,payload):
    if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("检查点特征版本不匹配")
    names=("returns","volumes","amounts","closes","highs","lows","overnights","limitups")
    for name in names:
        target=getattr(state,name); target.clear()
        for symbol,values in payload.get(name,{}).items():
            target[symbol]=deque(values,maxlen=FEATURE_WARMUP_SESSIONS)
    for name in ("ema12","ema26","macd_signal","kdj_k","kdj_d"):
        target=getattr(state,name); target.clear(); target.update(payload.get(name,{}))



def _log_event(path: Path, event: str, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": pd.Timestamp.utcnow().isoformat(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[进度] {event} {json.dumps(fields, ensure_ascii=False)}", flush=True)


def _save_checkpoint(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)


def _load_checkpoint(path: Path):
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _checkpoint_payload(next_i, state, trade_rows, forward_rows, daily, candidate_days, args, version, commit, status="running"):
    return {
        "schema_version": 2, "feature_schema_version": FEATURE_SCHEMA_VERSION, "status": status,
        "start": args.start, "end": args.end,
        "win_definition": "net_profit_at_least_1pct",
        "target_gross_pct": TARGET_PCT,
        "stop_loss_gross_pct": STOP_PCT,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "cost_bps": args.cost_bps, "slippage_bps": args.slippage_bps,
        "strategy_version": version, "strategy_commit": commit,
        "next_active_index": next_i, "state": _state_to_jsonable(state),
        "trade_rows": trade_rows,
        "forward_rows": {str(k):v for k,v in forward_rows.items()},
        "daily": {str(k):v for k,v in daily.items()},
        "candidate_days": candidate_days,
    }


def stats(rows, cost_bps, slippage_bps):
    if not rows:
        return {"samples":0,"win_rate_pct":None,"mean_return_pct":None,"median_return_pct":None,"total_return_pct":None,"max_drawdown_pct":None}
    df = pd.DataFrame(rows)
    gross = pd.to_numeric(df["gross_return_pct"], errors="coerce")
    net = gross - 2*(cost_bps+slippage_bps)/100.0
    eq = (1+net/100).cumprod()
    dd = eq/eq.cummax()-1
    return {
        "samples":int(len(net)),
        "win_rate_pct":float((net>=WIN_THRESHOLD_PCT).mean()*100),
        "positive_rate_pct":float((net>0).mean()*100),
        "threshold_win_rate_pct":float((net>=WIN_THRESHOLD_PCT).mean()*100),
        "win_threshold_pct":float(WIN_THRESHOLD_PCT),
        "mean_return_pct":float(net.mean()),
        "median_return_pct":float(net.median()),
        "total_return_pct":float((eq.iloc[-1]-1)*100),
        "max_drawdown_pct":float(dd.min()*100),
        "mean_holding_days":float(pd.to_numeric(df["holding_days"]).mean()),
    }


def run(args):
    model, version, commit = load_strategy()
    files = history_files()
    dates = [p.name[:10] for p in files]
    if args.start not in dates or args.end not in dates: raise ValueError("research dates must be trading dates")
    start_i, end_i = dates.index(args.start), dates.index(args.end)
    # The research end is a data/signal cutoff. Complete performance metrics
    # are calculated only for signal dates with a full five-session future window.
    begin = max(0, start_i-FEATURE_WARMUP_SESSIONS)
    active = files[begin:end_i+1]

    cache = {}
    def get(i):
        if i not in cache: cache[i]=read_daily(active[i])
        return cache[i]

    state=FeatureState()
    trade_rows=[]
    forward_rows={1:[],3:[],5:[]}
    daily={1:[],2:[],3:[]}
    candidate_days=0
    checkpoint_path = Path(args.checkpoint)
    progress_path = Path(args.progress_log)
    checkpoint = _load_checkpoint(checkpoint_path)
    resume_i = 0
    if checkpoint:
        expected = {"start":args.start,"end":args.end,"cost_bps":args.cost_bps,
                    "slippage_bps":args.slippage_bps,"strategy_version":version,
                    "strategy_commit":commit,"feature_schema_version":FEATURE_SCHEMA_VERSION}
        mismatches = {k:(checkpoint.get(k),v) for k,v in expected.items() if checkpoint.get(k) != v}
        if mismatches:
            _log_event(progress_path, "checkpoint_incompatible_restart", mismatches=mismatches, restart_index=0)
            checkpoint = None
    if checkpoint:
        _state_from_jsonable(state, checkpoint["state"])
        trade_rows = checkpoint["trade_rows"]
        forward_rows = {int(k):v for k,v in checkpoint["forward_rows"].items()}
        daily = {int(k):v for k,v in checkpoint["daily"].items()}
        candidate_days = int(checkpoint["candidate_days"])
        resume_i = int(checkpoint["next_active_index"])
        _log_event(progress_path, "retry_resume", resume_index=resume_i,
                   resume_signal_date=active[resume_i].name[:10] if resume_i < len(active) else None)
    else:
        _log_event(progress_path, "research_started", start=args.start, end=args.end,
                   strategy_version=version, strategy_commit=commit, total_active_days=len(active))
    started = time.time()
    for i in range(resume_i, len(active)):
        signal_date=active[i].name[:10]
        _log_event(progress_path, "signal_date_started", signal_date=signal_date,
                   active_index=i, total_active_days=len(active))
        frame=state.build(get(i))
        if signal_date < args.start or signal_date > args.end: continue
        # The latest signal dates may be censored because future sessions are
        # not yet in the database. Exclude them from historical performance
        # metrics rather than inventing future prices. Candidate generation
        # separately uses the latest available close, so a 2026-09-30 cutoff
        # can produce a valid post-holiday candidate snapshot.
        if i + MAX_HOLD >= len(active):
            _save_checkpoint(checkpoint_path, _checkpoint_payload(i+1,state,trade_rows,forward_rows,daily,candidate_days,args,version,commit))
            _log_event(progress_path, "signal_date_completed", signal_date=signal_date, active_index=i,
                       candidate_days=candidate_days, progress_pct=round((i+1)/len(active)*100.0,2),
                       performance_window_complete=False)
            continue
        scored=model.score_universe(frame) if not frame.empty else frame
        selected=getattr(model,"admit_candidates")(scored) if not frame.empty else frame
        if selected.empty:
            for n in (1,2,3): daily[n].append({"date":signal_date,"basket_return_pct":0.0,"candidate_count":0})
            _save_checkpoint(checkpoint_path, _checkpoint_payload(i+1,state,trade_rows,forward_rows,daily,candidate_days,args,version,commit))
            _log_event(progress_path, "signal_date_completed", signal_date=signal_date, active_index=i,
                       candidate_days=candidate_days, progress_pct=round((i+1)/len(active)*100.0,2),
                       candidate_count=0)
            continue
        candidate_days += 1
        futures=[get(i+j) for j in range(1,MAX_HOLD+1)]
        by_symbol={}
        for row in selected.itertuples(index=False):
            symbol=str(row.symbol).zfill(6)
            entry_row=futures[0].loc[futures[0]["symbol"].eq(symbol)]
            if not entry_row.empty and pd.notna(entry_row.iloc[0].get("open")):
                entry=float(entry_row.iloc[0]["open"])
                if np.isfinite(entry) and entry > 0:
                    for horizon in (1,3,5):
                        idx=horizon-1
                        if idx < len(futures):
                            close_value=futures[idx].loc[futures[idx]["symbol"].eq(symbol)]
                            if not close_value.empty and pd.notna(close_value.iloc[0].get("close")):
                                forward_rows[horizon].append((float(close_value.iloc[0]["close"])/entry-1.0)*100.0)
            tr=managed_trade(symbol,futures,round_trip_cost_bps=2*(args.cost_bps+args.slippage_bps))
            if tr:
                tr.update({"signal_date":signal_date,"symbol":symbol,"score":float(row.score)})
                trade_rows.append(tr); by_symbol[symbol]=tr
        for n in (1,2,3):
            syms=selected.head(n)["symbol"].astype(str).str.zfill(6).tolist()
            trs=[by_symbol[s] for s in syms if s in by_symbol]
            gross=float(np.mean([x["gross_return_pct"] for x in trs])) if trs else 0.0
            net=gross-2*(args.cost_bps+args.slippage_bps)/100.0 if trs else 0.0
            daily[n].append({"date":signal_date,"basket_return_pct":net,"candidate_count":len(syms),"executed_count":len(trs)})
        _save_checkpoint(checkpoint_path, _checkpoint_payload(i+1,state,trade_rows,forward_rows,daily,candidate_days,args,version,commit))
        _log_event(progress_path, "signal_date_completed", signal_date=signal_date, active_index=i,
                   candidate_days=candidate_days, progress_pct=round((i+1)/len(active)*100.0,2),
                   elapsed_seconds=round(time.time()-started,2))

    forward_signal_diagnostics={}
    for horizon, values in forward_rows.items():
        arr=np.asarray(values,dtype=float)
        forward_signal_diagnostics[str(horizon)+"d"] = {
            "samples":int(len(arr)),
            "mean_return_pct":float(np.mean(arr)) if len(arr) else None,
            "median_return_pct":float(np.median(arr)) if len(arr) else None,
            "positive_rate_pct":float((arr>0).mean()*100.0) if len(arr) else None,
            "p25_return_pct":float(np.quantile(arr,0.25)) if len(arr) else None,
            "p75_return_pct":float(np.quantile(arr,0.75)) if len(arr) else None,
        }

    sensitivity=[]
    for n in (1,2,3):
        sensitivity.append({
            "top_n":n,
            **stats([{"gross_return_pct":x["basket_return_pct"],"holding_days":1} for x in daily[n] if x["candidate_count"]>0],0,0),
            "strategy_version":version,"strategy_commit":commit,"future_function":False
        })

    result={
        "schema_version":2,"status":"ready","method":"short_term_signal_research_1_5_session",
        "start":args.start,"end":args.end,"strategy_source":"Aquant-Private/main",
        "cutoff_semantics":"signal_data_cutoff; performance metrics exclude signal dates without a complete five-session future window",
        "strategy_version":version,"strategy_commit":commit,"future_function":False,
        "signal_days":len(daily[3]),"candidate_days":candidate_days,
        "candidate_day_rate_pct":candidate_days/len(daily[3])*100 if daily[3] else 0,
        "forward_signal_diagnostics":forward_signal_diagnostics,
        "trade_performance":stats(trade_rows,args.cost_bps,args.slippage_bps),
        "candidate_count_sensitivity":sensitivity,
        "exit_distribution":pd.Series([x["exit_reason"] for x in trade_rows]).value_counts().to_dict() if trade_rows else {},
        "audit":{
            "signal_uses_only_T_close_information":True,
            "entry_uses_T_plus_1_open":True,
            "entry_limit_up_block":ENTRY_LIMIT_UP_BLOCK,
            "earliest_exit_is_T_plus_2":True,
            "future_function":False,
            "max_holding_sessions":MAX_HOLDING_SESSIONS,
            "earliest_exit_day_after_entry":MIN_EXIT_DAY,
            "stop_loss_pct":STOP_PCT,
            "target_return_pct":TARGET_PCT,
            "win_threshold_pct":WIN_THRESHOLD_PCT,
            "both_stop_and_target_same_day":"stop_first_conservative_assumption",
            "round_trip_cost_bps":2*(args.cost_bps+args.slippage_bps),
            "overlapping_signal_samples":True,
            "interpretation":"signal-quality research sample, not broker-connected capital simulation"
        }
    }
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    sens={"schema_version":1,"status":"ready","method":"short_term_candidate_count_sensitivity",
          "start":args.start,"end":args.end,"results":sensitivity,"strategy_source":"Aquant-Private/main",
          "strategy_version":version,"strategy_commit":commit,"future_function":False}
    (OUT_DIR/"short_term_sensitivity.json").write_text(json.dumps(sens,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    exits={"schema_version":1,"status":"ready","method":"short_term_exit_diagnostics",
           "start":args.start,"end":args.end,"exit_distribution":result["exit_distribution"],
           "trade_performance":result["trade_performance"],"strategy_source":"Aquant-Private/main",
           "strategy_version":version,"strategy_commit":commit,"future_function":False,"audit":result["audit"]}
    (OUT_DIR/"short_term_exit_diagnostics.json").write_text(json.dumps(exits,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    _save_checkpoint(checkpoint_path, _checkpoint_payload(len(active),state,trade_rows,forward_rows,daily,candidate_days,args,version,commit,status="completed"))
    _log_event(progress_path, "research_completed", signal_days=result["signal_days"], candidate_days=candidate_days,
               elapsed_seconds=round(time.time()-started,2))
    print(json.dumps({"status":"ready","strategy_version":version,"strategy_commit":commit,"signal_days":result["signal_days"],"candidate_days":candidate_days},ensure_ascii=False))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",required=True); ap.add_argument("--end",required=True)
    ap.add_argument("--cost-bps",type=float,default=3.0); ap.add_argument("--slippage-bps",type=float,default=2.0)
    ap.add_argument("--output",default=str(OUT_DIR/"short_term_latest.json"))
    ap.add_argument("--checkpoint",default="runtime/research_checkpoint.json.gz")
    ap.add_argument("--progress-log",default="runtime/progress.jsonl")
    parsed=ap.parse_args()
    try:
        run(parsed)
    except Exception as exc:
        _log_event(Path(parsed.progress_log), "research_failed", error_type=type(exc).__name__,
                   error=str(exc), traceback=traceback.format_exc(limit=8))
        raise
