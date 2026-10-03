"""Short-term 1-5 session research for the production candidate strategy."""
from __future__ import annotations

import argparse, gzip, importlib, json, os, subprocess, sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = ROOT / "data" / "history"
OUT_DIR = ROOT / "data" / "backtest"
MAX_HOLD = 6
MIN_EXIT_DAY = 2
MAX_HOLDING_SESSIONS = 5
ENTRY_LIMIT_UP_BLOCK = True
TARGET_PCT = 6.0
STOP_PCT = 3.0


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
    def __init__(self):
        self.returns = {}
        self.volumes = {}
        self.amounts = {}

    def build(self, day: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for r in day.itertuples(index=False):
            s = str(r.symbol).zfill(6)
            ret = float(r.pct_chg)/100.0 if pd.notna(r.pct_chg) else np.nan
            vol = float(r.volume) if pd.notna(r.volume) else np.nan
            amount = float(r.amount) if pd.notna(r.amount) else np.nan
            rr = self.returns.setdefault(s, deque(maxlen=20))
            vv = self.volumes.setdefault(s, deque(maxlen=20))
            aa = self.amounts.setdefault(s, deque(maxlen=20))
            rr.append(ret); vv.append(vol); aa.append(amount)

            def compound(n):
                vals = list(rr)[-n:]
                return ((np.prod(1.0 + np.asarray(vals)) - 1.0) * 100.0
                        if len(vals) == n and all(np.isfinite(vals)) else np.nan)

            prior_v = list(vv)[:-1]
            ratio = np.nan
            if len(prior_v) >= 5 and np.isfinite(vol):
                base = float(np.nanmean(prior_v[-5:]))
                if base > 0: ratio = vol / base

            vals10 = list(rr)[-10:]
            vol10 = float(np.std(vals10, ddof=1)*100.0) if len(vals10) == 10 and all(np.isfinite(vals10)) else np.nan
            hi = float(r.high) if pd.notna(r.high) else np.nan
            lo = float(r.low) if pd.notna(r.low) else np.nan
            cl = float(r.close) if pd.notna(r.close) else np.nan
            width = hi - lo
            strength = (cl-lo)/width if np.isfinite(width) and width > 0 else 0.5

            rows.append({
                "symbol": s, "close": cl, "high": hi, "low": lo, "amount": amount,
                "turnover_pct": float(r.turnover_pct) if pd.notna(r.turnover_pct) else np.nan,
                "change_pct": float(r.pct_chg) if pd.notna(r.pct_chg) else np.nan,
                "return_1d_pct": compound(1), "return_3d_pct": compound(3),
                "return_5d_pct": compound(5), "return_10d_pct": compound(10),
                "return_20d_pct": compound(20), "volume_ratio_5d": ratio,
                "amount_20d": float(np.nanmean(list(aa))) if len(aa) == 20 else np.nan,
                "volatility_10d_pct": vol10, "close_strength": strength,
                "is_paused": float(r.is_paused) if pd.notna(r.is_paused) else 0.0,
                "is_st": float(r.is_st) if pd.notna(r.is_st) else 0.0,
            })
        frame = pd.DataFrame(rows)
        if frame.empty: return frame
        names = {}
        if "name" in day.columns:
            names = dict(zip(day["symbol"].astype(str).str.zfill(6), day["name"].astype(str)))
        frame["name"] = frame["symbol"].map(names).fillna(frame["symbol"])
        eligible = (
            ~frame["name"].str.contains(r"ST|退", case=False, na=False)
            & frame["is_st"].eq(0) & frame["is_paused"].eq(0)
            & frame["close"].gt(2) & frame["amount"].ge(3e7)
        )
        market = frame.loc[eligible]
        breadth = float((market["change_pct"] > 0).mean()*100.0) if not market.empty else 0.0
        median_ret = float(market["change_pct"].median()) if not market.empty else 0.0
        frame["market_breadth_pct"] = breadth
        frame["market_median_return_pct"] = median_ret
        usable = eligible & frame["return_10d_pct"].notna() & frame["volume_ratio_5d"].notna() & frame["volatility_10d_pct"].notna()
        return frame.loc[usable].copy()


def managed_trade(symbol: str, future_days: list[pd.DataFrame]):
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
    stop = entry*(1-STOP_PCT/100.0); target = entry*(1+TARGET_PCT/100.0)

    for day_no, day in enumerate(future_days[:MAX_HOLD], 1):
        if day_no < MIN_EXIT_DAY:
            continue;
        rr = day.loc[day["symbol"].eq(symbol)]
        if rr.empty: continue
        r = rr.iloc[0]
        op = float(r.open) if pd.notna(r.open) else np.nan
        hi = float(r.high) if pd.notna(r.high) else np.nan
        lo = float(r.low) if pd.notna(r.low) else np.nan
        cl = float(r.close) if pd.notna(r.close) else np.nan
        if day_no == 1 and np.isfinite(op):
            if op <= stop: return {"gross_return_pct":(op/entry-1)*100,"holding_days":1,"exit_reason":"stop_gap"}
            if op >= target: return {"gross_return_pct":(op/entry-1)*100,"holding_days":1,"exit_reason":"target_gap"}
        if np.isfinite(lo) and lo <= stop:
            return {"gross_return_pct":(stop/entry-1)*100,"holding_days":day_no,"exit_reason":"stop"}
        if np.isfinite(hi) and hi >= target:
            return {"gross_return_pct":(target/entry-1)*100,"holding_days":day_no,"exit_reason":"target"}
        if day_no == MAX_HOLD and np.isfinite(cl):
            return {"gross_return_pct":(cl/entry-1)*100,"holding_days":day_no,"exit_reason":"max_hold"}
    return None


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
        "win_rate_pct":float((net>0).mean()*100),
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
    if end_i + MAX_HOLD >= len(files): raise ValueError("research end must leave six future trading sessions for a five-session T+1 holding window")
    begin = max(0, start_i-20)
    active = files[begin:end_i+MAX_HOLD+1]

    cache = {}
    def get(i):
        if i not in cache: cache[i]=read_daily(active[i])
        return cache[i]

    state=FeatureState()
    trade_rows=[]
    forward_rows={1:[],3:[],5:[]}
    daily={1:[],2:[],3:[]}
    candidate_days=0
    for i in range(len(active)-MAX_HOLD):
        signal_date=active[i].name[:10]
        frame=state.build(get(i))
        if signal_date < args.start or signal_date > args.end: continue
        scored=model.score_universe(frame) if not frame.empty else frame
        selected=getattr(model,"admit_candidates")(scored) if not frame.empty else frame
        if selected.empty:
            for n in (1,2,3): daily[n].append({"date":signal_date,"basket_return_pct":0.0,"candidate_count":0})
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
            tr=managed_trade(symbol,futures)
            if tr:
                tr.update({"signal_date":signal_date,"symbol":symbol,"score":float(row.score)})
                trade_rows.append(tr); by_symbol[symbol]=tr
        for n in (1,2,3):
            syms=selected.head(n)["symbol"].astype(str).str.zfill(6).tolist()
            trs=[by_symbol[s] for s in syms if s in by_symbol]
            gross=float(np.mean([x["gross_return_pct"] for x in trs])) if trs else 0.0
            net=gross-2*(args.cost_bps+args.slippage_bps)/100.0 if trs else 0.0
            daily[n].append({"date":signal_date,"basket_return_pct":net,"candidate_count":len(syms),"executed_count":len(trs)})

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
    print(json.dumps({"status":"ready","strategy_version":version,"strategy_commit":commit,"signal_days":result["signal_days"],"candidate_days":candidate_days},ensure_ascii=False))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--start",required=True); ap.add_argument("--end",required=True)
    ap.add_argument("--cost-bps",type=float,default=3.0); ap.add_argument("--slippage-bps",type=float,default=2.0)
    ap.add_argument("--output",default=str(OUT_DIR/"short_term_latest.json"))
    run(ap.parse_args())
