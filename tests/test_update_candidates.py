import pandas as pd
from scripts.market_scope import is_main_board_symbol
from scripts.update_candidates import build_candidates

class FakeStrategy:
    WEIGHTS={"momentum_short":0.35,"volume_activity":0.25,"price_strength":0.15,"liquidity":0.15,"safety":0.10}
    @staticmethod
    def score_universe(frame):
        out=frame.copy(); out["score"]=out["return_5d_pct"]; return out.sort_values("score",ascending=False)
    @staticmethod
    def admit_candidates(frame): return frame.head(2).copy()

def _history():
    dates=pd.date_range("2026-03-02",periods=25,freq="B"); rows=[]
    for symbol,start,step in [("000001",10.0,0.15),("000002",10.0,0.0),("688001.SH",10.0,0.20),("300001.SZ",10.0,0.25),("920001.BJ",10.0,0.30)]:
        for i,dt in enumerate(dates):
            close=start+step*i
            rows.append({"symbol":symbol,"date":dt.strftime("%Y-%m-%d"),"open":close-0.05,"high":close+0.1,"low":close-0.1,"close":close,"volume":1_000_000,"amount":50_000_000,"pct_chg":0.5,"turnover_pct":2.0,"is_paused":0,"is_st":0,"high_limit":round((close-0.05)*1.1,2)})
    return pd.DataFrame(rows)

def test_main_board_scope():
    accepted=["000001.SZ","001200.SZ","002594.SZ","003816.SZ","004001.SZ","600000.SH","601398.SH","603019.SH","605499.SH"]
    rejected=["001001.SZ","300001.SZ","688001.SH","920001.BJ","900901.SH"]
    assert all(is_main_board_symbol(c) for c in accepted)
    assert not any(is_main_board_symbol(c) for c in rejected)

def test_build_candidates_uses_short_term_contract():
    snapshot=build_candidates(_history(),FakeStrategy,"test","abc123")
    assert {r["symbol"] for r in snapshot["candidates"]}=={"000001","000002"}
    assert snapshot["lookback_trading_days"]==20
    assert snapshot["signal_horizon"]=="T收盘信号 → T+1开盘买入 → T+2起最早卖出 → 最长5个交易日"
    assert snapshot["strategy_source"]=="Aquant-Private/main"
    assert snapshot["diagnostics"]["candidate_count"]==2
    assert snapshot["audit"]["short_term_features_only"] is True
    assert set(["overnight_1d_pct","overnight_3d_pct","overnight_5d_pct","intraday_return_pct","limit_up_5d_count"]).issubset(snapshot["candidates"][0])
