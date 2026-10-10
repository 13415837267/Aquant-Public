import numpy as np
import pandas as pd

from scripts.short_term_research import FeatureState
from scripts.selection_factor_catalog import ALL_MARKET_FACTORS, ALL_STOCK_FACTORS


def _bar(index, close, previous_close):
    return {
        "symbol":"600000","open":close-0.04,"high":close+0.12,"low":close-0.10,"close":close,
        "volume":1000000.0+index*5000.0,"amount":100000000.0+index*1000000.0,
        "pct_chg":(close/previous_close-1.0)*100.0 if previous_close>0 else 0.0,
        "turnover_pct":1.0+index/100.0,"is_paused":0,"is_st":0,
        "high_limit":close*1.10,"low_limit":close*0.90,
    }


def test_feature_state_builds_all_configured_technical_and_market_factors():
    state=FeatureState()
    previous=10.0
    frame=pd.DataFrame()
    for index in range(70):
        close=10.0+index*0.04+(index%4)*0.015
        frame=state.build(pd.DataFrame([_bar(index,close,previous)]))
        previous=close
    assert not frame.empty
    for name in ALL_STOCK_FACTORS+ALL_MARKET_FACTORS:
        assert name in frame.columns, name
    row=frame.iloc[0]
    for name in ("close_vs_ma5_pct","breakout_20d_pct","breakout_60d_pct","volume_ratio_20d","rsi_6","rsi_14","bollinger_width_20d_pct","atr_14_pct","macd_histogram_pct","kdj_j_9"):
        assert np.isfinite(float(row[name])), name
    assert 0.0 <= float(row["rsi_14"]) <= 100.0
    assert float(row["bollinger_width_20d_pct"]) >= 0.0
    assert float(row["market_above_ma20_pct"]) == 100.0
    assert float(row["market_positive_5d_pct"]) == 100.0
