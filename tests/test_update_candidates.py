import pandas as pd

from scripts.update_candidates import build_candidates


def test_build_candidates_ranks_and_limits_rows():
    raw = pd.DataFrame([
        {"代码": "000001", "名称": "测试A", "最新价": 10, "涨跌幅": 1, "60日涨跌幅": 25, "换手率": 2.5, "市净率": 1.5, "市盈率-动态": 12, "成交额": 8e8, "最高": 10.5, "最低": 9.8, "昨收": 9.9, "量比": 1.8},
        {"代码": "000002", "名称": "测试B", "最新价": 9, "涨跌幅": -1, "60日涨跌幅": 5, "换手率": 1.0, "市净率": 3.0, "市盈率-动态": 30, "成交额": 3e8, "最高": 9.2, "最低": 8.9, "昨收": 9.1, "量比": 0.9},
        {"代码": "000003", "名称": "ST测试", "最新价": 12, "涨跌幅": 2, "60日涨跌幅": 50, "换手率": 8, "市净率": 1, "市盈率-动态": 8, "成交额": 9e8, "最高": 12.2, "最低": 11.8, "昨收": 11.9, "量比": 2.1},
    ])
    snap = build_candidates(raw, "2026-09-30T18:00:00+08:00")
    assert len(snap["candidates"]) == 2
    assert snap["candidates"][0]["symbol"] == "000001"
    assert 0 <= snap["candidates"][0]["score"] <= 100
