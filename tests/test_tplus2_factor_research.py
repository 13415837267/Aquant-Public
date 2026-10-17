import math
import numpy as np
import pandas as pd
from scripts.tplus2_factor_research import label_day, rank_frame, classify_net_return


def test_fixed_t1_open_buy_t2_open_sell_three_classes():
    result = label_day(
        np.array([10., 10., 10.], dtype=np.float32),
        np.array([10.2, 9.8, 10.01], dtype=np.float32),
        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.array([11., 11., 11.], dtype=np.float32),
        np.array([9., 9., 9.], dtype=np.float32), 10., .1)
    assert math.isclose(float(result.loc[0, "gross_return_pct"]), 2., rel_tol=1e-6)
    assert math.isclose(float(result.loc[0, "net_return_pct"]), 1.9, rel_tol=1e-6)
    assert result["result_class"].tolist() == ["盈利", "亏损", "平盘"]


def test_untradable_and_missing_prices_are_separated():
    result = label_day(
        np.array([11., 10., 10.], dtype=np.float32),
        np.array([11.2, 10.2, np.nan], dtype=np.float32),
        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.array([11., 11., 11.], dtype=np.float32),
        np.array([9., 9., 9.], dtype=np.float32), 10., .1)
    assert result.loc[0, "theoretical_result_class"] == "盈利"
    assert result.loc[0, "result_class"] == "不可执行"
    assert result.loc[0, "entry_status"] == "T+1开盘涨停，保守排除"
    assert result.loc[2, "result_class"] == "不可执行"
    assert result.loc[2, "label_status"] == "价格缺失"


def test_flat_band_boundaries():
    assert classify_net_return(.1, .1) == "平盘"
    assert classify_net_return(-.1, .1) == "平盘"
    assert classify_net_return(.10001, .1) == "盈利"
    assert classify_net_return(-.10001, .1) == "亏损"


def test_cross_sectional_ranks_and_market_scaling():
    frame = pd.DataFrame({
        "return_5d_pct": [1., 2., 3.],
        "market_breadth_pct": [60.] * 3,
        "market_median_return_pct": [.5] * 3,
    })
    config = {"市场因子缩放": {
        "market_breadth_pct": {"除数": 100, "偏移": 0, "最小值": 0, "最大值": 1},
        "market_median_return_pct": {"除数": 10, "偏移": .5, "最小值": 0, "最大值": 1},
    }}
    result = rank_frame(frame, list(frame.columns), config)
    assert np.allclose(result[:, 0], [1 / 3, 2 / 3, 1])
    assert np.allclose(result[:, 1], .6)
    assert np.allclose(result[:, 2], .55)
