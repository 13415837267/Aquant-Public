import numpy as np
import pandas as pd

from scripts.short_term_research import executable_entry_mask, managed_trade
from scripts.train_path_aware_model import path_targets
from scripts.train_short_term_model import build_targets


def _day(symbol, open_, high, low, close, high_limit=20.0):
    return pd.DataFrame([{
        "symbol": symbol,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "high_limit": high_limit,
    }])


def test_t1_position_cannot_exit_on_entry_session():
    future = [
        _day("600000", 10.0, 10.6, 9.8, 10.5),
        _day("600000", 10.5, 10.8, 10.4, 10.7),
        _day("600000", 10.7, 10.8, 10.6, 10.75),
    ]
    result = managed_trade("600000", future)
    assert result is not None
    assert result["holding_days"] >= 2


def test_entry_at_upper_limit_is_not_executable():
    future = [
        _day("600000", 20.0, 20.0, 19.5, 20.0, high_limit=20.0),
        _day("600000", 19.8, 20.2, 19.7, 20.0, high_limit=22.0),
        _day("600000", 20.0, 20.5, 19.9, 20.4, high_limit=22.0),
    ]
    assert managed_trade("600000", future) is None


def test_same_day_stop_takes_priority_over_target():
    future = [
        _day("600000", 10.0, 10.1, 9.9, 10.0),
        _day("600000", 10.0, 10.2, 9.6, 10.1),
        _day("600000", 10.1, 10.2, 10.0, 10.1),
    ]
    result = managed_trade("600000", future)
    assert result is not None
    assert result["exit_reason"] == "stop"
    assert result["holding_days"] == 2



def _five_future_days(entry_open=10.0, entry_high_limit=11.0):
    days = [_day("600000", entry_open, entry_open * 1.01, entry_open * 0.99,
                 entry_open, high_limit=entry_high_limit)]
    for _ in range(4):
        days.append(_day("600000", 10.0, 10.5, 10.0, 10.3, high_limit=11.0))
    return days


def test_one_percent_labels_exclude_non_executable_limit_up_entry():
    labels, _, _, complete = build_targets(["600000"], _five_future_days(20.0, 20.0))
    assert complete.tolist() == [False]
    assert labels.tolist() == [False]


def test_three_percent_labels_exclude_non_executable_limit_up_entry():
    labels, _, _, complete = path_targets(["600000"], _five_future_days(20.0, 20.0))
    assert complete.tolist() == [False]
    assert labels.tolist() == [False]


def test_executable_entry_remains_in_both_label_sets():
    futures = _five_future_days(10.0, 11.0)
    labels_1pct, _, _, complete_1pct = build_targets(["600000"], futures)
    labels_3pct, _, _, complete_3pct = path_targets(["600000"], futures)

    assert complete_1pct.tolist() == [True]
    assert complete_3pct.tolist() == [True]
    assert labels_1pct.tolist() == [True]
    assert labels_3pct.tolist() == [True]



def test_executable_entry_mask_matches_each_symbol_independently():
    entry_day = pd.DataFrame([
        {"symbol": "600000", "open": 20.0, "high_limit": 20.0},
        {"symbol": "000001", "open": 10.0, "high_limit": 11.0},
    ])

    assert executable_entry_mask(["600000", "000001"], entry_day).tolist() == [False, True]



def test_label_window_must_end_strictly_before_next_split():
    from scripts.short_term_research import label_window_precedes_boundary

    boundary = 100
    horizon = 5
    assert label_window_precedes_boundary(94, horizon, boundary)
    assert not label_window_precedes_boundary(95, horizon, boundary)
    assert not label_window_precedes_boundary(99, horizon, boundary)



def test_pairwise_ranker_learns_positive_over_negative_ordering():
    from scripts.high_precision_profit_mining import PairwiseRankingModel

    x = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    y = np.asarray([1.0, 0.0], dtype=np.float64)
    model = PairwiseRankingModel(
        n_features=2, learning_rate=0.1, l2=0.0,
        max_positives_per_day=1, max_negatives_per_day=1, seed=7
    )

    before = model.predict(x)[0] - model.predict(x)[1]
    pairs = model.update(x, y)
    after = model.predict(x)[0] - model.predict(x)[1]

    assert pairs == 1
    assert model.pair_updates == 1
    assert model.pairs_seen == 1
    assert after > before


def test_pairwise_ranker_skips_single_class_days():
    from scripts.high_precision_profit_mining import PairwiseRankingModel

    model = PairwiseRankingModel(n_features=2, seed=7)
    pairs = model.update(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64)
    )

    assert pairs == 0
    assert model.pair_updates == 0
    assert model.pairs_seen == 0


def test_managed_trade_rejects_paused_entry():
    future = [
        pd.DataFrame([{
            "symbol": "600000", "open": 10.0, "high": 10.1,
            "low": 9.9, "close": 10.0, "high_limit": 11.0, "is_paused": 1
        }]),
        _day("600000", 10.0, 10.2, 10.0, 10.1, high_limit=11.0),
        _day("600000", 10.1, 10.2, 10.0, 10.1, high_limit=11.0),
    ]
    assert managed_trade("600000", future) is None


def test_executable_entry_mask_rejects_paused_symbol_only():
    entry_day = pd.DataFrame([
        {"symbol": "600000", "open": 10.0, "high_limit": 11.0, "is_paused": 1},
        {"symbol": "000001", "open": 10.0, "high_limit": 11.0, "is_paused": 0},
    ])
    assert executable_entry_mask(["600000", "000001"], entry_day).tolist() == [False, True]



def test_pairwise_ranker_learns_positive_over_negative_ordering():
    from scripts.high_precision_profit_mining import PairwiseRankingModel

    x = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    y = np.asarray([1.0, 0.0], dtype=np.float64)
    model = PairwiseRankingModel(
        n_features=2, learning_rate=0.1, l2=0.0,
        max_positives_per_day=1, max_negatives_per_day=1, seed=7
    )

    before = model.predict(x)[0] - model.predict(x)[1]
    pairs = model.update(x, y)
    after = model.predict(x)[0] - model.predict(x)[1]

    assert pairs == 1
    assert model.pair_updates == 1
    assert model.pairs_seen == 1
    assert after > before


def test_pairwise_ranker_skips_single_class_days():
    from scripts.high_precision_profit_mining import PairwiseRankingModel

    model = PairwiseRankingModel(n_features=2, seed=7)
    pairs = model.update(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64)
    )

    assert pairs == 0
    assert model.pair_updates == 0
    assert model.pairs_seen == 0


def test_managed_trade_rejects_paused_entry():
    future = [
        pd.DataFrame([{
            "symbol": "600000", "open": 10.0, "high": 10.1,
            "low": 9.9, "close": 10.0, "high_limit": 11.0, "is_paused": 1
        }]),
        _day("600000", 10.0, 10.2, 10.0, 10.1, high_limit=11.0),
        _day("600000", 10.1, 10.2, 10.0, 10.1, high_limit=11.0),
    ]
    assert managed_trade("600000", future) is None


def test_executable_entry_mask_rejects_paused_symbol_only():
    entry_day = pd.DataFrame([
        {"symbol": "600000", "open": 10.0, "high_limit": 11.0, "is_paused": 1},
        {"symbol": "000001", "open": 10.0, "high_limit": 11.0, "is_paused": 0},
    ])
    assert executable_entry_mask(["600000", "000001"], entry_day).tolist() == [False, True]



def test_executable_entry_mask_rejects_missing_high_limit_when_required():
    entry_day = pd.DataFrame([{"symbol": "600000", "open": 10.0, "is_paused": 0}])
    assert executable_entry_mask(["600000"], entry_day).tolist() == [False]


def test_managed_trade_rejects_missing_high_limit_when_required():
    future = [
        pd.DataFrame([{
            "symbol": "600000", "open": 10.0, "high": 10.1,
            "low": 9.9, "close": 10.0, "is_paused": 0
        }]),
        _day("600000", 10.0, 10.2, 10.0, 10.1, high_limit=11.0),
        _day("600000", 10.1, 10.2, 10.0, 10.1, high_limit=11.0),
    ]
    assert managed_trade("600000", future) is None
