import pandas as pd

from scripts.train_short_term_model import build_targets
from scripts.train_path_aware_model import path_targets


def _days(highs):
    output=[]
    previous=10.0
    for idx,high in enumerate(highs):
        output.append(pd.DataFrame([{
            "symbol":"600000","open":10.0,"high":float(high),"low":9.9,"close":10.0,
            "high_limit":20.0,"is_paused":0
        }]))
    return output


def test_short_term_target_does_not_count_target_hit_on_t_plus_1():
    days=_days([10.5,10.05,10.06,10.07,10.08])
    labels,best,close,complete=build_targets(["600000"],days)
    assert bool(complete[0])
    assert not bool(labels[0])


def test_high_return_path_target_does_not_count_target_hit_on_t_plus_1():
    days=_days([11.0,10.1,10.12,10.15,10.2])
    labels,best,close,complete=path_targets(["600000"],days)
    assert bool(complete[0])
    assert not bool(labels[0])


def test_targets_can_be_hit_on_t_plus_2_and_later():
    one_percent_days=_days([10.5,10.2,10.2,10.2,10.2])
    one_labels,_,_,one_complete=build_targets(["600000"],one_percent_days)
    assert bool(one_complete[0])
    assert bool(one_labels[0])

    three_percent_days=_days([10.5,10.4,10.5,10.5,10.6])
    three_labels,_,_,three_complete=path_targets(["600000"],three_percent_days)
    assert bool(three_complete[0])
    assert bool(three_labels[0])
