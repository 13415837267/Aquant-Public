import pytest

from types import SimpleNamespace

from scripts.generate_research_candidates import observation_exclusion_reasons


@pytest.fixture
def precision():
    return SimpleNamespace(
        MIN_MARKET_BREADTH_PCT=65.0,
        MIN_MARKET_MEDIAN_RETURN_PCT=-0.30,
        MAX_VOLATILITY_10D_PCT=8.0,
        MIN_SINGLE_DAY_CHANGE_PCT=-7.0,
        MAX_RETURN_5D_PCT=15.0,
        MAX_LIMIT_UP_5D_COUNT=1.0,
        MAX_INTRADAY_RETURN_PCT=8.0,
        DAILY_TOP1_FALLBACK_PROBABILITY_THRESHOLD=0.85,
    )


def _row(**overrides):
    row = {
        "score": 91.0,
        "precision_probability": 0.91,
        "volatility_10d_pct": 4.0,
        "change_pct": 1.0,
        "return_5d_pct": 5.0,
        "limit_up_5d_count": 0.0,
        "intraday_return_pct": 2.0,
    }
    row.update(overrides)
    return row


def test_market_gate_explains_empty_pool_even_for_high_score(precision):
    reasons = observation_exclusion_reasons(
        _row(),
        precision,
        breadth=23.48,
        median_return=-1.609,
        risk_off=True,
    )

    assert any("上涨广度 23.48%" in reason and "65.00%" in reason for reason in reasons)
    assert any("市场中位收益 -1.61%" in reason and "-0.30%" in reason for reason in reasons)
    assert not any("模型评分" in reason or "精度概率不足" in reason for reason in reasons)


def test_individual_risk_and_minimum_admission_reasons_are_explained(precision):
    reasons = observation_exclusion_reasons(
        _row(
            score=80.0,
            precision_probability=0.80,
            volatility_10d_pct=9.2,
            change_pct=-7.5,
            return_5d_pct=16.5,
            limit_up_5d_count=2.0,
            intraday_return_pct=8.5,
        ),
        precision,
        breadth=70.0,
        median_return=0.2,
        risk_off=False,
    )

    assert any("10日波动率" in reason for reason in reasons)
    assert any("当日涨跌幅" in reason for reason in reasons)
    assert any("近5日涨幅" in reason for reason in reasons)
    assert any("近5日涨停次数" in reason for reason in reasons)
    assert any("日内涨幅" in reason for reason in reasons)
    assert any("模型评分" in reason for reason in reasons)
    assert any("精度概率不足" in reason for reason in reasons)
