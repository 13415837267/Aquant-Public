from scripts.selection_factor_catalog import (
    ALL_MARKET_FACTORS, ALL_STOCK_FACTORS, active_market_factors,
    active_stock_factors, load_research_config,
)

def test_default_configuration_enables_the_factor_directory():
    config=load_research_config()
    assert active_stock_factors(config)==list(ALL_STOCK_FACTORS)
    assert active_market_factors(config)==list(ALL_MARKET_FACTORS)
    assert config["正式模型自动替换"] is False

def test_threshold_and_exit_grids_are_editable():
    config=load_research_config()
    assert config["分位阈值"] and config["市场收益离散度阈值"]
    assert config["短线模型概率阈值网格"] and config["高收益模型概率阈值网格"]
    assert config["复利入场概率阈值网格"]
    assert config["复利止盈目标网格百分比"] and config["复利止损幅度网格百分比"]
