"""可配置选股因子目录与研究参数校验。仅供研究层使用。"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
CONFIG_PATH=ROOT/"config"/"选股条件研究配置.json"
BASE_STOCK_FACTORS=(
 "return_1d_pct","return_3d_pct","return_5d_pct","return_10d_pct","return_20d_pct",
 "overnight_1d_pct","overnight_3d_pct","overnight_5d_pct","overnight_10d_pct",
 "volume_ratio_5d","amount_20d","volatility_10d_pct","close_strength",
 "intraday_return_pct","limit_up_5d_count","turnover_pct","change_pct",
)
FACTOR_GROUPS={
 "趋势与突破":("close_vs_ma5_pct","close_vs_ma10_pct","close_vs_ma20_pct","close_vs_ma60_pct","ma5_vs_ma10_pct","ma5_vs_ma20_pct","ma10_vs_ma20_pct","ma20_vs_ma60_pct","ma5_slope_5d_pct","ma20_slope_5d_pct","return_60d_pct","breakout_20d_pct","breakout_60d_pct","distance_from_high_20d_pct","distance_from_low_20d_pct"),
 "成交量与流动性":("volume_ratio_10d","volume_ratio_20d","amount_ratio_5d","amount_ratio_20d"),
 "K线形态":("body_pct","upper_shadow_pct","lower_shadow_pct","range_pct"),
 "风险与波动":("volatility_5d_pct","volatility_20d_pct","downside_volatility_10d_pct","drawdown_20d_pct","bollinger_position_20d","bollinger_width_20d_pct","atr_14_pct"),
 "摆动指标":("rsi_6","rsi_14","macd_line_pct","macd_signal_pct","macd_histogram_pct","kdj_k_9","kdj_d_9","kdj_j_9"),
 "涨停行为":("limit_up_10d_count",),
}
STOCK_FACTOR_GROUP={name:"基础量价" for name in BASE_STOCK_FACTORS}
for group,names in FACTOR_GROUPS.items():
 STOCK_FACTOR_GROUP.update({name:group for name in names})
ALL_STOCK_FACTORS=tuple(STOCK_FACTOR_GROUP)
MARKET_FACTOR_GROUP={
 "market_breadth_pct":"市场广度","market_median_return_pct":"市场收益",
 "market_return_dispersion_pct":"市场波动","market_above_ma20_pct":"市场趋势",
 "market_above_ma60_pct":"市场趋势","market_positive_5d_pct":"市场动量",
}
ALL_MARKET_FACTORS=tuple(MARKET_FACTOR_GROUP)
MARKET_SHARE_FACTORS={"market_breadth_pct","market_above_ma20_pct","market_above_ma60_pct","market_positive_5d_pct"}
DEFAULT_CONFIG={
 "版本":1,"启用个股因子":list(ALL_STOCK_FACTORS),"启用市场因子":list(ALL_MARKET_FACTORS),
 "分位阈值":[0.20,0.30,0.40,0.50,0.60,0.70,0.80],
 "市场广度阈值":[40.0,45.0,50.0,55.0,60.0,65.0],
 "市场中位收益阈值":[-1.0,-0.5,0.0,0.5,1.0],"市场收益离散度阈值":[0.5,0.75,1.0,1.5,2.0,3.0,4.0],
 "短线模型概率阈值网格":[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90],
 "高收益模型概率阈值网格":[0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90],
 "高精度模型概率阈值网格":[0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80]+[round(x/100.0,2) for x in range(85,100)],
 "路径模型概率阈值网格":[round(x/100.0,2) for x in range(50,100)],
 "复利入场概率阈值网格":[0.55,0.60,0.65,0.70,0.75,0.80],
 "复利止盈目标网格百分比":[1.0,1.5,2.0,2.5,3.0],"复利止损幅度网格百分比":[1.0,1.5,2.0,2.5,3.0],
 "参数选择最低交易数":100,"参数选择最大允许回撤百分比":-30.0,
 "最少训练条件样本数":10000,"最少验证条件样本数":5000,"最少市场状态样本数":25000,
 "最大原子条件候选数":24,"最大双条件候选数":36,"每个市场状态原子候选数":10,
 "每个市场状态最多输出条件数":3,"最多输出条件数":30,"目标命中率百分比":80.0,"正式模型自动替换":False,
}
def load_research_config():
 config=dict(DEFAULT_CONFIG)
 if CONFIG_PATH.is_file():
  try: saved=json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
  except (OSError,UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError(f"选股条件配置无法读取：{exc}") from exc
  if not isinstance(saved,dict): raise ValueError("选股条件配置必须是JSON对象")
  config.update(saved)
 for key,allowed in (("启用个股因子",set(ALL_STOCK_FACTORS)),("启用市场因子",set(ALL_MARKET_FACTORS))):
  values=config.get(key)
  if not isinstance(values,list) or not values or any(not isinstance(x,str) for x in values): raise ValueError(f"配置项“{key}”必须是非空字符串列表")
  if len(values)!=len(set(values)): raise ValueError(f"配置项“{key}”存在重复因子")
  unknown=sorted(set(values)-allowed)
  if unknown: raise ValueError(f"配置项“{key}”包含未知因子：{unknown}")
 numeric_keys=("分位阈值","市场广度阈值","市场中位收益阈值","市场收益离散度阈值","短线模型概率阈值网格","高收益模型概率阈值网格","高精度模型概率阈值网格","路径模型概率阈值网格","复利入场概率阈值网格","复利止盈目标网格百分比","复利止损幅度网格百分比")
 for key in numeric_keys:
  values=config.get(key)
  if not isinstance(values,list) or not values: raise ValueError(f"配置项“{key}”必须是非空数字列表")
  try: numbers=[float(x) for x in values]
  except (TypeError,ValueError) as exc: raise ValueError(f"配置项“{key}”必须全部为数字") from exc
  if not all(math.isfinite(x) for x in numbers): raise ValueError(f"配置项“{key}”存在非有限数字")
  if len(numbers)!=len(set(numbers)): raise ValueError(f"配置项“{key}”存在重复值")
  if key=="分位阈值" and any(x<=0 or x>=1 for x in numbers): raise ValueError("分位阈值必须在0与1之间")
  if "概率阈值网格" in key and any(x<0 or x>1 for x in numbers): raise ValueError(f"配置项“{key}”必须在0与1之间")
 for key in ("参数选择最低交易数","最少训练条件样本数","最少验证条件样本数","最少市场状态样本数","最大原子条件候选数","最大双条件候选数","每个市场状态原子候选数","每个市场状态最多输出条件数","最多输出条件数"):
  value=config.get(key)
  if not isinstance(value,int) or isinstance(value,bool) or value<1: raise ValueError(f"配置项“{key}”必须为正整数")
 if config.get("正式模型自动替换") is not False: raise ValueError("研究配置禁止自动替换正式模型")
 return config
def active_stock_factors(config=None):
 return list((load_research_config() if config is None else config)["启用个股因子"])
def active_market_factors(config=None):
 return list((load_research_config() if config is None else config)["启用市场因子"])
def market_thresholds(feature,config=None):
 c=load_research_config() if config is None else config
 key={"market_median_return_pct":"市场中位收益阈值","market_return_dispersion_pct":"市场收益离散度阈值"}.get(feature,"市场广度阈值")
 return [float(x) for x in c[key]]
def raw_market_feature(frame,feature):
 default=50.0 if feature in MARKET_SHARE_FACTORS else 0.0
 return pd.to_numeric(frame[feature],errors="coerce").fillna(default).to_numpy(dtype=np.float64)
def transform_market_feature(frame,feature):
 values=raw_market_feature(frame,feature)
 if feature in MARKET_SHARE_FACTORS: return np.clip(values/100.0,0.0,1.0)
 if feature=="market_median_return_pct": return np.clip(values/5.0,-2.0,2.0)
 if feature=="market_return_dispersion_pct": return np.clip(values/5.0,0.0,2.0)
 raise ValueError(f"没有配置市场因子缩放方式：{feature}")
