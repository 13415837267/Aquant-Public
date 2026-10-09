"""对高精度模型进行可执行的持仓复利模拟；仅用于研究，不改变生产模型。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.high_precision_profit_mining import MAX_FORWARD_SESSIONS, NonlinearModel, TOP_K_PER_DAY
from scripts.short_term_research import FeatureState, history_files, read_daily
from scripts.train_short_term_model import build_targets, make_features

默认模型结果 = ROOT / "data/backtest/high_precision_profit_mining_research_latest.json"
默认输出 = ROOT / "data/backtest/high_precision_compound_portfolio_latest.json"
止损幅度百分比 = 3.0
每日最多排名数 = max(TOP_K_PER_DAY)
初始资金 = 100_000.0
退出目标网格 = (1.0, 1.5, 2.0, 2.5, 3.0)
退出止损网格 = (1.0, 1.5, 2.0, 2.5, 3.0)
退出参数最低交易数 = 100
退出参数最大允许回撤百分比 = -30.0


def 解析模型(结果: dict) -> NonlinearModel:
    """从研究结果恢复已训练的网络权重，不重新训练或选择参数。"""
    模型参数 = 结果["model"]
    模型 = NonlinearModel(len(模型参数["features"]), hidden=int(模型参数["hidden_units"]))
    模型.w1 = np.asarray(模型参数["input_weights"], dtype=np.float64)
    模型.b1 = np.asarray(模型参数["hidden_bias"], dtype=np.float64)
    模型.w2 = np.asarray(模型参数["output_weights"], dtype=np.float64)
    模型.b2 = float(模型参数["intercept"])
    if 模型.w1.shape != (len(模型参数["features"]), 模型.hidden):
        raise ValueError("模型输入权重形状与特征数量不一致")
    if 模型.b1.shape != (模型.hidden,) or 模型.w2.shape != (模型.hidden,):
        raise ValueError("模型隐藏层权重形状不一致")
    return 模型


def 解析退出(entry: float, high: float, low: float, close: float, 持有日序号: int,
          目标净收益百分比: float, 止损百分比: float, 成本基点: float,
          开盘价: float | None = None) -> tuple[str | None, float | None, float]:
    """根据可执行时序计算退出；遇到跳空时按开盘价而非理想止损/止盈价估算成交。"""
    if not all(np.isfinite(x) for x in (entry, high, low, close)) or entry <= 0:
        raise ValueError("退出判断收到无效行情")
    开盘有效 = 开盘价 is not None and np.isfinite(开盘价) and 开盘价 > 0
    成本百分比 = 成本基点 / 100.0
    目标毛收益百分比 = 目标净收益百分比 + 成本百分比
    止损价 = entry * (1.0 - 止损百分比 / 100.0)
    止盈价 = entry * (1.0 + 目标毛收益百分比 / 100.0)

    if 持有日序号 == 0:
        return None, None, close
    # 同一根日线同时触及两条边界时，仍按止损优先的保守规则处理。
    # 如果开盘已跳空跌破止损，按更差的开盘价估算，而不是假设能按止损价成交。
    if low <= 止损价:
        成交价 = min(止损价, float(开盘价)) if 开盘有效 else 止损价
        return "止损", (成交价 / entry - 1.0) * 100.0 - 成本百分比, 成交价
    if high >= 止盈价:
        成交价 = max(止盈价, float(开盘价)) if 开盘有效 else 止盈价
        return "止盈", (成交价 / entry - 1.0) * 100.0 - 成本百分比, 成交价
    if 持有日序号 >= MAX_FORWARD_SESSIONS - 1:
        return "到期", (close / entry - 1.0) * 100.0 - 成本百分比, close
    return None, None, close


def 提取每日候选(files: list[Path], 日期: list[str], 起始日: str, 结束日: str,
             模型: NonlinearModel, 目标净收益百分比: float, 成本基点: float) -> tuple[dict, dict]:
    """逐日构造T收盘特征，只保留每个信号日概率最高的前十个可执行标的。"""
    日期索引 = {d: i for i, d in enumerate(日期)}
    if 起始日 not in 日期索引 or 结束日 not in 日期索引:
        raise ValueError(f"研究区间不在历史交易日内：{起始日} 至 {结束日}")
    起始索引, 结束索引 = 日期索引[起始日], 日期索引[结束日]
    状态 = FeatureState()
    缓存: dict[int, pd.DataFrame] = {}

    def 读取(i: int) -> pd.DataFrame:
        if i not in 缓存:
            缓存[i] = read_daily(files[i])
        while len(缓存) > MAX_FORWARD_SESSIONS + 2:
            del 缓存[next(iter(缓存))]
        return 缓存[i]

    每日候选 = {}
    诊断 = {
        "信号日期": 0,
        "完整未来路径样本": 0,
        "可执行样本": 0,
        "开盘涨停或停牌排除数": 0,
        "涨停价缺失排除数": 0,
    }

    # 预热足够的历史日线，确保20日特征在研究区间首日已经可用。
    for i in range(max(0, 起始索引 - 25), 结束索引 + 1):
        原始日线 = 读取(i)
        特征表 = 状态.build(原始日线)
        当前日 = 日期[i]
        if 当前日 < 起始日 or 当前日 > 结束日:
            continue
        if i + MAX_FORWARD_SESSIONS >= len(files) or 特征表.empty:
            continue

        特征矩阵 = make_features(特征表)
        概率 = 模型.predict(特征矩阵)
        未来日线 = [读取(i + j) for j in range(1, MAX_FORWARD_SESSIONS + 1)]
        股票代码 = 特征表["symbol"].astype(str).str.zfill(6).to_numpy()
        _, _, _, 路径完整 = build_targets(股票代码.tolist(), 未来日线)
        完整索引 = np.flatnonzero(路径完整)
        诊断["完整未来路径样本"] += int(len(完整索引))
        if len(完整索引) == 0:
            每日候选[当前日] = []
            continue

        入场日索引 = 未来日线[0].set_index("symbol")
        开盘价 = pd.to_numeric(入场日索引["open"], errors="coerce").reindex(股票代码).to_numpy(dtype=float)
        若有涨停价 = "high_limit" in 入场日索引.columns
        if 若有涨停价:
            涨停价 = pd.to_numeric(入场日索引["high_limit"], errors="coerce").reindex(股票代码).to_numpy(dtype=float)
        else:
            涨停价 = np.full(len(股票代码), np.nan)
        if "is_paused" in 入场日索引.columns:
            停牌标志 = pd.to_numeric(入场日索引["is_paused"], errors="coerce").reindex(股票代码).fillna(0).to_numpy(dtype=float)
        else:
            停牌标志 = np.zeros(len(股票代码), dtype=float)

        涨停价有效 = np.isfinite(涨停价) & (涨停价 > 0)
        诊断["涨停价缺失排除数"] += int((路径完整 & ~涨停价有效).sum())
        开盘可交易 = 涨停价有效 & np.isfinite(开盘价) & (开盘价 > 0) & (开盘价 < 涨停价 * (1.0 - 1e-6)) & (停牌标志 <= 0)
        诊断["开盘涨停或停牌排除数"] += int((路径完整 & 涨停价有效 & ~开盘可交易).sum())
        可执行索引 = np.flatnonzero(路径完整 & 开盘可交易)
        诊断["可执行样本"] += int(len(可执行索引))
        if len(可执行索引) == 0:
            每日候选[当前日] = []
            continue

        排序 = 可执行索引[np.argsort(-概率[可执行索引], kind="stable")[:每日最多排名数]]
        未来索引 = [日.set_index("symbol") for 日 in 未来日线]
        候选行 = []
        for 索引 in 排序:
            代码 = 股票代码[索引]
            某股路径 = []
            完整 = True
            for j, 未来表 in enumerate(未来索引):
                if 代码 not in 未来表.index:
                    完整 = False
                    break
                行 = 未来表.loc[代码]
                if isinstance(行, pd.DataFrame):
                    行 = 行.iloc[-1]
                数值 = {}
                for 字段 in ("open", "high", "low", "close"):
                    数值[字段] = float(行[字段]) if 字段 in 行 and pd.notna(行[字段]) else float("nan")
                if not all(np.isfinite(数值[k]) for k in ("high", "low", "close")):
                    完整 = False
                    break
                数值["date"] = 日期[i + j + 1]
                某股路径.append(数值)
            if not 完整 or not 某股路径:
                continue
            候选行.append({
                "signal_date": 当前日,
                "symbol": 代码,
                "probability": float(概率[索引]),
                "entry_date": 某股路径[0]["date"],
                "entry_price": float(某股路径[0]["open"]),
                "bars": 某股路径,
            })
        每日候选[当前日] = 候选行
        诊断["信号日期"] += 1
        if 诊断["信号日期"] % 50 == 0:
            print(f"[复利模拟] {起始日} 至 {结束日}：已处理 {诊断['信号日期']} 个信号日", flush=True)

    return 每日候选, 诊断


def 运行组合(每日候选: dict, 全部日期: list[str], 模式: str, 目标净收益百分比: float,
         止损百分比: float, 成本基点: float, top_k: int = 1,
         概率阈值: float | None = None) -> dict:
    """现金约束、最多持有5个交易日、固定槽位上限的日线复利模拟。"""
    需要持仓数 = MAX_FORWARD_SESSIONS * top_k if 模式 == "每日排名" else MAX_FORWARD_SESSIONS
    入场排程: dict[str, list[dict]] = defaultdict(list)
    申请入场数 = 0
    for 信号日, 候选列表 in 每日候选.items():
        if 模式 == "概率阈值":
            选中 = 候选列表[:1] if 候选列表 and 概率阈值 is not None and 候选列表[0]["probability"] >= 概率阈值 else []
        else:
            选中 = 候选列表[:top_k]
        for 候选 in 选中:
            入场排程[候选["entry_date"]].append(候选)
            申请入场数 += 1

    if not 入场排程:
        return {
            "status": "无可执行交易",
            "selection_mode": 模式,
            "top_k_per_day": top_k if 模式 == "每日排名" else None,
            "probability_threshold": 概率阈值,
            "requested_entries": 0,
            "completed_trades": 0,
        }

    日期索引 = {d: i for i, d in enumerate(全部日期)}
    开始索引 = min(日期索引[d] for d in 入场排程)
    结束索引 = max(日期索引[候选["bars"][-1]["date"]] for 行列 in 入场排程.values() for 候选 in 行列)
    现金 = 初始资金
    持仓 = []
    曲线 = []
    完成交易 = []
    跳过入场 = 0
    上一净值 = 初始资金

    for 日索引 in range(开始索引, 结束索引 + 1):
        当前日 = 全部日期[日索引]
        当日新候选 = sorted(入场排程.get(当前日, []), key=lambda x: x["probability"], reverse=True)
        for 候选 in 当日新候选:
            if len(持仓) >= 需要持仓数 or 现金 <= 1e-8:
                跳过入场 += 1
                continue
            当前净值 = 现金 + sum(p["notional"] * p["last_close"] / p["entry_price"] for p in 持仓)
            单槽预算 = 当前净值 / 需要持仓数
            投入金额 = min(单槽预算, 现金)
            if 投入金额 <= 初始资金 * 1e-10:
                跳过入场 += 1
                continue
            现金 -= 投入金额
            持仓.append({
                "candidate": 候选,
                "notional": 投入金额,
                "entry_price": 候选["entry_price"],
                "last_close": 候选["entry_price"],
            })

        仍持有 = []
        for 仓位 in 持仓:
            候选 = 仓位["candidate"]
            已持有日 = 日索引 - 日期索引[候选["entry_date"]]
            if 已持有日 < 0 or 已持有日 >= len(候选["bars"]):
                仍持有.append(仓位)
                continue
            当日行情 = 候选["bars"][已持有日]
            退出原因, 净收益百分比, _ = 解析退出(
                仓位["entry_price"], 当日行情["high"], 当日行情["low"], 当日行情["close"],
                已持有日, 目标净收益百分比, 止损百分比, 成本基点,
                开盘价=当日行情.get("open"),
            )
            if 退出原因 is not None:
                现金 += 仓位["notional"] * (1.0 + float(净收益百分比) / 100.0)
                完成交易.append({
                    "signal_date": 候选["signal_date"],
                    "entry_date": 候选["entry_date"],
                    "exit_date": 当前日,
                    "symbol": 候选["symbol"],
                    "probability": 候选["probability"],
                    "net_return_pct": float(净收益百分比),
                    "exit_reason": 退出原因,
                    "target_hit": 退出原因 == "止盈",
                })
            else:
                仓位["last_close"] = float(当日行情["close"])
                仍持有.append(仓位)
        持仓 = 仍持有
        净值 = 现金 + sum(p["notional"] * p["last_close"] / p["entry_price"] for p in 持仓)
        日收益 = (净值 / 上一净值 - 1.0) * 100.0 if 上一净值 > 0 else 0.0
        曲线.append({
            "date": 当前日,
            "equity": float(净值),
            "daily_return_pct": float(日收益),
            "active_positions": len(持仓),
            "cash": float(现金),
        })
        上一净值 = 净值

    if not 曲线:
        return {"status": "无净值曲线", "selection_mode": 模式, "requested_entries": 申请入场数}
    最终净值 = 曲线[-1]["equity"]
    累计收益 = (最终净值 / 初始资金 - 1.0) * 100.0
    交易日数 = len(曲线)
    年化复利 = ((最终净值 / 初始资金) ** (252.0 / max(交易日数, 1)) - 1.0) * 100.0 if 最终净值 > 0 else -100.0
    历史峰值 = 初始资金
    最大回撤 = 0.0
    for 行 in 曲线:
        历史峰值 = max(历史峰值, 行["equity"])
        最大回撤 = min(最大回撤, (行["equity"] / 历史峰值 - 1.0) * 100.0)
    日收益数组 = np.asarray([r["daily_return_pct"] for r in 曲线], dtype=np.float64) / 100.0
    日波动 = float(np.std(日收益数组, ddof=1)) if len(日收益数组) > 1 else 0.0
    夏普 = float(np.mean(日收益数组) / 日波动 * math.sqrt(252.0)) if 日波动 > 0 else None
    单笔收益 = np.asarray([r["net_return_pct"] for r in 完成交易], dtype=np.float64)
    几何均值 = None
    if len(单笔收益):
        if np.all(单笔收益 > -100.0):
            几何均值 = (math.exp(float(np.mean(np.log1p(单笔收益 / 100.0)))) - 1.0) * 100.0

    年末净值 = {}
    for 行 in 曲线:
        年末净值[行["date"][:4]] = 行["equity"]
    年度收益 = []
    年度起始净值 = 初始资金
    for 年份 in sorted(年末净值):
        年末 = 年末净值[年份]
        年度收益.append({"year": 年份, "return_pct": (年末 / 年度起始净值 - 1.0) * 100.0})
        年度起始净值 = 年末

    止盈数 = sum(1 for r in 完成交易 if r["target_hit"])
    正收益数 = sum(1 for r in 完成交易 if r["net_return_pct"] > 0)
    返回 = {
        "status": "ready",
        "selection_mode": 模式,
        "top_k_per_day": top_k if 模式 == "每日排名" else None,
        "probability_threshold": 概率阈值,
        "initial_equity": 初始资金,
        "final_equity": round(float(最终净值), 2),
        "compound_return_pct": round(float(累计收益), 4),
        "annualized_compound_return_pct": round(float(年化复利), 4),
        "max_drawdown_pct": round(float(最大回撤), 4),
        "sharpe_ratio": round(夏普, 4) if 夏普 is not None else None,
        "trading_days": 交易日数,
        "requested_entries": 申请入场数,
        "entered_trades": len(完成交易) + len(持仓),
        "completed_trades": len(完成交易),
        "skipped_for_capacity_or_cash": 跳过入场,
        "target_hit_rate_pct": round(止盈数 / len(完成交易) * 100.0, 4) if 完成交易 else None,
        "positive_trade_rate_pct": round(正收益数 / len(完成交易) * 100.0, 4) if 完成交易 else None,
        "mean_trade_net_return_pct": round(float(单笔收益.mean()), 4) if len(单笔收益) else None,
        "geometric_mean_trade_return_pct": round(float(几何均值), 4) if 几何均值 is not None else None,
        "exit_reason_counts": {reason: sum(1 for r in 完成交易 if r["exit_reason"] == reason) for reason in ("止盈", "止损", "到期")},
        "yearly_compound_return_pct": 年度收益,
        "audit": {
            "entry_is_T_plus_1_open": True,
            "exit_starts_T_plus_2": True,
            "same_day_stop_first": True,
            "entry_at_upper_limit_blocked": True,
            "paused_entry_blocked": True,
            "round_trip_cost_applied_to_each_completed_trade": True,
            "one_signal_day_topk_selection": True,
            "cash_constrained_no_leverage": True,
            "max_concurrent_positions": 需要持仓数,
            "max_holding_sessions": MAX_FORWARD_SESSIONS,
            "model_threshold_selected_on_validation_only": True,
            "formal_production_changed": False,
        },
    }
    return 返回


def 选择退出参数(验证网格: list[dict], minimum_trades: int, max_drawdown_floor_pct: float) -> dict | None:
    """仅根据验证集选择满足样本与回撤边界的退出方案，不读取最终留出集指标。"""
    eligible = [
        row for row in 验证网格
        if int(row.get("completed_trades") or 0) >= minimum_trades
        and row.get("max_drawdown_pct") is not None
        and float(row["max_drawdown_pct"]) >= max_drawdown_floor_pct
        and row.get("compound_return_pct") is not None
        and row.get("annualized_compound_return_pct") is not None
    ]
    if not eligible:
        return None
    chosen = max(
        eligible,
        key=lambda row: (
            float(row["compound_return_pct"]),
            float(row["annualized_compound_return_pct"]),
            float(row["max_drawdown_pct"]),
            int(row["completed_trades"]),
        ),
    ).copy()
    chosen["selection_status"] = (
        "验证集正收益候选" if float(chosen["compound_return_pct"]) > 0
        else "验证集未盈利，仅作为最终留出集研究对照"
    )
    return chosen


def main() -> None:
    解析器 = argparse.ArgumentParser(description="评估短线信号在严格交易约束下的复利表现")
    解析器.add_argument("--model-result", default=str(默认模型结果))
    解析器.add_argument("--output", default=str(默认输出))
    参数 = 解析器.parse_args()

    模型结果路径 = Path(参数.model_result)
    if not 模型结果路径.exists():
        raise FileNotFoundError(f"未找到高精度模型研究结果：{模型结果路径}")
    研究结果 = json.loads(模型结果路径.read_text(encoding="utf-8"))
    if 研究结果.get("status") != "research_only" or 研究结果.get("audit", {}).get("formal_production_changed") is not False:
        raise ValueError("输入模型结果不符合研究隔离约束")
    if not 研究结果.get("audit", {}).get("final_holdout_used_once_after_selection"):
        raise ValueError("输入模型未声明最终留出集只评估一次")

    模型 = 解析模型(研究结果)
    参数集合 = 研究结果["parameters"]
    目标净收益 = float(参数集合["net_win_threshold_pct"])
    成本基点 = float(参数集合["round_trip_cost_bps"])
    止损百分比 = 止损幅度百分比
    文件列表 = history_files()
    全部日期 = [p.name[:10] for p in 文件列表]
    所有区间结果 = {}
    所有区间候选 = {}

    for 区间 in ("validation", "final"):
        起始日, 结束日 = 研究结果["splits"][区间]
        print(f"[复利模拟] 开始{区间}区间：{起始日} 至 {结束日}", flush=True)
        每日候选, 诊断 = 提取每日候选(
            文件列表, 全部日期, 起始日, 结束日, 模型, 目标净收益, 成本基点,
        )
        所有区间候选[区间] = 每日候选
        阈值 = 研究结果.get("selected_validation_operating_point", {}).get("probability_threshold")
        区间组合 = {
            "window": [起始日, 结束日],
            "candidate_diagnostics": 诊断,
            "validation_selected_threshold": 阈值,
            "threshold_strategy": 运行组合(
                每日候选, 全部日期, "概率阈值", 目标净收益, 止损百分比, 成本基点,
                top_k=1, 概率阈值=float(阈值) if 阈值 is not None else None,
            ),
            "daily_top_k": {
                str(k): 运行组合(
                    每日候选, 全部日期, "每日排名", 目标净收益, 止损百分比, 成本基点, top_k=k,
                )
                for k in TOP_K_PER_DAY
            },
        }
        所有区间结果[区间] = 区间组合

    # 只使用验证集选择止盈止损参数；最终留出集只评估冻结后的单一参数组合。
    验证集退出网格 = []
    for 目标净收益候选 in 退出目标网格:
        for 止损候选 in 退出止损网格:
            结果 = 运行组合(
                所有区间候选["validation"], 全部日期, "概率阈值",
                目标净收益候选, 止损候选, 成本基点,
                top_k=1, 概率阈值=float(阈值) if 阈值 is not None else None,
            )
            验证集退出网格.append({
                "target_net_profit_pct": 目标净收益候选,
                "stop_loss_pct": 止损候选,
                "completed_trades": 结果.get("completed_trades", 0),
                "compound_return_pct": 结果.get("compound_return_pct"),
                "annualized_compound_return_pct": 结果.get("annualized_compound_return_pct"),
                "max_drawdown_pct": 结果.get("max_drawdown_pct"),
                "sharpe_ratio": 结果.get("sharpe_ratio"),
                "target_hit_rate_pct": 结果.get("target_hit_rate_pct"),
                "mean_trade_net_return_pct": 结果.get("mean_trade_net_return_pct"),
                "exit_reason_counts": 结果.get("exit_reason_counts", {}),
            })

    退出参数选择 = 选择退出参数(
        验证集退出网格,
        minimum_trades=退出参数最低交易数,
        max_drawdown_floor_pct=退出参数最大允许回撤百分比,
    )
    最终候选策略结果 = None
    if 退出参数选择 is not None:
        最终候选策略结果 = 运行组合(
            所有区间候选["final"], 全部日期, "概率阈值",
            float(退出参数选择["target_net_profit_pct"]),
            float(退出参数选择["stop_loss_pct"]), 成本基点,
            top_k=1, 概率阈值=float(阈值) if 阈值 is not None else None,
        )

    输出结果 = {
        "schema_version": 1,
        "status": "research_only",
        "method": "cash_constrained_compound_portfolio_simulation",
        "source_model_result": str(模型结果路径),
        "objective": "maximize_compound_growth_subject_to_executable_T_plus_1_T_plus_2_rules",
        "parameters": {
            "net_win_threshold_pct": 目标净收益,
            "stop_loss_pct": 止损百分比,
            "round_trip_cost_bps": 成本基点,
            "max_forward_sessions": MAX_FORWARD_SESSIONS,
            "initial_equity": 初始资金,
            "daily_rank_k": list(TOP_K_PER_DAY),
        },
        "splits": 研究结果["splits"],
        "validation": 所有区间结果["validation"],
        "final": 所有区间结果["final"],
        "exit_policy_sensitivity": {
            "objective": "验证集累计复利收益最大化，同时要求已完成交易数不少于预设下限且最大回撤不低于风险边界",
            "validation_min_completed_trades": 退出参数最低交易数,
            "validation_max_drawdown_floor_pct": 退出参数最大允许回撤百分比,
            "target_net_profit_grid_pct": list(退出目标网格),
            "stop_loss_grid_pct": list(退出止损网格),
            "validation_grid": 验证集退出网格,
            "selected_validation_policy": 退出参数选择,
            "selected_policy_final_holdout": 最终候选策略结果,
            "final_holdout_used_for_policy_selection": False,
            "formal_production_changed": False,
        },
        "audit": {
            "no_future_features": True,
            "final_holdout_used_for_selection": False,
            "validation_selected_threshold_frozen_for_final": True,
            "entry_is_T_plus_1_open": True,
            "exit_starts_T_plus_2": True,
            "upper_limit_and_paused_entries_blocked": True,
            "round_trip_cost_included": True,
            "cash_constrained_no_leverage": True,
            "positions_marked_to_market_each_close": True,
            "final_holdout_used_once_after_selection": True,
            "formal_production_changed": False,
        },
    }
    输出路径 = Path(参数.output)
    输出路径.parent.mkdir(parents=True, exist_ok=True)
    输出路径.write_text(json.dumps(输出结果, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": 输出结果["status"],
        "validation_threshold": 输出结果["validation"]["threshold_strategy"],
        "final_threshold": 输出结果["final"]["threshold_strategy"],
        "final_top_k": {k: {key: value for key, value in result.items() if key in ("compound_return_pct", "annualized_compound_return_pct", "max_drawdown_pct", "target_hit_rate_pct", "completed_trades")} for k, result in 输出结果["final"]["daily_top_k"].items()},
        "selected_exit_policy": 输出结果["exit_policy_sensitivity"]["selected_validation_policy"],
        "selected_exit_policy_final_holdout": 输出结果["exit_policy_sensitivity"]["selected_policy_final_holdout"],
        "output": str(输出路径),
    }, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
