"""外部公开策略统一历史评估，仅限研究，不参与正式候选生产。"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.short_term_research import (  # noqa: E402
    ENTRY_LIMIT_UP_BLOCK,
    MAX_HOLDING_SESSIONS,
    MIN_EXIT_DAY,
    STOP_PCT,
    WIN_THRESHOLD_PCT,
    FeatureState,
    history_files,
    read_daily,
)

MODULE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = MODULE_DIR / "config.json"
OUTPUT_DEFAULT = ROOT / "data" / "backtest" / "external_strategy_evaluation_latest.json"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """读取集中参数，避免在业务代码中分散硬编码交易阈值。"""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


CONFIG = load_config()

STRATEGIES: list[dict[str, Any]] = [
    {
        "id": "public_picker_approx",
        "name": "公开短线选股规则（近似复现）",
        "source_url": "https://github.com/online0001/short-term-stock-picker",
        "replication": "近似复现",
        "limitations": [
            "公开说明中的流通市值条件无法由当前统一日线文件完整验证。",
            "在现有特征和统一可执行组合框架下复现公开的均线、近期涨停、换手与量能条件；不是原项目逐行复现。",
        ],
    },
    {
        "id": "khunter_pullback_proxy",
        "name": "趋势回调缩量代理规则",
        "source_url": "https://github.com/ling-0729/KHunter",
        "replication": "代理规则",
        "limitations": [
            "公开项目的数据接口与具体参数未全部对齐。",
            "按可用日线构造趋势、回撤和缩量代理条件，不代表 KHunter 原策略原样复现。",
        ],
    },
    {
        "id": "momentum20_baseline",
        "name": "20日动量基线",
        "source_url": "https://github.com/whilefog233/MiniQuantLab",
        "replication": "可复现基线，不是原策略复现",
        "limitations": [
            "仅用现有A股日线检验多头动量基线，不等于原项目多空组合或论文中的完整时序动量实现。",
        ],
    },
]

NON_REPRODUCIBLE = CONFIG.get("non_reproducible_sources", [])


def _series(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def score_candidates(strategy_id: str, features: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """仅使用输入日特征生成可排序信号，不读取未来行情。"""
    cfg = config or CONFIG
    if features.empty or "symbol" not in features.columns:
        return pd.DataFrame(columns=list(features.columns) + (["score"] if "score" not in features.columns else []))

    rules = cfg["strategy_rules"]
    if strategy_id == "public_picker_approx":
        rule = rules[strategy_id]
        ma5_ma10 = _series(features, "ma5_vs_ma10_pct")
        ma10_ma20 = _series(features, "ma10_vs_ma20_pct")
        recent_limitups = _series(features, "limit_up_10d_count")
        turnover = _series(features, "turnover_pct")
        volume_ratio = _series(features, "volume_ratio_5d")
        amount_20d = _series(features, "amount_20d")
        change = _series(features, "change_pct").fillna(0.0)
        mask = (
            ma5_ma10.gt(0)
            & ma10_ma20.gt(0)
            & recent_limitups.ge(float(rule["recent_limit_up_count_min"]))
            & turnover.between(float(rule["turnover_min_pct"]), float(rule["turnover_max_pct"]))
            & volume_ratio.ge(float(rule["volume_ratio_5d_min"]))
            & amount_20d.ge(float(rule["amount_20d_min"]))
        )
        score = recent_limitups * 20.0 + change + volume_ratio * 5.0

    elif strategy_id == "khunter_pullback_proxy":
        rule = rules[strategy_id]
        ma20_ma60 = _series(features, "ma20_vs_ma60_pct")
        close_ma20 = _series(features, "close_vs_ma20_pct")
        close_ma5 = _series(features, "close_vs_ma5_pct")
        pullback = _series(features, "distance_from_high_20d_pct")
        volume_ratio = _series(features, "volume_ratio_5d")
        turnover = _series(features, "turnover_pct")
        mask = (
            ma20_ma60.gt(float(rule["ma20_vs_ma60_min_pct"]))
            & close_ma20.ge(float(rule["close_vs_ma20_min_pct"]))
            & close_ma5.between(float(rule["close_vs_ma5_min_pct"]), float(rule["close_vs_ma5_max_pct"]))
            & pullback.between(float(rule["distance_from_high_20d_min_pct"]), float(rule["distance_from_high_20d_max_pct"]))
            & volume_ratio.le(float(rule["volume_ratio_5d_max"]))
            & turnover.between(float(rule["turnover_min_pct"]), float(rule["turnover_max_pct"]))
        )
        score = (
            ma20_ma60
            - close_ma5.abs()
            - (pullback - float(rule["target_pullback_pct"])).abs()
            + (1.0 - volume_ratio) * 5.0
        )

    elif strategy_id == "momentum20_baseline":
        rule = rules[strategy_id]
        momentum = _series(features, "return_20d_pct")
        close_ma20 = _series(features, "close_vs_ma20_pct")
        ma20_ma60 = _series(features, "ma20_vs_ma60_pct")
        volatility = _series(features, "volatility_10d_pct")
        mask = (
            momentum.gt(float(rule["return_20d_min_pct"]))
            & close_ma20.gt(float(rule["close_vs_ma20_min_pct"]))
            & ma20_ma60.gt(float(rule["ma20_vs_ma60_min_pct"]))
            & volatility.ge(float(rule["volatility_10d_min_pct"]))
        )
        score = momentum / volatility.replace(0.0, np.nan)

    else:
        raise ValueError(f"未知的外部策略标识：{strategy_id}")

    result = features.loc[mask].copy()
    result["score"] = pd.to_numeric(score.loc[result.index], errors="coerce")
    result = result.loc[np.isfinite(result["score"].to_numpy(dtype=float))]
    if result.empty:
        return result
    return result.sort_values(["score", "symbol"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def exit_decision(position: dict[str, Any], row: Any, days_since_entry: int) -> dict[str, Any] | None:
    """根据当前已完成日线判断是否退出；买入当日禁止退出，止损优先。"""
    holding_session = int(days_since_entry) + 1
    if holding_session < int(MIN_EXIT_DAY):
        return None

    paused = pd.to_numeric(pd.Series([row.get("is_paused", 0)]), errors="coerce").iloc[0]
    if pd.notna(paused) and float(paused) > 0:
        return None

    low = pd.to_numeric(pd.Series([row.get("low", np.nan)]), errors="coerce").iloc[0]
    high = pd.to_numeric(pd.Series([row.get("high", np.nan)]), errors="coerce").iloc[0]
    close = pd.to_numeric(pd.Series([row.get("close", np.nan)]), errors="coerce").iloc[0]

    if pd.notna(low) and np.isfinite(float(low)) and float(low) <= float(position["stop_price"]):
        return {"exit_price": float(position["stop_price"]), "exit_reason": "止损", "holding_days": holding_session}
    if pd.notna(high) and np.isfinite(float(high)) and float(high) >= float(position["target_price"]):
        return {"exit_price": float(position["target_price"]), "exit_reason": "止盈", "holding_days": holding_session}
    if holding_session >= int(MAX_HOLDING_SESSIONS) and pd.notna(close) and np.isfinite(float(close)) and float(close) > 0:
        return {"exit_price": float(close), "exit_reason": "持有期满", "holding_days": holding_session}
    return None


class Portfolio:
    """双槽位、现金约束、按日线收盘盯市的简化组合模拟器。"""

    def __init__(
        self,
        strategy: dict[str, Any],
        period: dict[str, Any],
        start_index: int,
        end_index: int,
        config: dict[str, Any],
        one_way_cost_rate: float,
        initial_equity: float,
    ) -> None:
        self.strategy = strategy
        self.period = period
        self.start_index = start_index
        self.end_index = end_index
        self.config = config
        self.one_way_cost_rate = one_way_cost_rate
        self.cash = float(initial_equity)
        self.last_equity = float(initial_equity)
        self.max_positions = int(config["max_concurrent_positions"])
        self.positions: dict[str, dict[str, Any]] = {}
        self.equity_curve: list[dict[str, Any]] = []
        self.trades: list[dict[str, Any]] = []
        self.signal_days = 0
        self.candidate_days = 0

    def enter_candidates(
        self,
        symbols: list[str],
        day_indexed: pd.DataFrame,
        day_index: int,
        dates: list[str],
    ) -> None:
        """按昨日排序信号在今日开盘执行，先买后判断当日退出，避免使用日内未来信息。"""
        if not symbols or len(self.positions) >= self.max_positions:
            return

        for symbol in symbols:
            if len(self.positions) >= self.max_positions:
                break
            if symbol in self.positions or symbol not in day_indexed.index:
                continue

            row = day_indexed.loc[symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            entry = pd.to_numeric(pd.Series([row.get("open", np.nan)]), errors="coerce").iloc[0]
            if pd.isna(entry) or not np.isfinite(float(entry)) or float(entry) <= 0:
                continue
            paused = pd.to_numeric(pd.Series([row.get("is_paused", 0)]), errors="coerce").iloc[0]
            st_flag = pd.to_numeric(pd.Series([row.get("is_st", 0)]), errors="coerce").iloc[0]
            if (pd.notna(paused) and float(paused) > 0) or (pd.notna(st_flag) and float(st_flag) > 0):
                continue

            name = str(row.get("name", "")).upper()
            if "ST" in name or "退" in name:
                continue

            high_limit = pd.to_numeric(pd.Series([row.get("high_limit", np.nan)]), errors="coerce").iloc[0]
            if ENTRY_LIMIT_UP_BLOCK:
                if pd.isna(high_limit) or not np.isfinite(float(high_limit)) or float(high_limit) <= 0:
                    continue
                if float(entry) >= float(high_limit) * (1.0 - 1e-6):
                    continue

            budget = min(self.cash, max(self.last_equity, 0.0) / self.max_positions)
            if budget <= np.finfo(float).eps:
                break

            entry_price = float(entry)
            invested = float(budget)
            quantity = invested / (entry_price * (1.0 + self.one_way_cost_rate))
            target_net = float(WIN_THRESHOLD_PCT) / 100.0
            target_price = entry_price * (1.0 + target_net) * (1.0 + self.one_way_cost_rate) / (1.0 - self.one_way_cost_rate)
            stop_price = entry_price * (1.0 - float(STOP_PCT) / 100.0)
            self.cash -= invested
            self.positions[symbol] = {
                "symbol": symbol,
                "signal_date": dates[day_index - 1],
                "entry_date": dates[day_index],
                "entry_index": day_index,
                "entry_price": entry_price,
                "quantity": quantity,
                "invested": invested,
                "stop_price": stop_price,
                "target_price": target_price,
                "last_close": entry_price,
            }

    def update_exits(self, day_indexed: pd.DataFrame, day_index: int, dates: list[str]) -> None:
        for symbol, position in list(self.positions.items()):
            if day_index <= int(position["entry_index"]):
                continue
            if symbol not in day_indexed.index:
                continue
            row = day_indexed.loc[symbol]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            close = pd.to_numeric(pd.Series([row.get("close", np.nan)]), errors="coerce").iloc[0]
            if pd.notna(close) and np.isfinite(float(close)) and float(close) > 0:
                position["last_close"] = float(close)

            decision = exit_decision(position, row, day_index - int(position["entry_index"]))
            if decision is None:
                continue

            exit_price = float(decision["exit_price"])
            proceeds = float(position["quantity"]) * exit_price * (1.0 - self.one_way_cost_rate)
            net_return_pct = (proceeds / float(position["invested"]) - 1.0) * 100.0
            self.cash += proceeds
            self.trades.append(
                {
                    "股票代码": symbol,
                    "信号日期": position["signal_date"],
                    "买入日期": position["entry_date"],
                    "卖出日期": dates[day_index],
                    "买入价格": round(float(position["entry_price"]), 6),
                    "卖出价格": round(exit_price, 6),
                    "净收益率百分比": round(net_return_pct, 6),
                    "退出原因": decision["exit_reason"],
                    "持有交易日": int(decision["holding_days"]),
                    "资金占比基数": round(float(position["invested"]), 8),
                }
            )
            del self.positions[symbol]

    def mark_to_market(self, day_indexed: pd.DataFrame, day_index: int, dates: list[str]) -> None:
        equity = float(self.cash)
        for symbol, position in self.positions.items():
            if symbol in day_indexed.index:
                row = day_indexed.loc[symbol]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                close = pd.to_numeric(pd.Series([row.get("close", np.nan)]), errors="coerce").iloc[0]
                paused = pd.to_numeric(pd.Series([row.get("is_paused", 0)]), errors="coerce").iloc[0]
                if pd.notna(close) and np.isfinite(float(close)) and float(close) > 0 and not (pd.notna(paused) and float(paused) > 0):
                    position["last_close"] = float(close)
            equity += float(position["quantity"]) * float(position["last_close"]) * (1.0 - self.one_way_cost_rate)

        self.last_equity = max(equity, 0.0)
        self.equity_curve.append({"日期": dates[day_index], "净值": round(self.last_equity, 10)})

    def result(self, annualization_days: int) -> dict[str, Any]:
        values = np.asarray([point["净值"] for point in self.equity_curve], dtype=float)
        returns = values[1:] / values[:-1] - 1.0 if len(values) > 1 and np.all(values[:-1] > 0) else np.array([], dtype=float)
        if len(values) > 1:
            peaks = np.maximum.accumulate(values)
            drawdowns = values / np.where(peaks > 0, peaks, np.nan) - 1.0
            max_drawdown = float(np.nanmin(drawdowns) * 100.0)
            days = max(len(values) - 1, 1)
            if values[0] > 0 and values[-1] > 0:
                annualized = (float(values[-1] / values[0]) ** (float(annualization_days) / days) - 1.0) * 100.0
            else:
                annualized = -100.0
            total_return = (float(values[-1] / values[0]) - 1.0) * 100.0 if values[0] > 0 else None
        else:
            max_drawdown = 0.0
            annualized = 0.0
            total_return = 0.0

        daily_std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        sharpe = float(np.mean(returns) / daily_std * math.sqrt(annualization_days)) if daily_std > 0 else None
        net_returns = np.asarray([trade["净收益率百分比"] for trade in self.trades], dtype=float)
        win_threshold = float(WIN_THRESHOLD_PCT)
        positive_returns = net_returns[net_returns > 0]
        negative_returns = net_returns[net_returns < 0]
        gross_profit = float(positive_returns.sum())
        gross_loss = float(abs(negative_returns.sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (None if gross_profit == 0 else float("inf"))

        return {
            "区间": self.period["label"],
            "开始日期": self.period["start"],
            "结束日期": self.period["end"],
            "信号日数": int(self.signal_days),
            "出现候选的信号日数": int(self.candidate_days),
            "候选覆盖率百分比": round(self.candidate_days / self.signal_days * 100.0, 4) if self.signal_days else 0.0,
            "初始净值": round(float(values[0]), 10) if len(values) else None,
            "期末净值": round(float(values[-1]), 10) if len(values) else None,
            "累计收益百分比": round(total_return, 6) if total_return is not None else None,
            "年化复利百分比": round(annualized, 6),
            "最大回撤百分比": round(max_drawdown, 6),
            "年化夏普": round(sharpe, 6) if sharpe is not None else None,
            "平仓交易数": int(len(self.trades)),
            "正收益交易率百分比": round(float((net_returns > 0).mean() * 100.0), 4) if len(net_returns) else None,
            "严格净收益目标胜率百分比": round(float((net_returns >= win_threshold).mean() * 100.0), 4) if len(net_returns) else None,
            "平均单笔净收益百分比": round(float(net_returns.mean()), 6) if len(net_returns) else None,
            "盈利因子": round(profit_factor, 6) if profit_factor is not None and np.isfinite(profit_factor) else None,
            "区间末未平仓数": int(len(self.positions)),
            "未平仓代码": sorted(self.positions.keys()),
            "净值曲线": self.equity_curve,
            "逐笔交易": self.trades,
        }


def run_evaluation(
    period_id: str = "all",
    output_path: Path = OUTPUT_DEFAULT,
    commission_bps_per_side: float | None = None,
    slippage_bps_per_side: float | None = None,
) -> dict[str, Any]:
    cfg = load_config()
    files = history_files()
    full_dates = [path.name[:10] for path in files]
    date_index = {date: index for index, date in enumerate(full_dates)}

    available_periods = cfg["evaluation_periods"]
    selected_periods = available_periods if period_id == "all" else [p for p in available_periods if p["id"] == period_id]
    if not selected_periods:
        raise ValueError(f"未配置评估区间：{period_id}")

    for period in selected_periods:
        missing = [date for date in (period["start"], period["end"]) if date not in date_index]
        if missing:
            raise RuntimeError(f"历史数据缺少区间边界交易日：{missing}；当前数据范围为 {full_dates[0]} 至 {full_dates[-1]}。")

    max_end_index = max(date_index[p["end"]] for p in selected_periods)
    files = files[: max_end_index + 1]
    dates = full_dates[: max_end_index + 1]
    period_specs: list[dict[str, Any]] = []
    for period in selected_periods:
        period_specs.append(
            {
                **period,
                "start_index": date_index[period["start"]],
                "end_index": date_index[period["end"]],
            }
        )

    commission = float(cfg["default_commission_bps_per_side"] if commission_bps_per_side is None else commission_bps_per_side)
    slippage = float(cfg["default_slippage_bps_per_side"] if slippage_bps_per_side is None else slippage_bps_per_side)
    if commission < 0 or slippage < 0:
        raise ValueError("佣金和滑点基点必须大于或等于零。")
    one_way_cost_rate = (commission + slippage) / 10000.0
    if one_way_cost_rate >= 1:
        raise ValueError("单边成本参数超出有效范围。")

    portfolios: list[Portfolio] = []
    for strategy in STRATEGIES:
        for period in period_specs:
            portfolios.append(
                Portfolio(
                    strategy=strategy,
                    period=period,
                    start_index=period["start_index"],
                    end_index=period["end_index"],
                    config=cfg,
                    one_way_cost_rate=one_way_cost_rate,
                    initial_equity=float(cfg["initial_equity"]),
                )
            )

    state = FeatureState()
    previous_signals: dict[str, list[str]] = {strategy["id"]: [] for strategy in STRATEGIES}
    progress_interval = max(len(files) // 20, 1)

    print(f"[进度] 外部策略评估开始，行情文件数：{len(files)}，评估区间数：{len(period_specs)}。", flush=True)
    for day_index, path in enumerate(files):
        date = dates[day_index]
        day = read_daily(path)
        day_indexed = day.set_index("symbol", drop=False)

        # 当日开盘只执行上一交易日收盘产生的排序结果。
        for portfolio in portfolios:
            if day_index <= portfolio.start_index or day_index > portfolio.end_index:
                continue
            signal_index = day_index - 1
            # 只允许未来完整持有窗口落在当前评估区间内。
            if signal_index < portfolio.start_index or signal_index > portfolio.end_index - int(MAX_HOLDING_SESSIONS):
                continue
            portfolio.enter_candidates(previous_signals[portfolio.strategy["id"]], day_indexed, day_index, dates)

        # 只有在买入后至少一个完整交易日，才会检查卖出；当日同时触及止盈止损按止损优先。
        for portfolio in portfolios:
            if portfolio.start_index <= day_index <= portfolio.end_index:
                portfolio.update_exits(day_indexed, day_index, dates)
                portfolio.mark_to_market(day_indexed, day_index, dates)

        # 计算今日收盘可见特征，供下一交易日开盘使用。
        features = state.build(day)
        current_signals: dict[str, list[str]] = {}
        for strategy in STRATEGIES:
            ranked = score_candidates(strategy["id"], features, cfg)
            current_signals[strategy["id"]] = ranked["symbol"].astype(str).str.zfill(6).tolist()
        previous_signals = current_signals

        for portfolio in portfolios:
            if portfolio.start_index <= day_index <= portfolio.end_index - int(MAX_HOLDING_SESSIONS):
                portfolio.signal_days += 1
                if current_signals[portfolio.strategy["id"]]:
                    portfolio.candidate_days += 1

        if day_index % progress_interval == 0 or day_index == len(files) - 1:
            pct = (day_index + 1) / len(files) * 100.0
            print(f"[进度] 已处理 {day_index + 1}/{len(files)} 个交易日文件（{pct:.1f}%），当前日期 {date}。", flush=True)

    results = []
    for strategy in STRATEGIES:
        period_results = []
        for portfolio in (p for p in portfolios if p.strategy["id"] == strategy["id"]):
            metrics = portfolio.result(int(cfg["annualization_trading_days"]))
            period_results.append(metrics)
        results.append(
            {
                **strategy,
                "区间结果": period_results,
            }
        )

    report = {
        "报告版本": 1,
        "生成时间UTC": pd.Timestamp.now(tz="UTC").isoformat(),
        "代码提交": os.environ.get("GITHUB_SHA", "未通过GitHub Actions环境设置"),
        "行情文件数": len(files),
        "实际数据范围": {"开始": dates[0], "结束": dates[-1]},
        "年化交易日数": int(cfg["annualization_trading_days"]),
        "执行成本": {
            "单边佣金基点": commission,
            "单边滑点基点": slippage,
            "单边总成本基点": commission + slippage,
        },
        "交易约束": {
            "信号": "只使用T日收盘及此前可见数据",
            "入场": "T+1开盘",
            "最早退出": "T+2",
            "止盈净收益目标百分比": float(WIN_THRESHOLD_PCT),
            "止损幅度百分比": float(STOP_PCT),
            "最长持有交易日": int(MAX_HOLDING_SESSIONS),
            "同日同时触及止盈止损": "止损优先",
            "最大同时持仓数": int(cfg["max_concurrent_positions"]),
            "最终正式模型是否变更": False,
            "是否接入正式候选池": False,
            "是否允许自动下单": False,
            "未来函数": False,
        },
        "可运行策略": results,
        "暂不可公平复现的策略": NON_REPRODUCIBLE,
        "解释限制": [
            "结果是统一日线与组合约束下的模块实测，不等于公开仓库作者披露的收益。",
            "两种外部规则是近似或代理实现，不能写成原仓库的逐行复现。",
            "历史补充期在其他研究中已经被查看，不是从未触碰的独立留出集。",
            "区间末仍未平仓的持仓按最后可用收盘价盯市并单独列出，不能视为全部已实现利润。",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "报告路径": str(output_path),
        "数据范围": report["实际数据范围"],
        "策略": [
            {
                "名称": strategy["name"],
                "结果": [
                    {
                        "区间": result["区间"],
                        "累计收益百分比": result["累计收益百分比"],
                        "年化复利百分比": result["年化复利百分比"],
                        "最大回撤百分比": result["最大回撤百分比"],
                        "交易数": result["平仓交易数"],
                        "区间末未平仓数": result["区间末未平仓数"],
                    }
                    for result in strategy["区间结果"]
                ],
            }
            for strategy in results
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="统一评估公开A股策略；仅研究用途，不影响正式候选池。")
    parser.add_argument("--period", choices=["all", "development", "validation", "supplementary"], default="all", help="选择完整评估或单个历史区间")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT, help="报告输出路径")
    parser.add_argument("--commission-bps-per-side", type=float, default=None, help="单边佣金基点；默认读取集中配置")
    parser.add_argument("--slippage-bps-per-side", type=float, default=None, help="单边滑点基点；默认读取集中配置")
    args = parser.parse_args()
    run_evaluation(
        period_id=args.period,
        output_path=args.output,
        commission_bps_per_side=args.commission_bps_per_side,
        slippage_bps_per_side=args.slippage_bps_per_side,
    )


if __name__ == "__main__":
    main()
