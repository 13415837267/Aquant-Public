"""筛选可执行上涨样本并分析其信号日量价特征；本脚本不训练模型。"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.short_term_research import FeatureState, executable_entry_mask, history_files, read_daily
from scripts.selection_factor_catalog import (
    BASE_STOCK_FACTORS,
    FACTOR_GROUPS,
    MARKET_FACTOR_GROUP,
    active_market_factors,
    active_stock_factors,
    load_research_config,
)

CONFIG_PATH = ROOT / "config" / "上涨样本分析配置.json"
RESEARCH_CONFIG_PATH = ROOT / "config" / "选股条件研究配置.json"
OUT_DIR = ROOT / "data" / "backtest"
OHLC = ("open", "high", "low", "close")
EXAMPLE_FACTORS = (
    "return_5d_pct", "return_20d_pct", "volume_ratio_5d", "amount_ratio_20d",
    "close_vs_ma20_pct", "ma5_slope_5d_pct", "breakout_20d_pct",
    "close_strength", "rsi_14", "macd_histogram_pct", "market_breadth_pct",
)


def get_feature_names(research_config: dict) -> list[str]:
    return active_stock_factors(research_config) + active_market_factors(research_config)


def load_attribution_config(feature_names: list[str]) -> dict:
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"上涨样本分析配置无法读取：{exc}") from exc
    if not isinstance(config, dict) or config.get("版本") != 1:
        raise ValueError("上涨样本分析配置格式或版本不正确")

    targets = config.get("上涨净收益目标百分比")
    if not isinstance(targets, list) or not targets:
        raise ValueError("上涨净收益目标百分比必须是非空列表")
    targets = [float(value) for value in targets]
    if targets != sorted(set(targets)) or any(not math.isfinite(value) or value <= 0 for value in targets):
        raise ValueError("上涨净收益目标必须为不重复、递增的正数")

    raw_windows = config.get("研究分段", {})
    required = ("发现期", "验证期", "最终留出期")
    if any(name not in raw_windows for name in required):
        raise ValueError("研究分段必须包含发现期、验证期和最终留出期")
    windows = {}
    previous_end = ""
    for name in required:
        bounds = raw_windows[name]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"研究分段“{name}”必须包含起止日期")
        start, end = map(str, bounds)
        pd.Timestamp(start)
        pd.Timestamp(end)
        if start > end or (previous_end and start <= previous_end):
            raise ValueError("研究分段必须按时间严格递增且不得重叠")
        windows[name] = (start, end)
        previous_end = end

    for name in ("特征差异排名数量", "每组上涨样本示例数量", "检查点间隔交易日"):
        value = config.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"配置项“{name}”必须为正整数")
    display = config.get("样本明细展示因子", [])
    if not isinstance(display, list) or any(name not in feature_names for name in display):
        raise ValueError("样本明细展示因子必须来自当前研究特征清单")
    return {**config, "_targets": targets, "_windows": windows, "_display": display}


def assign_period(date: str, windows: dict) -> str | None:
    for name, (start, end) in windows.items():
        if start <= date <= end:
            return name
    return None


def factor_group(name: str) -> str:
    if name in BASE_STOCK_FACTORS:
        return "基础量价"
    for group, names in FACTOR_GROUPS.items():
        if name in names:
            return group
    return MARKET_FACTOR_GROUP.get(name, "其他因子")


class FeatureMoments:
    """按因子累计有效值数量、均值与方差，避免持有整个市场的原始矩阵。"""

    def __init__(self, feature_count: int):
        self.sample_count = 0
        self.count = np.zeros(feature_count, dtype=np.int64)
        self.total = np.zeros(feature_count, dtype=np.float64)
        self.square = np.zeros(feature_count, dtype=np.float64)

    def update(self, matrix: np.ndarray) -> None:
        if matrix.size == 0:
            return
        valid = np.isfinite(matrix)
        values = np.where(valid, matrix, 0.0)
        self.sample_count += matrix.shape[0]
        self.count += valid.sum(axis=0)
        self.total += values.sum(axis=0)
        self.square += (values * values).sum(axis=0)

    def summary(self) -> dict:
        mean = np.full(len(self.total), np.nan)
        variance = np.full(len(self.total), np.nan)
        good = self.count > 0
        mean[good] = self.total[good] / self.count[good]
        second = np.zeros_like(self.total)
        second[good] = self.square[good] / self.count[good]
        variance[good] = np.maximum(second[good] - mean[good] ** 2, 0.0)
        return {"mean": mean, "variance": variance, "count": self.count.copy()}


class OutcomeStats:
    def __init__(self, feature_count: int):
        self.entries = self.resolved = self.unresolved = self.winners = 0
        self.return_sum = 0.0
        self.positive_returns = 0
        self.exit_reasons = Counter()
        self.win_features = FeatureMoments(feature_count)
        self.other_features = FeatureMoments(feature_count)

    def update(self, matrix: np.ndarray, outcomes: list[dict]) -> None:
        self.entries += len(outcomes)
        resolved_mask = np.asarray([item["resolved"] for item in outcomes], dtype=bool)
        self.unresolved += int((~resolved_mask).sum())
        if not resolved_mask.any():
            return
        resolved = [outcomes[index] for index in np.flatnonzero(resolved_mask)]
        resolved_matrix = matrix[resolved_mask]
        wins = np.asarray([item["reason"] == "止盈" for item in resolved], dtype=bool)
        returns = np.asarray([item["net_return_pct"] for item in resolved], dtype=np.float64)
        self.resolved += len(resolved)
        self.winners += int(wins.sum())
        self.return_sum += float(returns.sum())
        self.positive_returns += int((returns > 0).sum())
        self.exit_reasons.update(item["reason"] for item in resolved)
        self.win_features.update(resolved_matrix[wins])
        self.other_features.update(resolved_matrix[~wins])

    def report(self) -> dict:
        return {
            "可执行入场样本数": self.entries,
            "结果可确认样本数": self.resolved,
            "结果无法确认样本数": self.unresolved,
            "上涨目标命中样本数": self.winners,
            "上涨目标命中率百分比": round(self.winners / self.resolved * 100, 4) if self.resolved else None,
            "正净收益样本比例百分比": round(self.positive_returns / self.resolved * 100, 4) if self.resolved else None,
            "平均实际净收益百分比": round(self.return_sum / self.resolved, 4) if self.resolved else None,
            "退出原因统计": dict(sorted(self.exit_reasons.items())),
        }


def simulate_target_path(
    entry_price: float,
    bars: list[dict],
    target_net_pct: float,
    stop_loss_pct: float,
    round_trip_cost_bps: float,
    earliest_exit_day: int = 2,
) -> dict:
    """T+1开盘入场、T+2起允许退出；同日同时触价按止损优先。"""
    if not math.isfinite(entry_price) or entry_price <= 0:
        return {"resolved": False, "reason": "入场价格无效", "net_return_pct": None}
    cost_pct = round_trip_cost_bps / 100.0
    stop_price = entry_price * (1.0 - stop_loss_pct / 100.0)
    target_price = entry_price * (1.0 + (target_net_pct + cost_pct) / 100.0)
    last_close, last_date = None, None

    if bars and bars[0].get("close") is not None:
        try:
            value = float(bars[0]["close"])
            if math.isfinite(value) and value > 0:
                last_close, last_date = value, bars[0].get("date")
        except (TypeError, ValueError):
            pass

    for day_number, bar in enumerate(bars, start=1):
        if day_number < earliest_exit_day:
            continue
        try:
            if float(bar.get("is_paused", 0) or 0) > 0:
                continue
        except (TypeError, ValueError):
            pass
        values = {}
        for field in OHLC:
            try:
                values[field] = float(bar[field])
            except (KeyError, TypeError, ValueError):
                values = {}
                break
        if len(values) != len(OHLC) or any(
            not math.isfinite(value) or value <= 0 for value in values.values()
        ):
            continue

        open_price, high, low, close = (values[name] for name in OHLC)
        last_close, last_date = close, bar.get("date")
        if low <= stop_price:
            fill = open_price if open_price < stop_price else stop_price
            return {
                "resolved": True,
                "reason": "止损",
                "exit_date": bar.get("date"),
                "entry_price": entry_price,
                "exit_price": fill,
                "net_return_pct": (fill / entry_price - 1.0) * 100.0 - cost_pct,
            }
        if open_price >= target_price:
            fill = open_price
        elif high >= target_price:
            fill = target_price
        else:
            fill = None
        if fill is not None:
            return {
                "resolved": True,
                "reason": "止盈",
                "exit_date": bar.get("date"),
                "entry_price": entry_price,
                "exit_price": fill,
                "net_return_pct": (fill / entry_price - 1.0) * 100.0 - cost_pct,
            }
        if day_number == len(bars):
            return {
                "resolved": True,
                "reason": "到期",
                "exit_date": bar.get("date"),
                "entry_price": entry_price,
                "exit_price": close,
                "net_return_pct": (close / entry_price - 1.0) * 100.0 - cost_pct,
            }

    marked_return = (last_close / entry_price - 1.0) * 100.0 - cost_pct if last_close else None
    return {
        "resolved": False,
        "reason": "期末仍持仓，按最后有效收盘估值",
        "exit_date": last_date,
        "entry_price": entry_price,
        "exit_price": last_close,
        "net_return_pct": marked_return,
    }


def aligned_future_days(future: list[tuple[str, pd.DataFrame]], symbols: list[str]) -> list[dict]:
    result = []
    target_index = pd.Index(symbols)
    for date, day in future:
        aligned = day.set_index("symbol", drop=False).reindex(target_index)
        values = {"date": date}
        for field in OHLC:
            values[field] = (
                pd.to_numeric(aligned[field], errors="coerce").to_numpy(dtype=np.float64)
                if field in aligned else np.full(len(symbols), np.nan, dtype=np.float64)
            )
        values["is_paused"] = (
            pd.to_numeric(aligned["is_paused"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
            if "is_paused" in aligned else np.zeros(len(symbols), dtype=np.float64)
        )
        result.append(values)
    return result


def study_signature(files: list[Path]) -> str:
    """把研究脚本、配置和行情原始内容纳入签名，拒绝不兼容断点。"""
    digest = hashlib.sha256()
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts" / "short_term_research.py",
        ROOT / "scripts" / "selection_factor_catalog.py",
        CONFIG_PATH,
        RESEARCH_CONFIG_PATH,
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    for number, path in enumerate(files, start=1):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if number % 500 == 0:
            print(f"[签名校验] 已核对历史行情文件 {number}/{len(files)}", flush=True)
    return digest.hexdigest()


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def restore_checkpoint(path: Path, signature: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as stream:
            saved = pickle.load(stream)
        if not isinstance(saved, dict) or saved.get("signature") != signature:
            print("[断点恢复] 代码、配置或历史行情变化，丢弃不兼容检查点。", flush=True)
            path.unlink(missing_ok=True)
            return None
        if not isinstance(saved.get("next_index"), int) or saved["next_index"] < 0:
            raise ValueError("检查点位置无效")
        print(
            f"[断点恢复] 从历史行情位置 {saved['next_index']} 继续，"
            f"此前已处理交易日={saved['processed_days']}",
            flush=True,
        )
        return saved
    except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError) as exc:
        print(f"[断点恢复] 检查点不可读，丢弃后重建：{exc}", flush=True)
        path.unlink(missing_ok=True)
        return None


def standardized_effect(wins: dict, controls: dict, index: int) -> dict:
    if wins["count"][index] == 0 or controls["count"][index] == 0:
        return {
            "上涨样本均值": None,
            "对照样本均值": None,
            "标准化均值差": None,
            "上涨样本均值更高": None,
        }
    mean_win = float(wins["mean"][index])
    mean_control = float(controls["mean"][index])
    pooled_std = math.sqrt(max(
        0.0,
        (float(wins["variance"][index]) + float(controls["variance"][index])) / 2.0,
    ))
    effect = (mean_win - mean_control) / pooled_std if pooled_std > 1e-12 else 0.0
    return {
        "上涨样本均值": round(mean_win, 6),
        "对照样本均值": round(mean_control, 6),
        "标准化均值差": round(effect, 6),
        "上涨样本均值更高": bool(effect > 0),
    }


def feature_effects(accumulator: OutcomeStats, names: list[str], limit: int) -> list[dict]:
    wins, controls = accumulator.win_features.summary(), accumulator.other_features.summary()
    rows = []
    for index, name in enumerate(names):
        effect = standardized_effect(wins, controls, index)
        if effect["标准化均值差"] is not None:
            rows.append({
                "因子": name,
                "类别": factor_group(name),
                "上涨样本数": int(wins["count"][index]),
                "对照样本数": int(controls["count"][index]),
                **effect,
            })
    rows.sort(key=lambda row: abs(row["标准化均值差"]), reverse=True)
    return rows[:limit]


def factor_stability(stats: dict, targets: list[float], names: list[str], limit: int) -> list[dict]:
    report = []
    for target in targets:
        top = feature_effects(stats[("发现期", target)], names, limit)
        checks = []
        for row in top:
            feature = row["因子"]
            feature_index = names.index(feature)
            direction = row["标准化均值差"] > 0
            item = {"因子": feature, "发现期标准化均值差": row["标准化均值差"]}
            flags = []
            for period in ("验证期", "最终留出期"):
                effect = standardized_effect(
                    stats[(period, target)].win_features.summary(),
                    stats[(period, target)].other_features.summary(),
                    feature_index,
                )["标准化均值差"]
                same = effect is not None and abs(effect) > 1e-10 and ((effect > 0) == direction)
                item[period + "标准化均值差"] = round(effect, 6) if effect is not None else None
                item[period + "方向一致"] = bool(same)
                flags.append(bool(same))
            item["验证期与最终留出期方向均一致"] = all(flags)
            checks.append(item)
        report.append({
            "上涨净收益目标百分比": target,
            "发现期最显著因子": top,
            "跨时间分段方向稳定性": checks,
            "解释": "单因子差异属于历史关联性描述，不代表因果关系或已通过验收的选股规则。",
        })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="先筛选历史上涨样本，再分析其信号日前量价因子")
    parser.add_argument("--output", default=str(OUT_DIR / "uptrend_stock_attribution_latest.json"))
    parser.add_argument(
        "--checkpoint-path",
        default=str(ROOT / "data" / "research" / "checkpoints" / "uptrend_stock_attribution.pkl"),
    )
    parser.add_argument("--checkpoint-every-days", type=int, default=0)
    args = parser.parse_args()

    started = time.time()
    research_config = load_research_config()
    feature_names = get_feature_names(research_config)
    config = load_attribution_config(feature_names)
    files = history_files()
    dates = [path.name[:10] for path in files]
    required_dates = [
        bound for pair in config["_windows"].values() for bound in pair
    ]
    missing = [day for day in required_dates if day not in set(dates)]
    if missing:
        raise ValueError(f"历史行情缺少研究分段边界日期：{missing}")

    max_hold = int(research_config["最大前瞻交易日数"])
    earliest_exit = int(research_config["最早允许退出日序号"])
    stop_pct = float(research_config["止损幅度百分比"])
    cost_bps = float(research_config["往返交易成本基点"])
    targets, windows = config["_targets"], config["_windows"]
    checkpoint_every = args.checkpoint_every_days or int(config["检查点间隔交易日"])
    checkpoint_path, output_path = Path(args.checkpoint_path), Path(args.output)

    print(f"[上涨样本分析] 历史文件数={len(files)}，数据范围={dates[0]}至{dates[-1]}", flush=True)
    print(f"[上涨样本分析] 上涨目标={targets}%，止损={stop_pct}%，往返成本={cost_bps}基点", flush=True)
    print("[上涨样本分析] 只做样本筛选与特征归因，不训练模型。", flush=True)
    signature = study_signature(files)
    saved = restore_checkpoint(checkpoint_path, signature)
    if saved:
        state, stats, meta = saved["state"], saved["stats"], saved["meta"]
        heaps = saved["heaps"]
        next_index, processed = saved["next_index"], saved["processed_days"]
        sequence = meta.get("_sequence", 0)
    else:
        state = FeatureState()
        stats = {
            (period, target): OutcomeStats(len(feature_names))
            for period in windows for target in targets
        }
        meta = {
            period: {
                "信号日数": 0,
                "因子候选样本数": 0,
                "可执行入场样本数": 0,
                "因标签窗口跨分段而跳过的信号日数": 0,
                "最早信号日期": None,
                "最晚信号日期": None,
            }
            for period in windows
        }
        heaps = {(period, target): [] for period in windows for target in targets}
        next_index, processed, sequence = 0, 0, 0

    display_names = config["_display"] or [name for name in EXAMPLE_FACTORS if name in feature_names]
    example_limit = int(config["每组上涨样本示例数量"])
    top_limit = int(config["特征差异排名数量"])

    for i in range(next_index, max(0, len(files) - max_hold)):
        signal_date = dates[i]
        period = assign_period(signal_date, windows)
        frame = state.build(read_daily(files[i]))
        processed += 1

        if period is None:
            if processed % checkpoint_every == 0:
                save_checkpoint(checkpoint_path, {
                    "signature": signature, "next_index": i + 1, "processed_days": processed,
                    "state": state, "stats": stats, "meta": meta, "heaps": heaps,
                })
            continue

        period_meta = meta[period]
        period_meta["信号日数"] += 1
        period_meta["最早信号日期"] = period_meta["最早信号日期"] or signal_date
        period_meta["最晚信号日期"] = signal_date
        future_dates = dates[i + 1:i + max_hold + 1]
        if len(future_dates) != max_hold or assign_period(future_dates[-1], windows) != period:
            period_meta["因标签窗口跨分段而跳过的信号日数"] += 1
            if processed % checkpoint_every == 0:
                save_checkpoint(checkpoint_path, {
                    "signature": signature, "next_index": i + 1, "processed_days": processed,
                    "state": state, "stats": stats, "meta": meta, "heaps": heaps,
                })
            continue
        if frame.empty:
            if processed % checkpoint_every == 0:
                save_checkpoint(checkpoint_path, {
                    "signature": signature, "next_index": i + 1, "processed_days": processed,
                    "state": state, "stats": stats, "meta": meta, "heaps": heaps,
                })
            continue

        period_meta["因子候选样本数"] += len(frame)
        symbols = frame["symbol"].astype(str).tolist()
        future = [
            (dates[i + offset], read_daily(files[i + offset]))
            for offset in range(1, max_hold + 1)
        ]
        entry_mask = executable_entry_mask(symbols, future[0][1])
        entry_indices = np.flatnonzero(entry_mask)
        period_meta["可执行入场样本数"] += len(entry_indices)
        if len(entry_indices):
            matrix_all = frame.loc[:, feature_names].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=np.float64)
            matrix = matrix_all[entry_indices]
            candidate_symbols = [symbols[j] for j in entry_indices]
            entry_frame = future[0][1].set_index("symbol")
            entry_prices = pd.to_numeric(
                entry_frame["open"].reindex(candidate_symbols), errors="coerce"
            ).to_numpy(dtype=np.float64)
            aligned = aligned_future_days(future, candidate_symbols)
            outcomes_by_target = {target: [] for target in targets}
            winners_by_target = {target: [] for target in targets}

            for local_index, original_index in enumerate(entry_indices):
                entry_price = (
                    float(entry_prices[local_index])
                    if np.isfinite(entry_prices[local_index]) else float("nan")
                )
                bars = []
                for day_values in aligned:
                    bar = {
                        "date": day_values["date"],
                        "is_paused": float(day_values["is_paused"][local_index]),
                    }
                    for field in OHLC:
                        value = float(day_values[field][local_index])
                        bar[field] = value if np.isfinite(value) else None
                    bars.append(bar)

                for target in targets:
                    outcome = simulate_target_path(
                        entry_price, bars, target, stop_pct, cost_bps,
                        earliest_exit_day=earliest_exit,
                    )
                    outcomes_by_target[target].append(outcome)
                    if outcome["resolved"] and outcome["reason"] == "止盈":
                        sequence += 1
                        row = frame.iloc[original_index]
                        example = {
                            "信号日期": signal_date,
                            "证券代码": str(row["symbol"]),
                            "证券名称": str(row.get("name", row["symbol"])),
                            "入场日期": future_dates[0],
                            "目标净收益百分比": target,
                            "实际退出日期": outcome.get("exit_date"),
                            "入场价格": round(entry_price, 4),
                            "退出价格": round(float(outcome["exit_price"]), 4),
                            "实际净收益百分比": round(float(outcome["net_return_pct"]), 4),
                            "信号日可见因子": {
                                name: (
                                    round(float(row[name]), 6)
                                    if pd.notna(row.get(name)) and math.isfinite(float(row[name]))
                                    else None
                                )
                                for name in display_names
                            },
                        }
                        winners_by_target[target].append((
                            float(outcome["net_return_pct"]), sequence, example
                        ))

            for target in targets:
                stats[(period, target)].update(matrix, outcomes_by_target[target])
                heap = heaps[(period, target)]
                for example in winners_by_target[target]:
                    if len(heap) < example_limit:
                        heapq.heappush(heap, example)
                    elif example[0] > heap[0][0]:
                        heapq.heapreplace(heap, example)

        meta["_sequence"] = sequence
        if processed % 25 == 0:
            print(
                f"[上涨样本分析] 已处理交易日={processed}/{len(files)-max_hold}，"
                f"日期={signal_date}，分段={period}，"
                f"可执行入场累计={period_meta['可执行入场样本数']}",
                flush=True,
            )
        if processed % checkpoint_every == 0:
            save_checkpoint(checkpoint_path, {
                "signature": signature, "next_index": i + 1, "processed_days": processed,
                "state": state, "stats": stats, "meta": meta, "heaps": heaps,
            })

    period_reports = []
    for period in windows:
        for target in targets:
            accumulator = stats[(period, target)]
            period_reports.append({
                "研究分段": period,
                "上涨净收益目标百分比": target,
                "分段概况": {
                    key: value for key, value in meta[period].items()
                    if not key.startswith("_")
                },
                "上涨路径结果": accumulator.report(),
                "上涨样本与对照样本因子差异排名": feature_effects(
                    accumulator, feature_names, top_limit
                ),
                "历史上涨样本示例": [
                    item[2] for item in sorted(heaps[(period, target)], reverse=True)
                ],
            })

    output = {
        "状态": "研究完成，仅用于历史上涨样本归因",
        "研究方法": "历史上涨样本筛选、信号日量价特征差异、跨时间分段方向稳定性检查",
        "是否训练模型": False,
        "数据范围": {
            "历史行情起始日期": dates[0],
            "历史行情结束日期": dates[-1],
            "最大未来观察交易日数": max_hold,
            "信号特征截止时点": "信号日收盘",
            "结果标签规则": "T+1开盘可执行入场、T+2起止盈止损、最多观察配置的交易日数",
            "发现期": list(windows["发现期"]),
            "验证期": list(windows["验证期"]),
            "最终留出期": list(windows["最终留出期"]),
            "最终留出期说明": "此前系统其他研究已使用该时间段；本次作为跨期复核，不称全新独立留出。",
        },
        "研究参数": {
            "上涨净收益目标百分比": targets,
            "止损幅度百分比": stop_pct,
            "往返交易成本基点": cost_bps,
            "最大持有交易日数": max_hold,
            "最早允许退出日序号": earliest_exit,
            "因子数": len(feature_names),
        },
        "各分段结果": period_reports,
        "发现期因子方向稳定性复核": factor_stability(stats, targets, feature_names, top_limit),
        "解释限制": [
            "上涨样本由未来真实交易路径事后识别，只用于解释历史行情，不能作为当日候选的输入。",
            "归因因子只取信号日收盘时可见数据；未来价格只用于构造结果标签。",
            "单因子差异是关联性描述，不代表因果关系；仅在后续分段重复出现时才值得进一步验证。",
            "无法确认退出的持仓单独计数，不把停牌或缺失行情伪造成平仓。",
        ],
        "审计": {
            "信号特征只使用截至信号日收盘的数据": True,
            "未来行情只用于事后结果标签": True,
            "不同研究分段的结果标签不跨界": True,
            "T加1开盘入场": True,
            "T加2起允许退出": True,
            "同日日线止损优先": True,
            "过滤开盘涨停与停牌入场": True,
            "未确认退出不强行平仓": True,
            "没有训练模型或搜索阈值": True,
            "没有修改正式生产模型或候选池规则": True,
            "正式生产模型变更": False,
        },
        "运行信息": {
            "已处理信号日": processed,
            "耗时秒数": round(time.time() - started, 2),
            "检查点支持原子保存与签名恢复": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(output_path)
    checkpoint_path.unlink(missing_ok=True)

    summary = [
        {
            "研究分段": item["研究分段"],
            "上涨净收益目标百分比": item["上涨净收益目标百分比"],
            "上涨路径结果": item["上涨路径结果"],
            "最显著事前因子": item["上涨样本与对照样本因子差异排名"][:5],
        }
        for item in period_reports
    ]
    print("===== 历史上涨样本筛选与归因结果 =====", flush=True)
    print(json.dumps({
        "数据范围": output["数据范围"],
        "各分段结果": summary,
        "正式生产模型变更": False,
    }, ensure_ascii=False, indent=2))
    print(f"[上涨样本分析] 结果已保存：{output_path}", flush=True)


if __name__ == "__main__":
    main()
