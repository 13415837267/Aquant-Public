"""2015-now full-market feature mining and short-term 1% hit-rate model research.

This is research only. It never changes the formal production strategy.
The label asks whether a stock entered at T+1 open can reach a net +1% profit
within the following five sessions, using only information available at T.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from collections import deque
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.short_term_research import FeatureState, FEATURE_WARMUP_SESSIONS, history_files, read_daily
from scripts.selection_factor_catalog import active_market_factors, active_stock_factors, load_research_config, transform_market_feature

OUT_DIR = ROOT / "data" / "backtest"

RESEARCH_CONFIG = load_research_config()
STOCK_RANK_FEATURES = active_stock_factors(RESEARCH_CONFIG)
MARKET_FEATURES = active_market_factors(RESEARCH_CONFIG)
FEATURES = STOCK_RANK_FEATURES + MARKET_FEATURES
TRAIN_END = "2022-12-30"
VALIDATION_START = "2023-01-03"
VALIDATION_END = "2024-12-31"
FINAL_START = "2025-01-02"
FINAL_END = "2026-09-30"
NET_WIN_THRESHOLD_PCT = float(RESEARCH_CONFIG["短线目标净收益百分比"])
ROUND_TRIP_COST_BPS = float(RESEARCH_CONFIG["往返交易成本基点"])
MAX_FORWARD_SESSIONS = int(RESEARCH_CONFIG["最大前瞻交易日数"])
MIN_TRAIN_SAMPLES = 5000
MIN_SELECTION_SAMPLES = 5000
LEARNING_RATE = 0.08
L2 = 0.02
EPOCHS = 2
THRESHOLDS = [float(x) for x in RESEARCH_CONFIG["短线模型概率阈值网格"]]


def sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def percentile_rank(series):
    x = pd.to_numeric(series, errors="coerce")
    out = x.rank(method="average", pct=True)
    return out.fillna(0.5).to_numpy(dtype=np.float64)


def make_features(frame):
    if frame.empty:
        return np.empty((0, len(FEATURES)), dtype=np.float64)
    cols = [percentile_rank(frame[name]) for name in STOCK_RANK_FEATURES]
    cols.extend(transform_market_feature(frame, name) for name in MARKET_FEATURES)
    return np.column_stack(cols)


def build_targets(symbols, future_days):
    """按正式管理交易规则生成标签：T+1开盘入场，T+2起退出，先止损则失败。"""
    if len(future_days) < MAX_FORWARD_SESSIONS:
        n = len(symbols)
        return np.zeros(n, dtype=bool), np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool)
    keys = pd.Index(pd.Series(symbols, dtype="string").astype(str).str.zfill(6))
    first = future_days[0].set_index("symbol")
    entry = pd.to_numeric(first["open"], errors="coerce").reindex(keys).to_numpy(dtype=float)
    highs, lows, closes = [], [], []
    for day in future_days[:MAX_FORWARD_SESSIONS]:
        indexed = day.set_index("symbol")
        highs.append(pd.to_numeric(indexed["high"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        lows.append(pd.to_numeric(indexed["low"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        closes.append(pd.to_numeric(indexed["close"], errors="coerce").reindex(keys).to_numpy(dtype=float))
    highs = np.column_stack(highs); lows = np.column_stack(lows); closes = np.column_stack(closes)
    complete = (np.isfinite(entry) & (entry > 0) & np.isfinite(highs).all(axis=1)
                & np.isfinite(lows).all(axis=1) & np.isfinite(closes).all(axis=1))
    target_gross = NET_WIN_THRESHOLD_PCT + ROUND_TRIP_COST_BPS / 100.0
    stop_gross = -float(RESEARCH_CONFIG["止损幅度百分比"])
    target_price = entry * (1.0 + target_gross / 100.0)
    stop_price = entry * (1.0 + stop_gross / 100.0)
    labels = np.zeros(len(keys), dtype=bool)
    best_net = np.full(len(keys), np.nan)
    close_net = np.full(len(keys), np.nan)
    for j in np.flatnonzero(complete):
        outcome = False
        for day_no in range(2, MAX_FORWARD_SESSIONS + 1):
            k = day_no - 1
            hit_stop = lows[j, k] <= stop_price[j]
            hit_target = highs[j, k] >= target_price[j]
            if hit_stop:
                outcome = False
                break
            if hit_target:
                outcome = True
                break
        labels[j] = outcome
        # 研究指标仍记录5日收盘收益，但标签严格服从止盈/止损路径。
        best_net[j] = np.nanmax(highs[j, 1:]) / entry[j] * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
        close_net[j] = closes[j, -1] / entry[j] * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
    return labels, best_net, close_net, complete


class LogisticModel:
    def __init__(self, n_features):
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

    def update(self, x, y):
        if len(y) == 0:
            return
        p = sigmoid(x @ self.w + self.b)
        err = p - y
        self.w -= LEARNING_RATE * ((x.T @ err) / len(y) + L2 * self.w)
        self.b -= LEARNING_RATE * float(err.mean())

    def predict(self, x):
        return sigmoid(x @ self.w + self.b)


def date_split(date):
    if date <= TRAIN_END:
        return "train"
    if VALIDATION_START <= date <= VALIDATION_END:
        return "validation"
    if FINAL_START <= date <= FINAL_END:
        return "final"
    return "ignore"


def stream_dataset(files, start, end, state, mode, model=None, metrics=None, feature_stats=None):
    dates = [p.name[:10] for p in files]
    start_i = dates.index(start)
    end_i = dates.index(end)
    cache = {}
    def get(i):
        if i not in cache:
            cache[i] = read_daily(files[i])
        while len(cache) > MAX_FORWARD_SESSIONS + 1:
            del cache[next(iter(cache))]
        return cache[i]

    processed = 0
    samples = 0
    for i in range(max(0, start_i - FEATURE_WARMUP_SESSIONS), end_i + 1):
        date = dates[i]
        if date < start or date > end:
            state.build(get(i))
            continue
        frame = state.build(get(i))
        if i + MAX_FORWARD_SESSIONS >= len(files):
            continue
        futures = [get(i + j) for j in range(1, MAX_FORWARD_SESSIONS + 1)]
        if frame.empty:
            continue
        x = make_features(frame)
        labels, best_net, close_net, complete = build_targets(
            frame["symbol"].astype(str).str.zfill(6).tolist(), futures
        )
        keep = np.flatnonzero(complete)
        if len(keep) == 0:
            continue
        x = x[keep]
        y = labels[keep].astype(np.float64)
        best = best_net[keep]
        close = close_net[keep]
        samples += len(y)
        processed += 1

        if mode == "train":
            for _ in range(EPOCHS):
                model.update(x, y)
            if feature_stats is not None:
                feature_stats["positives"] += int(y.sum())
                feature_stats["samples"] += int(len(y))
                feature_stats["best_returns"].extend(best[::max(1, len(best)//500)])
        elif mode == "evaluate":
            p = model.predict(x)
            metrics.append((date, p, y, best, close))
        if processed % 50 == 0:
            print(f"[特征训练] {mode} 已处理交易日={processed} 样本={samples}", flush=True)
    return processed, samples


def evaluate(metrics):
    rows = []
    all_probs = []
    all_y = []
    all_best = []
    all_close = []
    for date, p, y, best, close in metrics:
        all_probs.append(p); all_y.append(y); all_best.append(best); all_close.append(close)
    if not all_probs:
        return {"samples": 0, "thresholds": [], "daily": []}
    p = np.concatenate(all_probs); y = np.concatenate(all_y)
    best = np.concatenate(all_best); close = np.concatenate(all_close)
    for threshold in THRESHOLDS:
        mask = p >= threshold
        n = int(mask.sum())
        rows.append({
            "probability_threshold": threshold,
            "samples": n,
            "sample_share_pct": float(n / len(y) * 100),
            "hit_1pct_rate_pct": float(y[mask].mean() * 100) if n else None,
            "mean_best_return_pct": float(best[mask].mean()) if n else None,
            "mean_5d_close_return_pct": float(close[mask].mean()) if n else None,
        })
    daily = []
    for date, pp, yy, bb, cc in metrics:
        daily.append({
            "date": date,
            "samples": int(len(yy)),
            "base_hit_1pct_rate_pct": float(yy.mean() * 100),
            "base_mean_best_return_pct": float(bb.mean()),
            "base_mean_5d_close_return_pct": float(cc.mean()),
            "model_mean_probability": float(pp.mean()),
        })
    return {
        "samples": int(len(y)),
        "base_hit_1pct_rate_pct": float(y.mean() * 100),
        "base_mean_best_return_pct": float(best.mean()),
        "base_mean_5d_close_return_pct": float(close.mean()),
        "thresholds": rows,
        "daily": daily,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(OUT_DIR / "feature_training_latest.json"))
    ap.add_argument("--start", default="2015-01-05")
    ap.add_argument("--final-end", default=FINAL_END)
    args = ap.parse_args()
    started = time.time()
    files = history_files()
    dates = [p.name[:10] for p in files]
    if args.start not in dates or args.final_end not in dates:
        raise ValueError("训练区间必须落在历史交易日文件范围内")

    state = FeatureState()
    model = LogisticModel(len(FEATURES))
    feature_stats = {"samples": 0, "positives": 0, "best_returns": []}

    train_processed, train_samples = stream_dataset(
        files, args.start, TRAIN_END, state, "train", model=model, feature_stats=feature_stats
    )
    if train_samples < MIN_TRAIN_SAMPLES:
        raise RuntimeError(f"训练样本不足: {train_samples}")

    validation_metrics = []
    state = FeatureState()
    validation_processed, validation_samples = stream_dataset(
        files, VALIDATION_START, VALIDATION_END, state, "evaluate", model=model, metrics=validation_metrics
    )

    final_metrics = []
    state = FeatureState()
    final_processed, final_samples = stream_dataset(
        files, FINAL_START, args.final_end, state, "evaluate", model=model, metrics=final_metrics
    )

    validation = evaluate(validation_metrics)
    final = evaluate(final_metrics)

    selected = None
    for row in validation["thresholds"]:
        if row["samples"] >= MIN_SELECTION_SAMPLES and row["hit_1pct_rate_pct"] is not None and row["hit_1pct_rate_pct"] >= 80.0:
            selected = row
            break
    if selected is None:
        eligible = [r for r in validation["thresholds"] if r["samples"] >= MIN_SELECTION_SAMPLES]
        selected = max(eligible, key=lambda r: (r["hit_1pct_rate_pct"] or -1, r["samples"])) if eligible else None

    result = {
        "schema_version": 1,
        "status": "research_only",
        "method": "full_market_streaming_logistic_feature_research",
        "objective": "net_profit_at_least_1pct_within_5_sessions",
        "data_start": args.start,
        "data_end": args.final_end,
        "splits": {
            "train": [args.start, TRAIN_END],
            "validation": [VALIDATION_START, VALIDATION_END],
            "final": [FINAL_START, args.final_end],
        },
        "features": FEATURES,
        "stock_features_are_cross_sectional_percentile_ranked": True,
        "net_win_threshold_pct": NET_WIN_THRESHOLD_PCT,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "min_selection_samples": MIN_SELECTION_SAMPLES,
        "model": {
            "type": "logistic_regression_sgd",
            "learning_rate": LEARNING_RATE,
            "l2": L2,
            "epochs_per_day": EPOCHS,
            "intercept": float(model.b),
            "coefficients": {name: float(value) for name, value in zip(FEATURES, model.w)},
        },
        "train": {
            "processed_days": train_processed,
            "samples": train_samples,
            "positive_samples": feature_stats["positives"],
            "base_hit_1pct_rate_pct": feature_stats["positives"] / train_samples * 100,
        },
        "validation": validation,
        "final": final,
        "selected_validation_operating_point": selected,
        "audit": {
            "no_future_features": True,
            "label_uses_T_plus_1_open_through_T_plus_5_high_close": True,
            "recent_incomplete_windows_excluded": True,
            "formal_production_changed": False,
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "final_samples": final_samples,
        "selected_validation_operating_point": selected,
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
