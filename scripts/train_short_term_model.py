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

import numpy as np
import pandas as pd

from scripts.short_term_research import FeatureState, history_files, read_daily

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "backtest"

FEATURES = [
    "return_1d_pct", "return_3d_pct", "return_5d_pct", "return_10d_pct",
    "return_20d_pct", "overnight_1d_pct", "overnight_3d_pct",
    "overnight_5d_pct", "overnight_10d_pct", "volume_ratio_5d",
    "amount_20d", "volatility_10d_pct", "close_strength",
    "intraday_return_pct", "limit_up_5d_count", "turnover_pct",
    "change_pct", "market_breadth_pct", "market_median_return_pct",
]
STOCK_RANK_FEATURES = FEATURES[:16] + ["change_pct"]
MARKET_FEATURES = ["market_breadth_pct", "market_median_return_pct"]
TRAIN_END = "2022-12-30"
VALIDATION_START = "2023-01-03"
VALIDATION_END = "2024-12-31"
FINAL_START = "2025-01-02"
FINAL_END = "2026-09-30"
NET_WIN_THRESHOLD_PCT = 1.0
ROUND_TRIP_COST_BPS = 10.0
MAX_FORWARD_SESSIONS = 5
MIN_TRAIN_SAMPLES = 5000
MIN_SELECTION_SAMPLES = 5000
LEARNING_RATE = 0.08
L2 = 0.02
EPOCHS = 2
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


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
    cols = []
    for name in STOCK_RANK_FEATURES:
        cols.append(percentile_rank(frame[name]))
    breadth = np.clip(pd.to_numeric(frame["market_breadth_pct"], errors="coerce").fillna(50).to_numpy(dtype=float) / 100.0, 0, 1)
    median_ret = np.clip(pd.to_numeric(frame["market_median_return_pct"], errors="coerce").fillna(0).to_numpy(dtype=float) / 5.0, -2, 2)
    cols.extend([breadth, median_ret])
    return np.column_stack(cols)


def future_label(symbol, future_days):
    if not future_days:
        return None
    first = future_days[0]
    row = first.loc[first["symbol"].eq(symbol)]
    if row.empty or pd.isna(row.iloc[0].get("open")):
        return None
    entry = float(row.iloc[0]["open"])
    if not np.isfinite(entry) or entry <= 0:
        return None
    highs = []
    closes = []
    for day in future_days[:MAX_FORWARD_SESSIONS]:
        r = day.loc[day["symbol"].eq(symbol)]
        if r.empty:
            continue
        hi = pd.to_numeric(r.iloc[0].get("high"), errors="coerce")
        cl = pd.to_numeric(r.iloc[0].get("close"), errors="coerce")
        if pd.notna(hi):
            highs.append(float(hi))
        if pd.notna(cl):
            closes.append(float(cl))
    if len(highs) < MAX_FORWARD_SESSIONS or len(closes) < MAX_FORWARD_SESSIONS:
        return None
    gross_best = max(highs) / entry - 1.0
    net_best_pct = gross_best * 100.0 - ROUND_TRIP_COST_BPS / 100.0
    net_close_pct = closes[-1] / entry * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
    return int(net_best_pct >= NET_WIN_THRESHOLD_PCT), net_best_pct, net_close_pct


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
    for i in range(max(20, start_i - 1), end_i + 1):
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
        ys, bests, closes, keep = [], [], [], []
        for row_idx, symbol in enumerate(frame["symbol"].astype(str).str.zfill(6)):
            label = future_label(symbol, futures)
            if label is None:
                continue
            keep.append(row_idx)
            ys.append(label[0]); bests.append(label[1]); closes.append(label[2])
        if not keep:
            continue
        x = x[np.asarray(keep)]
        y = np.asarray(ys, dtype=np.float64)
        best = np.asarray(bests, dtype=np.float64)
        close = np.asarray(closes, dtype=np.float64)
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
