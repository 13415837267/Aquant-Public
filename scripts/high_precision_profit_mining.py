"""High-precision short-term profit mining under the user's +1% win definition.

Research only. Train on 2015-2022, select an operating point on 2023-2024,
then evaluate once on 2025-2026-09-30. A win means net best return reaches
at least +1% within five sessions after T+1 open.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.short_term_research import FeatureState, history_files, read_daily
from scripts.train_short_term_model import (
    FEATURES,
    LogisticModel,
    build_targets,
    make_features,
)

OUT_DIR = ROOT / "data" / "backtest"

TRAIN_END = "2022-12-30"
VALIDATION_START = "2023-01-03"
VALIDATION_END = "2024-12-31"
FINAL_START = "2025-01-02"
FINAL_END = "2026-09-30"

NET_WIN_THRESHOLD_PCT = 1.0
ROUND_TRIP_COST_BPS = 10.0
MAX_FORWARD_SESSIONS = 5

HIGH_PRECISION_THRESHOLDS = (
    0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93,
    0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
)
MIN_SELECTION_SAMPLES = 100
TOP_K_PER_DAY = (1, 2, 3, 5, 10)
MIN_DAILY_TOPK_DAYS = 50


def wilson_lower_bound(wins: int, samples: int, z: float = 1.96) -> float:
    if samples <= 0:
        return 0.0
    p = wins / samples
    denom = 1.0 + z * z / samples
    center = p + z * z / (2.0 * samples)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * samples)) / samples)
    return (center - spread) / denom


def collect_scored(files, start, end, state, model):
    dates = [p.name[:10] for p in files]
    start_i, end_i = dates.index(start), dates.index(end)
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = read_daily(files[i])
        while len(cache) > MAX_FORWARD_SESSIONS + 1:
            del cache[next(iter(cache))]
        return cache[i]

    metrics = []
    processed = samples = 0

    for i in range(max(0, start_i - 20), end_i + 1):
        date = dates[i]
        frame = state.build(get(i))
        if date < start or date > end:
            continue
        if i + MAX_FORWARD_SESSIONS >= len(files) or frame.empty:
            continue

        futures = [get(i + j) for j in range(1, MAX_FORWARD_SESSIONS + 1)]
        x_all = make_features(frame)
        labels, best_net, close_net, complete = build_targets(
            frame["symbol"].astype(str).str.zfill(6).tolist(), futures
        )
        keep = np.flatnonzero(complete)
        if len(keep) == 0:
            continue

        x = x_all[keep]
        y = labels[keep].astype(np.uint8)
        best = best_net[keep]
        close = close_net[keep]
        symbols = frame["symbol"].astype(str).str.zfill(6).to_numpy()[keep]
        probs = model.predict(x)

        metrics.append((date, probs, y, best, close, symbols))
        samples += len(y)
        processed += 1

        if processed % 50 == 0:
            print(f"[高精度挖掘] {start}-{end} 已处理{processed}日，样本{samples}", flush=True)

    return metrics, processed, samples


def flatten(metrics):
    if not metrics:
        return tuple(np.empty(0) for _ in range(5))
    probs = np.concatenate([x[1] for x in metrics])
    y = np.concatenate([x[2] for x in metrics])
    best = np.concatenate([x[3] for x in metrics])
    close = np.concatenate([x[4] for x in metrics])
    return probs, y, best, close


def threshold_rows(metrics):
    probs, y, best, close = flatten(metrics)
    rows = []
    for threshold in HIGH_PRECISION_THRESHOLDS:
        mask = probs >= threshold
        n = int(mask.sum())
        wins = int(y[mask].sum())
        rows.append({
            "probability_threshold": threshold,
            "samples": n,
            "sample_share_pct": float(n / len(y) * 100.0) if len(y) else 0.0,
            "wins": wins,
            "hit_1pct_rate_pct": float(wins / n * 100.0) if n else None,
            "wilson_lower_pct": float(wilson_lower_bound(wins, n) * 100.0) if n else None,
            "mean_best_return_pct": float(best[mask].mean()) if n else None,
            "mean_5d_close_return_pct": float(close[mask].mean()) if n else None,
        })
    return rows


def daily_topk_rows(metrics):
    rows = []
    for k in TOP_K_PER_DAY:
        selected_y = []
        selected_best = []
        selected_close = []
        day_count = 0
        for date, probs, y, best, close, symbols in metrics:
            order = np.argsort(-probs, kind="stable")
            take = order[: min(k, len(order))]
            if len(take) == 0:
                continue
            day_count += 1
            selected_y.append(y[take])
            selected_best.append(best[take])
            selected_close.append(close[take])

        yy = np.concatenate(selected_y) if selected_y else np.empty(0, dtype=np.uint8)
        bb = np.concatenate(selected_best) if selected_best else np.empty(0)
        cc = np.concatenate(selected_close) if selected_close else np.empty(0)

        rows.append({
            "top_k_per_day": k,
            "days": day_count,
            "samples": int(len(yy)),
            "days_meet_minimum": bool(day_count >= MIN_DAILY_TOPK_DAYS),
            "hit_1pct_rate_pct": float(yy.mean() * 100.0) if len(yy) else None,
            "wilson_lower_pct": float(wilson_lower_bound(int(yy.sum()), len(yy)) * 100.0) if len(yy) else None,
            "mean_best_return_pct": float(bb.mean()) if len(bb) else None,
            "mean_5d_close_return_pct": float(cc.mean()) if len(cc) else None,
        })
    return rows


def yearly_threshold_rows(metrics, threshold):
    by_year = {}
    for date, probs, y, best, close, symbols in metrics:
        year = date[:4]
        item = by_year.setdefault(year, [[], [], [], []])
        item[0].append(probs)
        item[1].append(y)
        item[2].append(best)
        item[3].append(close)

    out = []
    for year in sorted(by_year):
        pp = np.concatenate(by_year[year][0])
        yy = np.concatenate(by_year[year][1])
        bb = np.concatenate(by_year[year][2])
        cc = np.concatenate(by_year[year][3])
        mask = pp >= threshold
        n = int(mask.sum())
        w = int(yy[mask].sum())
        out.append({
            "year": year,
            "samples": n,
            "wins": w,
            "hit_1pct_rate_pct": float(w / n * 100.0) if n else None,
            "wilson_lower_pct": float(wilson_lower_bound(w, n) * 100.0) if n else None,
            "mean_best_return_pct": float(bb[mask].mean()) if n else None,
            "mean_5d_close_return_pct": float(cc[mask].mean()) if n else None,
        })
    return out


def select_threshold(validation_rows):
    eligible = [
        row for row in validation_rows
        if row["samples"] >= MIN_SELECTION_SAMPLES
    ]
    if not eligible:
        return None

    # Highest validation precision first; Wilson lower bound breaks ties.
    return max(
        eligible,
        key=lambda r: (
            r["hit_1pct_rate_pct"] if r["hit_1pct_rate_pct"] is not None else -1.0,
            r["wilson_lower_pct"] if r["wilson_lower_pct"] is not None else -1.0,
            r["samples"],
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-05")
    ap.add_argument("--final-end", default=FINAL_END)
    ap.add_argument("--output", default=str(OUT_DIR / "high_precision_profit_mining_latest.json"))
    args = ap.parse_args()

    started = time.time()
    files = history_files()
    dates = [p.name[:10] for p in files]
    if args.start not in dates or args.final_end not in dates:
        raise ValueError("训练区间必须落在历史数据文件范围内")

    model = LogisticModel(len(FEATURES))
    train_state = FeatureState()
    # Match the existing full-market model training semantics exactly.
    train_dates = [p.name[:10] for p in files]
    train_start_i, train_end_i = train_dates.index(args.start), train_dates.index(TRAIN_END)
    train_cache = {}

    def get_train(i):
        if i not in train_cache:
            train_cache[i] = read_daily(files[i])
        return train_cache[i]

    train_samples = 0
    train_days = 0
    for i in range(max(0, train_start_i - 20), train_end_i + 1):
        date = train_dates[i]
        frame = train_state.build(get_train(i))
        if date < args.start or date > TRAIN_END:
            continue
        if i + MAX_FORWARD_SESSIONS >= len(files) or frame.empty:
            continue
        futures = [get_train(i + j) for j in range(1, MAX_FORWARD_SESSIONS + 1)]
        x_all = make_features(frame)
        labels, _, _, complete = build_targets(
            frame["symbol"].astype(str).str.zfill(6).tolist(), futures
        )
        keep = np.flatnonzero(complete)
        if len(keep) == 0:
            continue
        x = x_all[keep]
        y = labels[keep].astype(np.float64)
        model.update(x, y)
        model.update(x, y)
        train_samples += len(y)
        train_days += 1
        if train_days % 50 == 0:
            print(f"[高精度挖掘] 训练已处理{train_days}日，样本{train_samples}", flush=True)

    validation_state = FeatureState()
    validation_metrics, validation_days, validation_samples = collect_scored(
        files, VALIDATION_START, VALIDATION_END, validation_state, model
    )
    final_state = FeatureState()
    final_metrics, final_days, final_samples = collect_scored(
        files, FINAL_START, args.final_end, final_state, model
    )

    validation_thresholds = threshold_rows(validation_metrics)
    final_thresholds = threshold_rows(final_metrics)
    selected = select_threshold(validation_thresholds)
    selected_threshold = selected["probability_threshold"] if selected else None

    selected_final = None
    selected_final_yearly = []
    if selected_threshold is not None:
        selected_final = next(
            row for row in final_thresholds
            if row["probability_threshold"] == selected_threshold
        )
        selected_final_yearly = yearly_threshold_rows(final_metrics, selected_threshold)

    result = {
        "schema_version": 1,
        "status": "research_only",
        "method": "high_precision_profit_mining",
        "objective": "net_profit_at_least_1pct_within_5_sessions",
        "data_start": args.start,
        "data_end": args.final_end,
        "splits": {
            "train": [args.start, TRAIN_END],
            "validation": [VALIDATION_START, VALIDATION_END],
            "final": [FINAL_START, args.final_end],
        },
        "parameters": {
            "net_win_threshold_pct": NET_WIN_THRESHOLD_PCT,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "max_forward_sessions": MAX_FORWARD_SESSIONS,
            "high_precision_thresholds": list(HIGH_PRECISION_THRESHOLDS),
            "min_selection_samples": MIN_SELECTION_SAMPLES,
            "top_k_per_day": list(TOP_K_PER_DAY),
            "min_daily_topk_days": MIN_DAILY_TOPK_DAYS,
        },
        "model": {
            "type": "same_as_full_market_feature_training",
            "features": FEATURES,
            "intercept": float(model.b),
            "coefficients": {name: float(value) for name, value in zip(FEATURES, model.w)},
        },
        "train": {
            "processed_days": train_days,
            "samples": train_samples,
        },
        "validation": {
            "processed_days": validation_days,
            "samples": validation_samples,
            "thresholds": validation_thresholds,
            "daily_topk": daily_topk_rows(validation_metrics),
        },
        "final": {
            "processed_days": final_days,
            "samples": final_samples,
            "thresholds": final_thresholds,
            "daily_topk": daily_topk_rows(final_metrics),
            "selected_threshold": selected_final,
            "selected_threshold_yearly": selected_final_yearly,
        },
        "selected_validation_operating_point": selected,
        "audit": {
            "no_future_features": True,
            "entry_is_T_plus_1_open": True,
            "label_is_net_profit_at_least_1pct": True,
            "future_window_is_five_sessions": True,
            "threshold_selected_only_on_validation": True,
            "final_holdout_used_once_after_selection": True,
            "formal_production_changed": False,
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "status": result["status"],
        "selected_validation_operating_point": selected,
        "selected_final_operating_point": selected_final,
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
