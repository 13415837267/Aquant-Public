"""Path-aware full-market training for the +3% short-term profit opportunity.

Research only. A positive label means the T+1 entry price can reach at least
+3% net profit within five sessions before a -3% stop from T+2 onward. T+1
hitting +3% is counted as a win because the project explicitly treats maximum
profit potential above +3% as a successful signal, even though selling on T+1
is prohibited by the A-share T+1 rule.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.short_term_research import FeatureState, history_files, read_daily

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

TRAIN_END = "2024-12-31"
VALIDATION_START = "2025-01-02"
VALIDATION_END = "2025-12-31"
FINAL_START = "2026-01-05"
FINAL_END = "2026-09-30"

NET_WIN_THRESHOLD_PCT = 3.0
STOP_LOSS_PCT = 3.0
ROUND_TRIP_COST_BPS = 10.0
MAX_FORWARD_SESSIONS = 5
MIN_SELECTION_SAMPLES = 5000

LEARNING_RATE = 0.08
L2 = 0.02
EPOCHS_PER_DAY = 2
THRESHOLDS = [0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90]


def percentile_rank(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.rank(method="average", pct=True).fillna(0.5).to_numpy(dtype=np.float64)


def make_features(frame):
    if frame.empty:
        return np.empty((0, len(FEATURES)), dtype=np.float64)
    cols = [percentile_rank(frame[name]) for name in STOCK_RANK_FEATURES]
    breadth = np.clip(
        pd.to_numeric(frame["market_breadth_pct"], errors="coerce").fillna(50).to_numpy(dtype=float) / 100.0,
        0.0, 1.0
    )
    median_ret = np.clip(
        pd.to_numeric(frame["market_median_return_pct"], errors="coerce").fillna(0).to_numpy(dtype=float) / 5.0,
        -2.0, 2.0
    )
    cols.extend([breadth, median_ret])
    return np.column_stack(cols)


def path_targets(symbols, future_days):
    """生成 +3% 机会标签，并把 3% 止盈与 3% 止损纳入路径判断。"""
    n = len(symbols)
    if len(future_days) < MAX_FORWARD_SESSIONS:
        return np.zeros(n, dtype=bool), np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=bool)

    keys = pd.Index(pd.Series(symbols, dtype="string").astype(str).str.zfill(6))
    entry = pd.to_numeric(future_days[0].set_index("symbol")["open"], errors="coerce").reindex(keys).to_numpy(dtype=float)
    target_gross_pct = NET_WIN_THRESHOLD_PCT + ROUND_TRIP_COST_BPS / 100.0
    target = entry * (1.0 + target_gross_pct / 100.0)
    stop = entry * (1.0 - STOP_LOSS_PCT / 100.0)

    opens, highs, lows, closes = [], [], [], []
    for day in future_days[:MAX_FORWARD_SESSIONS]:
        idx = day.set_index("symbol")
        opens.append(pd.to_numeric(idx["open"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        highs.append(pd.to_numeric(idx["high"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        lows.append(pd.to_numeric(idx["low"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        closes.append(pd.to_numeric(idx["close"], errors="coerce").reindex(keys).to_numpy(dtype=float))
    opens, highs, lows, closes = map(np.column_stack, (opens, highs, lows, closes))
    complete = (
        np.isfinite(entry) & (entry > 0) &
        np.isfinite(opens).all(axis=1) &
        np.isfinite(highs).all(axis=1) &
        np.isfinite(lows).all(axis=1) &
        np.isfinite(closes).all(axis=1)
    )

    target_hit = np.zeros(n, dtype=bool)
    stopped_before_target = np.zeros(n, dtype=bool)
    best_net = np.full(n, np.nan)
    close_net = np.full(n, np.nan)

    for row in np.flatnonzero(complete):
        best_net[row] = np.max(highs[row]) / entry[row] * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
        close_net[row] = closes[row, -1] / entry[row] * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
        for d in range(MAX_FORWARD_SESSIONS):
            # T+1 允许观察是否达到 +3%，但禁止卖出；止损从 T+2 才可执行。
            if highs[row, d] >= target[row]:
                target_hit[row] = True
                break
            if d >= 1 and lows[row, d] <= stop[row]:
                stopped_before_target[row] = True
                break

    # +3% 机会优先：一旦 T+1~T+5 触达目标即为正样本；若先触发 T+2~T+5 止损则为负样本。
    labels = target_hit & ~stopped_before_target
    return labels, best_net, close_net, complete



class LogisticModel:
    def __init__(self, n_features):
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

    def update(self, x, y):
        if len(y) == 0:
            return
        z = np.clip(x @ self.w + self.b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        self.w -= LEARNING_RATE * ((x.T @ err) / len(y) + L2 * self.w)
        self.b -= LEARNING_RATE * float(err.mean())

    def predict(self, x):
        z = np.clip(x @ self.w + self.b, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z))


def collect(files, start, end, state, model=None, train=False):
    dates = [p.name[:10] for p in files]
    start_i, end_i = dates.index(start), dates.index(end)
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = read_daily(files[i])
        return cache[i]

    metrics = []
    samples = 0
    processed = 0

    for i in range(max(0, start_i - 20), end_i + 1):
        date = dates[i]
        frame = state.build(get(i))
        if date < start or date > end:
            continue
        if i + MAX_FORWARD_SESSIONS >= len(files) or frame.empty:
            continue

        futures = [get(i + j) for j in range(1, MAX_FORWARD_SESSIONS + 1)]
        x_all = make_features(frame)
        y, best, close5, complete = path_targets(
            frame["symbol"].astype(str).str.zfill(6).tolist(), futures
        )
        keep = np.flatnonzero(complete)
        if len(keep) == 0:
            continue

        x = x_all[keep]
        yy = y[keep].astype(np.float64)
        bb = best[keep]
        cc = close5[keep]
        samples += len(yy)
        processed += 1

        if train:
            for _ in range(EPOCHS_PER_DAY):
                model.update(x, yy)
        else:
            metrics.append((date, model.predict(x), yy, bb, cc))

        if processed % 50 == 0:
            print(f"[路径模型] {start}-{end} 已处理{processed}日，样本{samples}", flush=True)

    return metrics, processed, samples


def summarize(metrics):
    if not metrics:
        return {"samples": 0, "thresholds": [], "yearly": []}

    probs = np.concatenate([m[1] for m in metrics])
    y = np.concatenate([m[2] for m in metrics])
    best = np.concatenate([m[3] for m in metrics])
    close5 = np.concatenate([m[4] for m in metrics])

    thresholds = []
    for t in THRESHOLDS:
        mask = probs >= t
        n = int(mask.sum())
        thresholds.append({
            "probability_threshold": t,
            "samples": n,
            "sample_share_pct": float(n / len(y) * 100.0),
            "path_win_3pct_rate_pct": float(y[mask].mean() * 100.0) if n else None,
            "mean_best_return_pct": float(best[mask].mean()) if n else None,
            "mean_5d_close_return_pct": float(close5[mask].mean()) if n else None,
        })

    yearly_map = {}
    for date, p, yy, bb, cc in metrics:
        year = date[:4]
        item = yearly_map.setdefault(year, [[], [], [], []])
        item[0].append(p)
        item[1].append(yy)
        item[2].append(bb)
        item[3].append(cc)

    yearly = []
    for year in sorted(yearly_map):
        pp = np.concatenate(yearly_map[year][0])
        yy = np.concatenate(yearly_map[year][1])
        bb = np.concatenate(yearly_map[year][2])
        cc = np.concatenate(yearly_map[year][3])
        row = {
            "year": year,
            "samples": int(len(yy)),
            "base_path_win_3pct_rate_pct": float(yy.mean() * 100.0),
        }
        for t in (0.80, 0.85):
            mask = pp >= t
            row[f"threshold_{str(t).replace('.', '_')}_samples"] = int(mask.sum())
            row[f"threshold_{str(t).replace('.', '_')}_path_win_3pct_rate_pct"] = (
                float(yy[mask].mean() * 100.0) if mask.any() else None
            )
        yearly.append(row)

    return {
        "samples": int(len(y)),
        "base_path_win_3pct_rate_pct": float(y.mean() * 100.0),
        "base_mean_best_return_pct": float(best.mean()),
        "base_mean_5d_close_return_pct": float(close5.mean()),
        "thresholds": thresholds,
        "yearly": yearly,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-05")
    ap.add_argument("--final-end", default=FINAL_END)
    ap.add_argument("--output", default=str(OUT_DIR / "path_aware_profit_3pct_training_latest.json"))
    args = ap.parse_args()

    started = time.time()
    files = history_files()
    dates = [p.name[:10] for p in files]
    if args.start not in dates or args.final_end not in dates:
        raise ValueError("训练区间必须落在历史数据文件范围内")

    model = LogisticModel(len(FEATURES))

    train_state = FeatureState()
    _, train_days, train_samples = collect(
        files, args.start, TRAIN_END, train_state, model=model, train=True
    )
    if train_samples < MIN_SELECTION_SAMPLES:
        raise RuntimeError(f"训练样本不足: {train_samples}")

    validation_state = FeatureState()
    validation_metrics, validation_days, validation_samples = collect(
        files, VALIDATION_START, VALIDATION_END, validation_state, model=model, train=False
    )

    final_state = FeatureState()
    final_metrics, final_days, final_samples = collect(
        files, FINAL_START, args.final_end, final_state, model=model, train=False
    )

    validation = summarize(validation_metrics)
    final = summarize(final_metrics)

    selected = None
    for row in validation["thresholds"]:
        if (
            row["samples"] >= MIN_SELECTION_SAMPLES
            and row["path_win_3pct_rate_pct"] is not None
            and row["path_win_3pct_rate_pct"] >= 80.0
        ):
            selected = row
            break
    if selected is None:
        eligible = [r for r in validation["thresholds"] if r["samples"] >= MIN_SELECTION_SAMPLES]
        selected = (
            max(eligible, key=lambda r: (r["path_win_3pct_rate_pct"] or -1, r["samples"]))
            if eligible else None
        )

    result = {
        "schema_version": 2,
        "status": "research_only",
        "method": "full_market_path_aware_short_term_training",
        "objective": "net_profit_at_least_3pct_opportunity_before_3pct_stop_within_5_sessions",
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
        "take_profit_pct": NET_WIN_THRESHOLD_PCT,
        "stop_loss_pct": STOP_LOSS_PCT,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "model": {
            "type": "logistic_regression_sgd",
            "learning_rate": LEARNING_RATE,
            "l2": L2,
            "epochs_per_day": EPOCHS_PER_DAY,
            "intercept": float(model.b),
            "coefficients": {name: float(value) for name, value in zip(FEATURES, model.w)},
        },
        "train": {
            "processed_days": train_days,
            "samples": train_samples,
        },
        "validation": {
            "processed_days": validation_days,
            **validation,
        },
        "final": {
            "processed_days": final_days,
            **final,
        },
        "selected_validation_operating_point": selected,
        "audit": {
            "no_future_features": True,
            "entry_is_T_plus_1_open": True,
            "target_is_net_profit_at_least_3pct_opportunity": True,
            "take_profit_pct": NET_WIN_THRESHOLD_PCT,
            "stop_loss_gross_pct": STOP_LOSS_PCT,
            "t_plus_1_target_counts_as_win": True,
            "stop_loss_applies_from_t_plus_2": True,
            "same_day_stop_first": True,
            "formal_production_changed": False,
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
