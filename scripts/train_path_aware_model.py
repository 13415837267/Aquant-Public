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

