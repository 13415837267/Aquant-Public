"""Market-regime conditioned rule mining for the strict 1% path objective.

Research only. Discovery uses 2015-2022, validation selection uses 2023-2024,
and the final holdout is 2025-2026-09-30. Formal production is never changed.

Trade label:
T+1 open entry, executable only if the entry is below the upper limit.
Within the next five sessions, net +1% must be reached before a gross -3% stop.
If both are touched in the same daily bar, stop is treated as first.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.short_term_research import FeatureState, history_files, read_daily

OUT_DIR = ROOT / "data" / "backtest"

STOCK_FEATURES = [
    "return_1d_pct", "return_3d_pct", "return_5d_pct", "return_10d_pct",
    "return_20d_pct", "overnight_1d_pct", "overnight_3d_pct",
    "overnight_5d_pct", "overnight_10d_pct", "volume_ratio_5d",
    "amount_20d", "volatility_10d_pct", "close_strength",
    "intraday_return_pct", "limit_up_5d_count", "turnover_pct", "change_pct",
]
RANK_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)

TRAIN_END = "2022-12-30"
VALIDATION_START = "2023-01-03"
VALIDATION_END = "2024-12-31"
FINAL_START = "2025-01-02"
FINAL_END = "2026-09-30"

NET_WIN_THRESHOLD_PCT = 1.0
STOP_LOSS_PCT = 3.0
ROUND_TRIP_COST_BPS = 10.0
MAX_FORWARD_SESSIONS = 5
ENTRY_LIMIT_UP_BLOCK = True

MIN_TRAIN_RULE_SAMPLES = 10_000
MIN_VALIDATION_RULE_SAMPLES = 5_000
MIN_REGIME_SAMPLES = 25_000
TOP_STOCK_ATOMICS_PER_REGIME = 10
MAX_TRAIN_PAIR_OUTPUT = 120
TOP_OUTPUT_RULES = 30
MAX_SELECTED_PER_REGIME = 3
TARGET_WIN_RATE_PCT = 80.0

BREADTH_BINS = (-np.inf, 40.0, 50.0, 60.0, 70.0, np.inf)
MEDIAN_BINS = (-np.inf, -1.0, -0.5, 0.0, 0.5, 1.0, np.inf)


def percentile_rank(series: pd.Series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce")
    return x.rank(method="average", pct=True).fillna(0.5).to_numpy(dtype=np.float64)


def make_stock_features(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([percentile_rank(frame[name]) for name in STOCK_FEATURES])


def build_regimes() -> list[dict]:
    out = []
    for bi in range(len(BREADTH_BINS) - 1):
        for mi in range(len(MEDIAN_BINS) - 1):
            out.append({
                "regime_index": len(out),
                "breadth_low": None if not np.isfinite(BREADTH_BINS[bi]) else float(BREADTH_BINS[bi]),
                "breadth_high": None if not np.isfinite(BREADTH_BINS[bi + 1]) else float(BREADTH_BINS[bi + 1]),
                "median_low": None if not np.isfinite(MEDIAN_BINS[mi]) else float(MEDIAN_BINS[mi]),
                "median_high": None if not np.isfinite(MEDIAN_BINS[mi + 1]) else float(MEDIAN_BINS[mi + 1]),
            })
    return out


REGIMES = build_regimes()


def regime_masks(frame: pd.DataFrame) -> np.ndarray:
    breadth = pd.to_numeric(frame["market_breadth_pct"], errors="coerce").fillna(50.0).to_numpy(dtype=float)
    median = pd.to_numeric(frame["market_median_return_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    result = []
    for rule in REGIMES:
        mask = np.ones(len(frame), dtype=bool)
        if rule["breadth_low"] is not None:
            mask &= breadth >= rule["breadth_low"]
        if rule["breadth_high"] is not None:
            mask &= breadth < rule["breadth_high"]
        if rule["median_low"] is not None:
            mask &= median >= rule["median_low"]
        if rule["median_high"] is not None:
            mask &= median < rule["median_high"]
        result.append(mask)
    return np.column_stack(result).astype(np.uint8)


def stock_rule_defs() -> list[dict]:
    rules = []
    for feature_idx, feature in enumerate(STOCK_FEATURES):
        for threshold in RANK_THRESHOLDS:
            rules.append({
                "rule_index": len(rules),
                "feature": feature,
                "feature_index": feature_idx,
                "op": ">=",
                "threshold": threshold,
            })
            rules.append({
                "rule_index": len(rules),
                "feature": feature,
                "feature_index": feature_idx,
                "op": "<=",
                "threshold": threshold,
            })
    return rules


RULES = stock_rule_defs()


def rule_label(rule: dict) -> str:
    return f'{rule["feature"]}{rule["op"]}{rule["threshold"]:g}'


def wilson_lower_bound(wins: int, samples: int, z: float = 1.96) -> float:
    if samples <= 0:
        return 0.0
    p = wins / samples
    denom = 1.0 + z * z / samples
    center = p + z * z / (2.0 * samples)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * samples)) / samples)
    return (center - spread) / denom


def empty_target_arrays(n: int):
    return (
        np.zeros(n, dtype=bool),
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.zeros(n, dtype=bool),
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.full(n, np.nan),
    )


def path_targets(symbols: list[str], future_days: list[pd.DataFrame]):
    n = len(symbols)
    if len(future_days) < MAX_FORWARD_SESSIONS:
        return empty_target_arrays(n)

    keys = pd.Index(pd.Series(symbols, dtype="string").astype(str).str.zfill(6))
    first = future_days[0].set_index("symbol")
    entry = pd.to_numeric(first["open"], errors="coerce").reindex(keys).to_numpy(dtype=float)

    if ENTRY_LIMIT_UP_BLOCK and "high_limit" in first.columns:
        high_limit = pd.to_numeric(first["high_limit"], errors="coerce").reindex(keys).to_numpy(dtype=float)
        executable = ~np.isfinite(high_limit) | (entry < high_limit * (1.0 - 1e-6))
    else:
        executable = np.ones(n, dtype=bool)

    target_gross_pct = NET_WIN_THRESHOLD_PCT + ROUND_TRIP_COST_BPS / 100.0
    target = entry * (1.0 + target_gross_pct / 100.0)
    stop = entry * (1.0 - STOP_LOSS_PCT / 100.0)

    opens, highs, lows, closes = [], [], [], []
    for day in future_days[:MAX_FORWARD_SESSIONS]:
        indexed = day.set_index("symbol")
        opens.append(pd.to_numeric(indexed["open"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        highs.append(pd.to_numeric(indexed["high"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        lows.append(pd.to_numeric(indexed["low"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        closes.append(pd.to_numeric(indexed["close"], errors="coerce").reindex(keys).to_numpy(dtype=float))

    opens = np.column_stack(opens)
    highs = np.column_stack(highs)
    lows = np.column_stack(lows)
    closes = np.column_stack(closes)
    complete = (
        np.isfinite(entry) & (entry > 0) & executable
        & np.isfinite(opens).all(axis=1)
        & np.isfinite(highs).all(axis=1)
        & np.isfinite(lows).all(axis=1)
        & np.isfinite(closes).all(axis=1)
    )

    win = np.zeros(n, dtype=bool)
    best_net = np.full(n, np.nan)
    close_net = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    target_day = np.full(n, np.nan)
    stop_day = np.full(n, np.nan)

    for row in np.flatnonzero(complete):
        best_net[row] = np.max(highs[row]) / entry[row] * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
        close_net[row] = closes[row, -1] / entry[row] * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
        mae[row] = np.min(lows[row]) / entry[row] * 100.0 - 100.0
        mfe[row] = np.max(highs[row]) / entry[row] * 100.0 - 100.0

        for d in range(MAX_FORWARD_SESSIONS):
            day_open, day_high, day_low = opens[row, d], highs[row, d], lows[row, d]
            if day_open <= stop[row] or day_low <= stop[row]:
                stop_day[row] = d + 1
                break
            if day_high >= target[row]:
                target_day[row] = d + 1
                win[row] = True
                break

    return win, best_net, close_net, complete, mae, mfe, target_day, stop_day


def make_stock_masks(x: np.ndarray) -> np.ndarray:
    masks = np.empty((len(x), len(RULES)), dtype=np.uint8)
    for pos, rule in enumerate(RULES):
        values = x[:, rule["feature_index"]]
        masks[:, pos] = (
            values >= rule["threshold"] if rule["op"] == ">=" else values <= rule["threshold"]
        ).astype(np.uint8)
    return masks


def new_acc(shape) -> dict:
    return {
        "samples": np.zeros(shape, dtype=np.int64),
        "wins": np.zeros(shape, dtype=np.int64),
        "best_sum": np.zeros(shape, dtype=np.float64),
        "close_sum": np.zeros(shape, dtype=np.float64),
        "mae_sum": np.zeros(shape, dtype=np.float64),
        "mfe_sum": np.zeros(shape, dtype=np.float64),
        "target_day_sum": np.zeros(shape, dtype=np.float64),
        "target_day_count": np.zeros(shape, dtype=np.int64),
        "stop_day_sum": np.zeros(shape, dtype=np.float64),
        "stop_day_count": np.zeros(shape, dtype=np.int64),
    }


def int_dot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Integer matrix products must use int64; uint8 matmul silently wraps."""
    return left.astype(np.int64).T @ right.astype(np.int64)


def int_vector_dot(left: np.ndarray, right: np.ndarray) -> int:
    """Integer vector products must use int64 to avoid overflow on large samples."""
    return int(np.dot(left.astype(np.int64), right.astype(np.int64)))


def accumulate_matrix(acc, masks, y, best, close, mae, mfe, target_day, stop_day):
    valid_target = np.isfinite(target_day) & (target_day > 0)
    valid_stop = np.isfinite(stop_day) & (stop_day > 0)
    target_clean = np.where(valid_target, target_day, 0.0)
    stop_clean = np.where(valid_stop, stop_day, 0.0)

    acc["samples"] += masks.T @ np.ones(len(y), dtype=np.int64)
    acc["wins"] += masks.T @ y.astype(np.int64)
    acc["best_sum"] += masks.T @ best
    acc["close_sum"] += masks.T @ close
    acc["mae_sum"] += masks.T @ mae
    acc["mfe_sum"] += masks.T @ mfe
    acc["target_day_sum"] += masks.T @ target_clean
    acc["target_day_count"] += masks.T @ valid_target.astype(np.int64)
    acc["stop_day_sum"] += masks.T @ stop_clean
    acc["stop_day_count"] += masks.T @ valid_stop.astype(np.int64)


def row_from_acc(rule_type, rule_payload, acc, pos, regime=None):
    n = int(acc["samples"][pos])
    w = int(acc["wins"][pos])
    row = {
        "rule_type": rule_type,
        "samples": n,
        "wins": w,
        "path_win_1pct_rate_pct": w / n * 100.0 if n else None,
        "wilson_lower_pct": wilson_lower_bound(w, n) * 100.0 if n else None,
        "mean_best_return_pct": acc["best_sum"][pos] / n if n else None,
        "mean_5d_close_return_pct": acc["close_sum"][pos] / n if n else None,
        "mean_mae_pct": acc["mae_sum"][pos] / n if n else None,
        "mean_mfe_pct": acc["mfe_sum"][pos] / n if n else None,
        "target_day_mean": (
            acc["target_day_sum"][pos] / acc["target_day_count"][pos]
            if acc["target_day_count"][pos] else None
        ),
        "stop_day_mean": (
            acc["stop_day_sum"][pos] / acc["stop_day_count"][pos]
            if acc["stop_day_count"][pos] else None
        ),
    }
    row.update(rule_payload)
    if regime is not None:
        row["regime"] = regime
    return row


def iter_split(files, start, end, state):
    dates = [p.name[:10] for p in files]
    start_i, end_i = dates.index(start), dates.index(end)
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = read_daily(files[i])
        while len(cache) > MAX_FORWARD_SESSIONS + 1:
            del cache[next(iter(cache))]
        return cache[i]

    for i in range(max(0, start_i - 20), end_i + 1):
        date = dates[i]
        frame = state.build(get(i))
        if date < start or date > end:
            continue
        if frame.empty or i + MAX_FORWARD_SESSIONS >= len(files):
            continue
        futures = [get(i + j) for j in range(1, MAX_FORWARD_SESSIONS + 1)]
        x = make_stock_features(frame)
        targets = path_targets(
            frame["symbol"].astype(str).str.zfill(6).tolist(), futures
        )
        keep = np.flatnonzero(targets[3])
        if not len(keep):
            continue
        y, best, close, _, mae, mfe, target_day, stop_day = targets
        yield (
            date,
            x[keep],
            y[keep],
            best[keep],
            close[keep],
            mae[keep],
            mfe[keep],
            target_day[keep],
            stop_day[keep],
            regime_masks(frame)[keep],
        )


def summarize_atomic_by_regime(files, start, end):
    state = FeatureState()
    acc = new_acc((len(REGIMES), len(RULES)))
    regime_counts = np.zeros(len(REGIMES), dtype=np.int64)
    processed = samples = 0

    for date, x, y, best, close, mae, mfe, target_day, stop_day, rmask in iter_split(files, start, end, state):
        stock_masks = make_stock_masks(x)
        r_counts = int_dot(stock_masks, rmask)
        r_wins = int_dot(stock_masks, rmask * y[:, None].astype(np.int64))
        r_best = stock_masks.T @ (rmask * best[:, None])
        r_close = stock_masks.T @ (rmask * close[:, None])
        r_mae = stock_masks.T @ (rmask * mae[:, None])
        r_mfe = stock_masks.T @ (rmask * mfe[:, None])
        valid_target = (np.isfinite(target_day) & (target_day > 0)).astype(np.uint8)
        valid_stop = (np.isfinite(stop_day) & (stop_day > 0)).astype(np.uint8)
        r_target = stock_masks.T @ (rmask * np.where(valid_target[:, None], target_day[:, None], 0.0))
        r_target_count = int_dot(stock_masks, rmask * valid_target[:, None].astype(np.int64))
        r_stop = stock_masks.T @ (rmask * np.where(valid_stop[:, None], stop_day[:, None], 0.0))
        r_stop_count = int_dot(stock_masks, rmask * valid_stop[:, None].astype(np.int64))

        acc["samples"] += r_counts.T
        acc["wins"] += r_wins.T
        acc["best_sum"] += r_best.T
        acc["close_sum"] += r_close.T
        acc["mae_sum"] += r_mae.T
        acc["mfe_sum"] += r_mfe.T
        acc["target_day_sum"] += r_target.T
        acc["target_day_count"] += r_target_count.T
        acc["stop_day_sum"] += r_stop.T
        acc["stop_day_count"] += r_stop_count.T
        regime_counts += rmask.sum(axis=0, dtype=np.int64)

        processed += 1
        samples += len(y)
        if processed % 50 == 0:
            print(f"[状态挖掘] 原子条件 {start}-{end} 已处理{processed}日，样本{samples}", flush=True)

    candidates = {}
    for ridx, regime in enumerate(REGIMES):
        rows = []
        for rule_idx, rule in enumerate(RULES):
            n = int(acc["samples"][ridx, rule_idx])
            if n < max(MIN_TRAIN_RULE_SAMPLES, MIN_REGIME_SAMPLES):
                continue
            row = row_from_acc(
                "regime_stock_atomic",
                {
                    "rule_index": rule_idx,
                    "stock_rule": rule,
                    "stock_rule_label": rule_label(rule),
                    "rule_label": f'regime_{ridx} | {rule_label(rule)}',
                },
                {k: v[ridx] for k, v in acc.items()},
                rule_idx,
                regime,
            )
            rows.append(row)
        rows.sort(key=lambda r: (r["wilson_lower_pct"], r["path_win_1pct_rate_pct"], r["samples"]), reverse=True)
        candidates[ridx] = rows[:TOP_STOCK_ATOMICS_PER_REGIME]
    return candidates, regime_counts, processed, samples


def pair_specs_by_regime(candidates):
    all_specs = []
    for ridx, rows in candidates.items():
        indices = [r["rule_index"] for r in rows]
        for pos, left in enumerate(indices):
            for right in indices[pos + 1:]:
                if RULES[left]["feature"] == RULES[right]["feature"]:
                    continue
                all_specs.append((ridx, left, right))
    return all_specs


def scan_pairs(files, start, end, specs):
    if not specs:
        return [], None, 0, 0

    groups = {}
    for spec in specs:
        groups.setdefault(spec[0], []).append(spec)

    accumulators = {}
    for ridx, group in groups.items():
        accumulators[ridx] = new_acc(len(group))

    state = FeatureState()
    processed = samples = 0

    for date, x, y, best, close, mae, mfe, target_day, stop_day, rmask in iter_split(files, start, end, state):
        stock_masks = make_stock_masks(x)
        for ridx, group in groups.items():
            rm = rmask[:, ridx].astype(np.uint8)
            if not rm.any():
                continue
            unique = sorted({idx for _, left, right in group for idx in (left, right)})
            local = stock_masks[:, unique] * rm[:, None]
            pos_map = {idx: pos for pos, idx in enumerate(unique)}
            a = len(group)
            acc = accumulators[ridx]

            pair_counts = np.zeros(a, dtype=np.int64)
            pair_wins = np.zeros(a, dtype=np.int64)
            pair_best = np.zeros(a, dtype=np.float64)
            pair_close = np.zeros(a, dtype=np.float64)
            pair_mae = np.zeros(a, dtype=np.float64)
            pair_mfe = np.zeros(a, dtype=np.float64)
            pair_target = np.zeros(a, dtype=np.float64)
            pair_target_count = np.zeros(a, dtype=np.int64)
            pair_stop = np.zeros(a, dtype=np.float64)
            pair_stop_count = np.zeros(a, dtype=np.int64)

            valid_target = np.isfinite(target_day) & (target_day > 0)
            valid_stop = np.isfinite(stop_day) & (stop_day > 0)
            target_clean = np.where(valid_target, target_day, 0.0)
            stop_clean = np.where(valid_stop, stop_day, 0.0)

            for p, (_, left, right) in enumerate(group):
                mask = (local[:, pos_map[left]] & local[:, pos_map[right]]).astype(np.uint8)
                pair_counts[p] = int(mask.sum())
                pair_wins[p] = int_vector_dot(mask, y.astype(np.uint8))
                pair_best[p] = float(np.dot(mask, best))
                pair_close[p] = float(np.dot(mask, close))
                pair_mae[p] = float(np.dot(mask, mae))
                pair_mfe[p] = float(np.dot(mask, mfe))
                pair_target[p] = float(np.dot(mask, target_clean))
                pair_target_count[p] = int_vector_dot(mask, valid_target.astype(np.uint8))
                pair_stop[p] = float(np.dot(mask, stop_clean))
                pair_stop_count[p] = int_vector_dot(mask, valid_stop.astype(np.uint8))

            acc["samples"] += pair_counts
            acc["wins"] += pair_wins
            acc["best_sum"] += pair_best
            acc["close_sum"] += pair_close
            acc["mae_sum"] += pair_mae
            acc["mfe_sum"] += pair_mfe
            acc["target_day_sum"] += pair_target
            acc["target_day_count"] += pair_target_count
            acc["stop_day_sum"] += pair_stop
            acc["stop_day_count"] += pair_stop_count

        processed += 1
        samples += len(y)
        if processed % 50 == 0:
            print(f"[状态挖掘] 双个股条件 {start}-{end} 已处理{processed}日，样本{samples}", flush=True)

    rows = []
    for ridx, group in groups.items():
        acc = accumulators[ridx]
        for pos, (_, left, right) in enumerate(group):
            if int(acc["samples"][pos]) < MIN_TRAIN_RULE_SAMPLES:
                continue
            left_rule, right_rule = RULES[left], RULES[right]
            row = row_from_acc(
                "regime_stock_pair",
                {
                    "rule_indices": [left, right],
                    "stock_rules": [left_rule, right_rule],
                    "stock_rule_labels": [rule_label(left_rule), rule_label(right_rule)],
                    "rule_label": f'regime_{ridx} | {rule_label(left_rule)} AND {rule_label(right_rule)}',
                },
                acc,
                pos,
                REGIMES[ridx],
            )
            rows.append(row)

    rows.sort(key=lambda r: (r["wilson_lower_pct"], r["path_win_1pct_rate_pct"], r["samples"]), reverse=True)
    return rows[:MAX_TRAIN_PAIR_OUTPUT], None, processed, samples


def evaluate_specs(files, start, end, specs):
    if not specs:
        return []

    groups = {}
    for pos, spec in enumerate(specs):
        # specs are (rule_type, regime_index, left_rule, right_rule_or_none).
        groups.setdefault(spec[1], []).append((pos, spec))

    accumulators = {}
    for ridx, group in groups.items():
        accumulators[ridx] = new_acc(len(group))

    state = FeatureState()
    processed = samples = 0

    for date, x, y, best, close, mae, mfe, target_day, stop_day, rmask in iter_split(files, start, end, state):
        stock_masks = make_stock_masks(x)
        for ridx, group in groups.items():
            rm = rmask[:, ridx].astype(np.uint8)
            if not rm.any():
                continue
            unique = sorted({
                idx
                for _, spec in group
                for idx in ((spec[1],) if spec[2] is None else (spec[1], spec[2]))
            })
            local = stock_masks[:, unique] * rm[:, None]
            pos_map = {idx: pos for pos, idx in enumerate(unique)}
            acc = accumulators[ridx]
            valid_target = np.isfinite(target_day) & (target_day > 0)
            valid_stop = np.isfinite(stop_day) & (stop_day > 0)
            target_clean = np.where(valid_target, target_day, 0.0)
            stop_clean = np.where(valid_stop, stop_day, 0.0)

            for local_pos, (_, spec) in enumerate(group):
                left, right = spec[1], spec[2]
                if right is None:
                    mask = local[:, pos_map[left]]
                else:
                    mask = local[:, pos_map[left]] & local[:, pos_map[right]]
                mask = mask.astype(np.uint8)
                acc["samples"][local_pos] += int(mask.sum())
                acc["wins"][local_pos] += int(np.dot(mask, y.astype(np.uint8)))
                acc["best_sum"][local_pos] += float(np.dot(mask, best))
                acc["close_sum"][local_pos] += float(np.dot(mask, close))
                acc["mae_sum"][local_pos] += float(np.dot(mask, mae))
                acc["mfe_sum"][local_pos] += float(np.dot(mask, mfe))
                acc["target_day_sum"][local_pos] += float(np.dot(mask, target_clean))
                acc["target_day_count"][local_pos] += int(np.dot(mask, valid_target.astype(np.uint8)))
                acc["stop_day_sum"][local_pos] += float(np.dot(mask, stop_clean))
                acc["stop_day_count"][local_pos] += int(np.dot(mask, valid_stop.astype(np.uint8)))

        processed += 1
        samples += len(y)
        if processed % 50 == 0:
            print(f"[状态挖掘] 评估 {start}-{end} 已处理{processed}日，样本{samples}", flush=True)

    rows = []
    for ridx, group in groups.items():
        acc = accumulators[ridx]
        for local_pos, (global_pos, spec) in enumerate(group):
            rule_type, _, left, right = spec
            if rule_type == "regime_stock_atomic":
                rule = RULES[left]
                payload = {
                    "rule_index": left,
                    "stock_rule": rule,
                    "stock_rule_label": rule_label(rule),
                    "rule_label": f'regime_{ridx} | {rule_label(rule)}',
                }
            else:
                left_rule, right_rule = RULES[left], RULES[right]
                payload = {
                    "rule_indices": [left, right],
                    "stock_rules": [left_rule, right_rule],
                    "stock_rule_labels": [rule_label(left_rule), rule_label(right_rule)],
                    "rule_label": f'regime_{ridx} | {rule_label(left_rule)} AND {rule_label(right_rule)}',
                }
            rows.append(row_from_acc(rule_type, payload, acc, local_pos, REGIMES[ridx]))
    rows.sort(key=lambda r: (r["path_win_1pct_rate_pct"] or -1, r["wilson_lower_pct"] or -1, r["samples"]), reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-05")
    ap.add_argument("--final-end", default=FINAL_END)
    ap.add_argument("--output", default=str(OUT_DIR / "path_regime_rule_mining_latest.json"))
    args = ap.parse_args()

    started = time.time()
    files = history_files()
    dates = [p.name[:10] for p in files]
    if args.start not in dates or args.final_end not in dates:
        raise ValueError("训练区间必须落在历史数据范围内")

    train_atomic, regime_counts, train_days, train_samples = summarize_atomic_by_regime(
        files, args.start, TRAIN_END
    )
    train_pair_rows, _, pair_days, pair_samples = scan_pairs(
        files, args.start, TRAIN_END, pair_specs_by_regime(train_atomic)
    )

    train_atomic_specs = [
        ("regime_stock_atomic", ridx, row["rule_index"])
        for ridx, rows in train_atomic.items()
        for row in rows
    ]
    train_pair_specs = [
        ("regime_stock_pair", row["regime"]["regime_index"], row["rule_indices"][0], row["rule_indices"][1])
        for row in train_pair_rows
    ]

    validation_specs = []
    validation_specs.extend(("regime_stock_atomic", ridx, row["rule_index"], None)
                            for ridx, rows in train_atomic.items() for row in rows)
    validation_specs.extend(("regime_stock_pair", ridx, left, right)
                            for _, ridx, left, right in train_pair_specs)

    validation_rows = evaluate_specs(files, VALIDATION_START, VALIDATION_END, validation_specs)
    validation_rows = [
        row for row in validation_rows
        if row["samples"] >= MIN_VALIDATION_RULE_SAMPLES
    ]
    validation_rows.sort(
        key=lambda r: (r["path_win_1pct_rate_pct"] or -1, r["wilson_lower_pct"] or -1, r["samples"]),
        reverse=True,
    )

    selected = []
    regime_selected = {}
    for row in validation_rows:
        ridx = row["regime"]["regime_index"]
        if regime_selected.get(ridx, 0) >= MAX_SELECTED_PER_REGIME:
            continue
        selected.append(row)
        regime_selected[ridx] = regime_selected.get(ridx, 0) + 1
        if len(selected) >= TOP_OUTPUT_RULES:
            break

    final_specs = []
    for row in selected:
        ridx = row["regime"]["regime_index"]
        if row["rule_type"] == "regime_stock_atomic":
            final_specs.append(("regime_stock_atomic", ridx, row["rule_index"], None))
        else:
            final_specs.append(("regime_stock_pair", ridx, row["rule_indices"][0], row["rule_indices"][1]))

    final_rows = evaluate_specs(files, FINAL_START, args.final_end, final_specs)
    final_map = {}
    for row in final_rows:
        key = (
            row["rule_type"],
            row["regime"]["regime_index"],
            row["rule_index"] if row["rule_type"] == "regime_stock_atomic" else tuple(row["rule_indices"]),
        )
        final_map[key] = row

    train_maps = {}
    for row in train_atomic_specs:
        train_maps[("regime_stock_atomic", row[1], row[2])] = next(
            r for r in train_atomic[row[1]] if r["rule_index"] == row[2]
        )
    for row in train_pair_rows:
        train_maps[("regime_stock_pair", row["regime"]["regime_index"], tuple(row["rule_indices"]))] = row

    selected_output = []
    for row in selected:
        ridx = row["regime"]["regime_index"]
        if row["rule_type"] == "regime_stock_atomic":
            key = ("regime_stock_atomic", ridx, row["rule_index"])
        else:
            key = ("regime_stock_pair", ridx, tuple(row["rule_indices"]))
        selected_output.append({
            "rule_type": row["rule_type"],
            "regime": row["regime"],
            "rule_label": row["rule_label"],
            "train": train_maps.get(key),
            "validation": row,
            "final": final_map.get(key),
            "validation_target_80pct": row["path_win_1pct_rate_pct"] >= TARGET_WIN_RATE_PCT,
        })

    result = {
        "schema_version": 1,
        "status": "research_only",
        "method": "market_regime_conditioned_three_condition_rule_mining",
        "objective": "net_profit_at_least_1pct_before_3pct_stop_within_5_sessions",
        "entry_rule": "T_plus_1_open; limit_up_entry_blocked",
        "data_start": args.start,
        "data_end": args.final_end,
        "splits": {
            "train": [args.start, TRAIN_END],
            "validation": [VALIDATION_START, VALIDATION_END],
            "final": [FINAL_START, args.final_end],
        },
        "parameters": {
            "net_win_threshold_pct": NET_WIN_THRESHOLD_PCT,
            "stop_loss_pct": STOP_LOSS_PCT,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "max_forward_sessions": MAX_FORWARD_SESSIONS,
            "min_train_rule_samples": MIN_TRAIN_RULE_SAMPLES,
            "min_validation_rule_samples": MIN_VALIDATION_RULE_SAMPLES,
            "min_regime_samples": MIN_REGIME_SAMPLES,
            "top_stock_atomics_per_regime": TOP_STOCK_ATOMICS_PER_REGIME,
            "max_train_pair_output": MAX_TRAIN_PAIR_OUTPUT,
            "top_output_rules": TOP_OUTPUT_RULES,
            "max_selected_per_regime": MAX_SELECTED_PER_REGIME,
            "target_win_rate_pct": TARGET_WIN_RATE_PCT,
            "rank_thresholds": list(RANK_THRESHOLDS),
            "breadth_bins": list(BREADTH_BINS),
            "median_bins": list(MEDIAN_BINS),
        },
        "regimes": REGIMES,
        "train": {
            "samples": train_samples,
            "processed_days": train_days,
            "regime_sample_counts": regime_counts.tolist(),
        },
        "train_top_atomic_by_regime": train_atomic,
        "train_top_pairs": train_pair_rows,
        "validation_top_rules": selected,
        "validation_qualified_rules_ge_80pct": [
            row for row in selected if row["path_win_1pct_rate_pct"] >= TARGET_WIN_RATE_PCT
        ],
        "selected_rules": selected_output,
        "final_qualified_rules_ge_80pct": [
            item["final"]
            for item in selected_output
            if item["final"] is not None
            and item["final"]["samples"] >= MIN_VALIDATION_RULE_SAMPLES
            and item["final"]["path_win_1pct_rate_pct"] >= TARGET_WIN_RATE_PCT
        ],
        "audit": {
            "no_future_features": True,
            "path_label_target_first": True,
            "same_day_stop_first": True,
            "entry_is_T_plus_1_open": True,
            "limit_up_entry_blocked": ENTRY_LIMIT_UP_BLOCK,
            "final_holdout_used_only_after_validation_selection": True,
            "rule_discovery_split": "train",
            "rule_selection_split": "validation",
            "final_holdout_split": "final",
            "same_feature_pairs_blocked": True,
            "formal_production_changed": False,
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        __import__("json").dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(__import__("json").dumps({
        "status": result["status"],
        "train_samples": train_samples,
        "validation_samples": sum(r["samples"] for r in selected),
        "selected_rules": len(selected),
        "validation_ge_80pct": len(result["validation_qualified_rules_ge_80pct"]),
        "final_ge_80pct": len(result["final_qualified_rules_ge_80pct"]),
        "top_validation_rule": selected[0]["rule_label"] if selected else None,
        "top_validation_rate": selected[0]["path_win_1pct_rate_pct"] if selected else None,
        "top_final_rate": selected_output[0]["final"]["path_win_1pct_rate_pct"] if selected_output and selected_output[0]["final"] else None,
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
